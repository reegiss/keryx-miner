#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;

use clap::{App, FromArgMatches, IntoApp};
use keryx_miner::PluginManager;
use log::{error, info};
use rand::{thread_rng, RngCore};
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use crate::cli::Opt;
use crate::client::grpc::KeryxdHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod cli;
mod client;
mod escrow;
mod ipfs;
mod keryxd_messages;
mod miner;
mod pow;
mod target;
mod watch;

const WHITELIST: [&str; 4] = ["libkeryxcuda", "libkeryxopencl", "keryxcuda", "keryxopencl"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
    // include!("protowire.rs"); // FIXME: https://github.com/intellij-rust/intellij-rust/issues/6579
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

/// Query GPU stats via nvidia-smi and warn on power/VRAM issues for the selected model tier.
///
/// VRAM requirements (GGUF weights only, not counting CUDA workspace):
///   TinyLlama-1.1B  →  ~1.5 GB
///   DeepSeek-R1-8B  →  ~5 GB
///   DeepSeek-R1-32B → ~19 GB   (requires ≥24 GB card)
///   LLaMA-3.3-70B   → ~28 GB   (requires ≥40 GB card — does NOT fit on RTX 3090)
///
/// Power thresholds empirically derived: Xid 32 observed at ≤300W on RTX 3090 with 32B GGUF.
fn check_gpu_power_limit(needs_high: bool, needs_very_high: bool) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.limit,power.max_limit,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let (current_w, vram_mb) = match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let mut parts = s.trim().split(',');
            let cur: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
            let _max: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
            let vram: u64 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
            (cur as u32, vram)
        }
        _ => return,
    };

    // VRAM check for 70B: 28 GB weights + ~2.5 GB KV cache → requires ≥32 GB card (RTX 5090+).
    // On 24 GB cards (RTX 3090 / 4090), loading will OOM and crash the miner.
    if needs_very_high && vram_mb < 30_000 {
        log::error!(
            "✗  LLaMA-3.3-70B requires ≥32 GB VRAM (RTX 5090 or equivalent) — \
             this GPU has only {} MB ({} GB).",
            vram_mb,
            vram_mb / 1024
        );
        log::error!(
            "   Use --high (DeepSeek-R1-32B, fits in 24 GB) or --light (TinyLlama only)."
        );
        // Non-fatal: let candle fail with its own OOM so the miner logs the actual error.
    }

    let model_label = if needs_very_high {
        "LLaMA-3.3-70B (--very-high)"
    } else if needs_high {
        "DeepSeek-R1-32B (--high)"
    } else {
        "DeepSeek-R1-8B (default)"
    };
    log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
}

async fn get_client(
    keryxd_address: String,
    mining_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    escrow_privkey: Option<String>,
    escrow_state_file: String,
    ipfs_url: String,
) -> Result<Box<dyn Client + 'static>, Error> {
    if keryxd_address.starts_with("stratum+tcp://") {
        let (_schema, address) = keryxd_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
        )
        .await?)
    } else if keryxd_address.starts_with("grpc://") {
        Ok(KeryxdHandler::connect(
            keryxd_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            escrow_privkey,
            escrow_state_file,
            ipfs_url,
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
) -> Result<(), Error> {
    let ipfs_url = opt.ipfs_url.clone();
    tokio::task::spawn_blocking(move || crate::ipfs::ensure_daemon(&ipfs_url)).await.ok();

    let mut client = get_client(
        opt.keryxd_address.clone(),
        opt.mining_address.clone().unwrap_or_default(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
        escrow_privkey,
        opt.escrow_state_file.clone(),
        opt.ipfs_url.clone(),
    )
    .await?;

    if opt.devfund_percent > 0 {
        client.add_devfund(opt.devfund_address.clone(), opt.devfund_percent);
    }
    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    client.listen(&mut miner_manager).await?;
    drop(miner_manager);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Getting the parent directory
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    let (app, mut plugin_manager): (App, PluginManager) = keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    let matches = app.get_matches();

    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    opt.process()?;
    env_logger::builder().filter_level(opt.log_level()).parse_default_env().init();
    info!("=================================================================================");
    info!("                 Keryx-Miner GPU {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address.as_deref().unwrap_or("(recovery mode)"));
    info!("=================================================================================");

    // Recovery mode: rebuild escrow_state.json from the Keryx public API, then exit.
    // Must run before escrow key loading to avoid creating a new random key on disk.
    // Uses escrow.key to derive the pubkey — only claimable UTXOs are returned.
    if opt.recover_escrow {
        let escrow_privkey = match escrow::load_key(&opt.escrow_key_file) {
            Ok(k) => k,
            Err(e) => {
                error!("{}", e);
                return Err(e.into());
            }
        };
        let pubkey_hex = match escrow::pubkey_hex_from_privkey(&escrow_privkey) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to derive pubkey from escrow key: {}", e);
                return Err(e.into());
            }
        };
        let url = format!("{}/api/v1/escrow/{}", opt.recover_escrow_api.trim_end_matches('/'), pubkey_hex);
        info!("Querying escrow UTXOs from {}", url);

        #[derive(serde::Deserialize)]
        struct ApiEscrowEntry {
            coinbase_txid: String,
            block_hash: String,
            confirm_daa: i64,
            amount_sompi: i64,
            output_index: i64,
        }

        let url_clone = url.clone();
        let api_entries: Vec<ApiEscrowEntry> = tokio::task::spawn_blocking(move || {
            let response = ureq::get(&url_clone)
                .call()
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            serde_json::from_reader::<_, Vec<ApiEscrowEntry>>(response.into_reader())
                .map_err(|e| format!("JSON parse error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {}", e))??;

        let entries: Vec<escrow::EscrowEntry> = api_entries
            .into_iter()
            .map(|a| escrow::EscrowEntry {
                coinbase_txid: a.coinbase_txid,
                block_hash: a.block_hash,
                confirm_daa: a.confirm_daa as u64,
                amount_sompi: a.amount_sompi as u64,
                output_index: a.output_index as u32,
                claimed: false,
                slashed: false,
                orphan_slashed: false,
                orphan_retries: 0,
                orphan_retry_after_daa: None,
                is_inference: false,
            })
            .collect();

        let total_sompi: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let count = entries.len();
        let state = escrow::EscrowState { entries };
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&opt.escrow_state_file, &json)?;

        info!(
            "Recovered {} escrow entries — claimable: {:.4} KRX",
            count,
            total_sompi as f64 / 1e8
        );
        info!("State saved to '{}'.", opt.escrow_state_file);
        return Ok(());
    }

    // Resolve OPoI escrow private key (once, before the reconnect loop).
    let escrow_privkey: Option<String> = match escrow::load_or_generate_key(&opt.escrow_key_file) {
        Ok(k) => {
            info!("OPoI: escrow key loaded from '{}'.", opt.escrow_key_file);
            Some(k)
        }
        Err(e) => {
            error!("Failed to load/generate OPoI escrow key: {}", e);
            return Err(e.into());
        }
    };

    // Phase-3 OPoI: load inference models before mining starts.
    //   (no flag)    → TinyLlama + DeepSeek-R1-8B  [default]
    //   --light      → TinyLlama only
    //   --high       → TinyLlama + DeepSeek-R1-8B + DeepSeek-R1-32B
    //   --very-high  → all 4 models

    // Warn if GPU power limit is below safe threshold for the selected model tier.
    // Low PL causes CUDA FIFO instability (Xid 32) under large GEMM workloads.
    check_gpu_power_limit(opt.high || opt.very_high, opt.very_high);

    let specs: &'static [&'static keryx_miner::models::ModelSpec] = if opt.very_high {
        info!("--very-high mode: loading all 4 models (TinyLlama + DeepSeek-8B + DeepSeek-32B + LLaMA-70B).");
        &[
            &keryx_miner::models::TINYLLAMA,
            &keryx_miner::models::DEEPSEEK_R1_8B,
            &keryx_miner::models::DEEPSEEK_R1_32B,
            &keryx_miner::models::LLAMA_3_3_70B,
        ]
    } else if opt.high {
        info!("--high mode: loading TinyLlama + DeepSeek-R1-8B + DeepSeek-R1-32B.");
        &[
            &keryx_miner::models::TINYLLAMA,
            &keryx_miner::models::DEEPSEEK_R1_8B,
            &keryx_miner::models::DEEPSEEK_R1_32B,
        ]
    } else if opt.light {
        info!("--light mode: loading TinyLlama only.");
        &[&keryx_miner::models::TINYLLAMA]
    } else {
        &[&keryx_miner::models::TINYLLAMA, &keryx_miner::models::DEEPSEEK_R1_8B]
    };
    keryx_miner::slm::init_supported(specs);
    info!("OPoI Phase-3 active — {} model(s) supported.", specs.len());
    info!("Prefetching model files before mining starts…");
    tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(specs)).await.ok();
    info!("Model files ready — starting mining.");
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16));
    if opt.devfund_percent > 0 {
        info!(
            "devfund enabled, mining {}.{}% of the time to devfund address: {} ",
            opt.devfund_percent / 100,
            opt.devfund_percent % 100,
            opt.devfund_address
        );
    }
    loop {
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager, escrow_privkey.clone()).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        info!("Client closed, reconnecting");
        sleep(Duration::from_millis(100));
    }
}
