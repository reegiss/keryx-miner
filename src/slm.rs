/// Phase-3 OPoI: TinyLlama-1.1B-Chat inference engine via candle-transformers.
///
/// The engine is initialised once via `load_blocking()`, which must be called
/// (and awaited) before any mining begins. Mining is blocked until the model is
/// ready — there is no synthetic fallback.
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, LlamaConfig, Llama};
use std::io::{Read, Write};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const HF_BASE: &str = "https://huggingface.co";
const SYSTEM_PROMPT: &str =
    "You are a concise AI assistant embedded in the Keryx blockchain network. \
     Reply in 1-3 sentences maximum.";

// ── Static singleton ─────────────────────────────────────────────────────────

/// Holds either a loaded engine or the error string from the failed load.
static ENGINE: OnceLock<std::result::Result<SlmEngine, String>> = OnceLock::new();

pub struct SlmEngine {
    model: Llama,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

// Candle tensors use Arc internally and are safe to share across threads.
unsafe impl Send for SlmEngine {}
unsafe impl Sync for SlmEngine {}

impl SlmEngine {
    /// Return the directory where model files are stored, next to the miner binary.
    ///
    /// Layout: `<exe_dir>/models/TinyLlama-1.1B/`
    /// Falls back to `./models/TinyLlama-1.1B/` if the exe path cannot be resolved.
    fn model_dir() -> std::path::PathBuf {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        exe_dir.join("models").join("TinyLlama-1.1B")
    }

    /// Look for the three required files in the model directory.
    fn find_in_model_dir() -> Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        let dir = Self::model_dir();
        let tok = dir.join("tokenizer.json");
        let cfg = dir.join("config.json");
        let wts = dir.join("model.safetensors");
        if tok.exists() && cfg.exists() && wts.exists() {
            log::info!("SlmEngine: found local model at {}", dir.display());
            Some((tok, cfg, wts))
        } else {
            None
        }
    }

    /// Download a single file from HuggingFace with a stderr progress bar.
    fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
        eprintln!("[keryx-miner] Downloading {} ...", url);
        let response = ureq::get(url)
            .call()
            .map_err(|e| anyhow!("HTTP GET {}: {}", url, e))?;

        let content_length: Option<u64> = response
            .header("Content-Length")
            .and_then(|s| s.parse().ok());

        let mut reader = response.into_reader();
        let mut file = std::fs::File::create(dest)
            .with_context(|| format!("create {}", dest.display()))?;

        let mut downloaded: u64 = 0;
        let mut buf = vec![0u8; 65_536];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            if let Some(total) = content_length {
                let pct = downloaded * 100 / total;
                eprint!(
                    "\r  {:.1}/{:.1} MB ({}%)   ",
                    downloaded as f64 / 1_000_000.0,
                    total as f64 / 1_000_000.0,
                    pct
                );
                let _ = std::io::stderr().flush();
            }
        }
        eprintln!();
        Ok(())
    }

    /// Ensure the three model files exist locally, downloading them if needed.
    ///
    /// Files are stored in `<exe_dir>/models/TinyLlama-1.1B/`.
    fn ensure_files() -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        if let Some(paths) = Self::find_in_model_dir() {
            return Ok(paths);
        }

        let model_dir = Self::model_dir();
        std::fs::create_dir_all(&model_dir)
            .with_context(|| format!("create model dir {}", model_dir.display()))?;

        let tok_path = model_dir.join("tokenizer.json");
        let cfg_path = model_dir.join("config.json");
        let wts_path = model_dir.join("model.safetensors");

        let base = format!("{}/{}/resolve/main", HF_BASE, MODEL_ID);

        eprintln!(
            "\n[keryx-miner] TinyLlama-1.1B model not found in local cache."
        );
        eprintln!("[keryx-miner] Downloading model files (~2.2 GB). This happens once.\n");

        if !tok_path.exists() {
            Self::download_file(&format!("{}/tokenizer.json", base), &tok_path)?;
        }
        if !cfg_path.exists() {
            Self::download_file(&format!("{}/config.json", base), &cfg_path)?;
        }
        if !wts_path.exists() {
            Self::download_file(&format!("{}/model.safetensors", base), &wts_path)?;
        }

        eprintln!("[keryx-miner] Model download complete.\n");
        Ok((tok_path, cfg_path, wts_path))
    }

    /// Load TinyLlama from the local cache (downloading first if absent).
    fn load() -> Result<Self> {
        let device = Device::Cpu;
        log::info!("SlmEngine: loading TinyLlama-1.1B-Chat-v1.0…");

        let (tokenizer_path, config_path, weights_path) = Self::ensure_files()?;

        let config_str = std::fs::read_to_string(&config_path)?;
        let llama_cfg: LlamaConfig =
            serde_json::from_str(&config_str).context("parse config.json")?;
        let config = llama_cfg.into_config(false);

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("load tokenizer: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
        }
        .map_err(|e| anyhow!("mmap weights: {}", e))?;

        let model = Llama::load(vb, &config).map_err(|e| anyhow!("build model: {}", e))?;

        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(2);

        log::info!("SlmEngine: TinyLlama ready (eos_id={})", eos_token_id);
        Ok(Self { model, config, tokenizer, device, eos_token_id })
    }

    /// Generate a response using the ChatML prompt format.
    fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let formatted = format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            SYSTEM_PROMPT, prompt
        );

        let enc = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow!("encode: {}", e))?;
        let mut all_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut generated: Vec<u32> = Vec::new();

        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)
            .map_err(|e| anyhow!("create KV cache: {}", e))?;
        let mut lp = LogitsProcessor::new(42, Some(0.7), Some(0.9));
        let max_steps = max_new_tokens.min(256);

        for step in 0..max_steps {
            let (input_ids, pos) = if step == 0 {
                (all_tokens.as_slice(), 0usize)
            } else {
                let last = all_tokens.len() - 1;
                (&all_tokens[last..], last)
            };

            let input = Tensor::new(input_ids, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| anyhow!("build input tensor: {}", e))?;

            // forward → [1, seq_len, vocab_size]
            let logits = self
                .model
                .forward(&input, pos, &mut cache)
                .map_err(|e| anyhow!("forward: {}", e))?;

            let last_logits = extract_last_token_logits(&logits)?;

            let next_token =
                lp.sample(&last_logits).map_err(|e| anyhow!("sample: {}", e))?;
            if next_token == self.eos_token_id {
                break;
            }
            all_tokens.push(next_token);
            generated.push(next_token);
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {}", e))?;
        Ok(text.trim().to_string())
    }
}

/// Extract the logit vector for the last sequence position.
/// Input shape: [1, seq_len, vocab] or [seq_len, vocab].
fn extract_last_token_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    match dims.len() {
        3 => {
            let seq_len = dims[1];
            logits
                .narrow(1, seq_len - 1, 1)
                .and_then(|t| t.squeeze(1))
                .and_then(|t| t.squeeze(0))
                .map_err(|e| anyhow!("extract logits (3d): {}", e))
        }
        2 => {
            let seq_len = dims[0];
            logits
                .narrow(0, seq_len - 1, 1)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| anyhow!("extract logits (2d): {}", e))
        }
        _ => Err(anyhow!("unexpected logits shape {:?}", dims)),
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Load the SLM engine (blocking). Must be called and awaited before mining starts.
///
/// Downloads TinyLlama-1.1B on first run (~2.2 GB). Returns `Err` if the model
/// cannot be loaded; in that case the miner must refuse to start.
pub fn load_blocking() -> Result<(), String> {
    let result = ENGINE.get_or_init(|| SlmEngine::load().map_err(|e| e.to_string()));
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

/// Run TinyLlama inference (blocking — must be called from `spawn_blocking`).
///
/// Panics if `load_blocking` was not called first (programming error).
pub fn run_inference(prompt: &str, max_tokens: usize) -> String {
    let engine_ref = ENGINE
        .get()
        .expect("SlmEngine must be loaded via load_blocking() before calling run_inference()");

    match engine_ref {
        Ok(engine) => match engine.generate(prompt, max_tokens) {
            Ok(text) if !text.is_empty() => text,
            Ok(_) => "[empty response]".to_string(),
            Err(e) => {
                log::warn!("SlmEngine generate error: {}", e);
                format!("[inference error: {}]", e)
            }
        },
        // load_blocking() returned Ok, so this branch is unreachable.
        Err(e) => panic!("SlmEngine is in an error state: {}", e),
    }
}
