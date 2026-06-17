# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Prerequisites:** Rust + Cargo, `protoc` (protobuf-compiler)

```bash
# Standard build (PoW + OPoI inference via CUDA)
cargo build --release --bin keryx-miner

# With no-asm keccak (software fallback, no x86-64 asm)
cargo build --release --bin keryx-miner --features=no-asm

# CUDA build with specific compute capability
CUDA_COMPUTE_CAP=86 cargo build --release --bin keryx-miner
```

## Tests

```bash
# Run all tests for the main crate
cargo test -p keryx-miner

# Run tests without default features (no parking_lot)
cargo test -p keryx-miner --no-default-features

# Run tests with no-asm feature
cargo test -p keryx-miner --features=no-asm

# Run tests for the inference crate
cargo test -p keryx-inference

# Run a single test by name
cargo test -p keryx-miner test_serialize_header
```

## Lint & Format

```bash
cargo fmt --all
cargo fmt --all -- --check   # CI check
cargo clippy --tests -- -D warnings
```

## Architecture

This is a Cargo workspace with three main components:

### Main crate (`src/`)
The binary that orchestrates mining. Entry point is `src/main.rs`, which:
1. Loads GPU plugins (CUDA/OpenCL shared libraries) via `PluginManager`
2. Prefetches LLM model files
3. Probes GPU inference (cuBLAS)
4. Connects to keryxd via gRPC or a stratum pool (`src/client/`)
5. Runs the reconnect loop

Key modules:
- `src/pow/` — kHeavyHash PoW implementation. `heavy_hash.rs` manages the salt-versioned matrix (SALT v1/v2/v4 activated at specific DAA scores). **DAA score thresholds in `heavy_hash.rs` must stay in sync with the node's consensus params.**
- `src/miner.rs` — thread pool managing CPU/GPU worker threads
- `src/slm.rs` — Phase-3 OPoI multi-model inference engine (candle). Loads GGUF/safetensors models on demand and caches them between requests. Mining pauses during inference.
- `src/models.rs` — static registry of `ModelSpec` entries (TinyLlama, DeepSeek-R1-8B/32B, LLaMA-3.3-70B), each identified by a 32-byte `model_id` = sha256 of primary weight file
- `src/escrow.rs` — automated OPoI escrow claim logic: scans coinbase outputs for this miner's script, builds Schnorr-signed claim TXs after the 36,000-block CSV window, persists state to `escrow_state.json`
- `src/client/grpc.rs` + `src/client/stratum.rs` — two protocol backends (full node gRPC vs. pool stratum+tcp)
- `src/inference.rs` — re-exports from the `keryx-inference` crate; Phase-3 OPoI challenge dispatch
- `src/ipfs.rs` — ensures an IPFS Kubo daemon is running for uploading inference results

### `inference/` crate (`keryx-inference`)
OPoI verification logic shared between miner and node:
- `model.rs` — Phase-1 candle f32 MLP (non-deterministic, reference only)
- `model_fixed.rs` — Phase-2 bit-exact fixed-point MLP for on-chain verification
- `fraud_proof.rs` — ZK fraud proof stubs (Groth16, circuit VK lands in Phase 3C)
- `ai_payload.rs` — serialization for AiRequest/AiResponse/AiChallenge on-chain transactions
- The exported `parse_opoi()` / `verify_tag_fixed()` / `tag_fixed()` are the consensus-critical path

### GPU plugins (`plugins/`)
`plugins/cuda/` and `plugins/opencl/` are independent crates that compile to `cdylib` shared libraries (`libkeryxcuda.so`, `libkeryxopencl.so`). They are discovered at runtime by the main binary via `filter_plugins()` and loaded with `libloading`. A plugin must export `_plugin_create` (declared via the `declare_plugin!` macro in `src/lib.rs`).

### Proto (`proto/`)
gRPC definitions for keryxd communication. `build.rs` compiles them with `tonic-build` into `protowire` (included at `src/main.rs:39`). Re-run `cargo build` to regenerate after proto changes.

## OPoI (Optimistic Proof of Inference) Overview

OPoI is the on-chain AI inference proof system layered on top of kHeavyHash PoW:
- **Phase 2** (active): fixed-point MLP (`inference/src/model_fixed.rs`) produces a deterministic 16-hex tag embedded in coinbase `extra_data` as `/{nonce_hex16}/ai:v1:{tag_hex16}`. Miners using older code produce wrong tags and their blocks are rejected.
- **Phase 3** (current): real LLM inference via candle. AiRequest/AiResponse/AiChallenge are on-chain transactions. Collateral is held in escrow and slashed for fraudulent responses.
- The `PHASE2_OPOI_SALT` constant in `inference/src/lib.rs` and the matrix salts in `src/pow/heavy_hash.rs` are the forced-upgrade mechanism — changing them invalidates older miner builds.

## CUDA Build Notes

- Build with CUDA 12.2 toolkit (not 12.6) for widest driver compatibility (driver ≥ 535 / HiveOS)
- `CUDA_COMPUTE_CAP` env var selects PTX target (86=Ampere RTX 30xx, 89=Ada RTX 40xx, 100=Blackwell RTX 50xx)
- GPU inference requires `libcublas-12-2` and `libcurand-12-2` at runtime (not included in the bare NVIDIA driver); on Linux/HiveOS the miner auto-installs them via `install_cuda_libs()`
- The keccak Keccak-f[1600] permutation uses hand-written x86-64 assembly (`src/keccakf1600_x86-64.s` / `*-osx.s`); use `--features=no-asm` to fall back to the pure-Rust `keccak` crate
