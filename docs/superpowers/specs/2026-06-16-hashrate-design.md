# Hashrate Improvement — Design Spec

**Date:** 2026-06-16  
**Scope:** NVIDIA CUDA pipeline only (RTX 30xx / 40xx / 50xx)  
**Kernel:** unchanged — no PTX modifications, no consensus risk

---

## Problem Statement

The CUDA dispatch loop in `miner.rs` has two structural inefficiencies that cap GPU utilization independently of kernel throughput:

1. **Redundant constant uploads** — `load_to_gpu()` is called on every kernel batch, even when the block template has not changed. Each call allocates a 4 KB `Arc` on the heap, constructs 3 `CString`s, and issues 3 `cuMemcpyHtoD` transfers (72 B + 4 KB + 32 B) to GPU constant memory. At 1 000+ batches/second this adds measurable dispatch overhead.

2. **Synchronous single-stream loop** — the CPU blocks on `sync()` waiting for the GPU to finish, then processes the result, then launches the next batch. The GPU sits idle during CPU processing time, and there is no mechanism to pick up a fresh block template while a batch is in flight.

---

## Goals

- Increase effective hashrate (MH/s) on a given GPU without modifying the `.cu` kernel or any PTX file.
- Ensure the miner always mines the most recent block template (critical for stratum pools).
- Provide measurable logging to confirm each improvement independently.

---

## Non-Goals

- OpenCL / AMD optimizations.
- CPU miner thread changes.
- CUDA kernel tuning (INT4 tensor cores, shared memory layout). This is follow-up work after baseline profiling with Nsight Compute.

---

## Architecture

Two independent improvements applied in layers:

```
┌─────────────────────────────────────────────────┐
│  src/miner.rs  (dispatch loop)                  │
│  ← A: lazy upload + non-blocking template check │
└─────────────────────┬───────────────────────────┘
                      │ Worker trait
┌─────────────────────▼───────────────────────────┐
│  plugins/cuda/src/worker.rs  (CudaGPUWorker)    │
│  ← B: two streams, two output buffers           │
└─────────────────────┬───────────────────────────┘
                      │ PTX (unchanged)
┌─────────────────────▼───────────────────────────┐
│  kaspa-cuda-native/src/kaspa-cuda.cu             │
│  NO CHANGES                                      │
└─────────────────────────────────────────────────┘
```

**Files changed:**
- `src/miner.rs`
- `plugins/cuda/src/worker.rs`
- `src/lib.rs` (Worker trait — add `drain()`)

---

## Improvement A — Lazy Constant Upload

### Root cause

`launch_gpu_miner` in `miner.rs` calls `state_ref.load_to_gpu(gpu_work)` on every loop iteration regardless of whether the block template changed:

```rust
// current — every iteration:
let state_ref = match &state {
    Some(s) => {
        s.load_to_gpu(gpu_work);  // always
        s
    }
    None => continue,
};
```

`load_block_constants` in `worker.rs` also allocates an `Arc<[[u8;64];64]>` (4 KB heap) on every call.

### Fix

**`miner.rs`** — track last-uploaded state ID; skip upload when template is unchanged:

```rust
let mut last_uploaded_id: Option<usize> = None;

// in loop:
let state_ref = match &state {
    Some(s) => {
        if last_uploaded_id != Some(s.id) {
            s.load_to_gpu(gpu_work);
            last_uploaded_id = Some(s.id);
        }
        s
    }
    None => continue,
};
```

`last_uploaded_id` is reset to `None` whenever `state` is set to `None` (block found) or a new template is detected (see below).

**`worker.rs`** — remove the unnecessary `Arc`; the conversion is synchronous and the value is not shared:

```rust
// before:
let u8matrix: Arc<[[u8; 64]; 64]> = Arc::new(matrix.map(|row| row.map(|v| v as u8)));

// after:
let u8matrix: [[u8; 64]; 64] = matrix.map(|row| row.map(|v| v as u8));
```

### Non-blocking template refresh

Currently the GPU thread only reads a new template when `state.is_none()`. On stratum pools the miner can mine a stale job for extended periods. After every `sync()`, perform a non-blocking check:

```rust
gpu_work.sync()?;

if block_channel.has_changed().unwrap_or(false) {
    if let Some(cmd) = block_channel.borrow_and_update().as_deref() {
        state = match cmd {
            WorkerCommand::Job(s) => { last_uploaded_id = None; Some(s.clone()) }
            WorkerCommand::Close  => return Ok(()),
            _ => state,
        };
    }
}
```

Setting `last_uploaded_id = None` on template change ensures the new constants are uploaded before the next kernel launch.

---

## Improvement B — Async Double-Buffered Pipeline

### Root cause

The single-stream loop serializes GPU execution and CPU work:

```
CPU: [launch A] [block on sync A] [check A] [launch B] [block on sync B] ...
GPU:            [=== A ===]                             [=== B ===]
```

CPU time between sync completion and the next launch is pure latency.

### Design

`CudaGPUWorker` adds a second stream, a second output buffer, a second event pair, and a second xoshiro rand-state buffer. A `ping: usize` field (0 or 1) selects the active stream; it flips after each `copy_output_to()`.

**New fields:**

| Field | Type | Purpose |
|---|---|---|
| `streams` | `[Stream; 2]` | Alternating CUDA streams |
| `start_events` | `[Event; 2]` | Per-stream kernel start |
| `stop_events` | `[Event; 2]` | Per-stream kernel stop |
| `final_nonce_buffs` | `[DeviceBuffer<u64>; 2]` | One nonce slot per stream |
| `rand_states` | `[DeviceBuffer<u64>; 2]` | Separate xoshiro states per stream |
| `ping` | `usize` | Active stream index (0 or 1) |
| `warmed_up` | `bool` | Skip pong sync on first iteration |

**Execution timeline:**

```
iter 0:  launch(A)  sync(noop)  copy(A, skip)  ping→1
iter 1:  launch(B)  sync(A)     copy(A)         ping→0
iter 2:  launch(A)  sync(B)     copy(B)         ping→1
...
```

- `calculate_hash()` — launches kernel on `streams[ping]`, records `start_events[ping]` and `stop_events[ping]`.
- `sync()` — synchronizes `stop_events[1-ping]`; returns `Ok(())` immediately when `!warmed_up`; sets `warmed_up = true` on first real sync. Applies the existing elapsed-time guard against overload.
- `copy_output_to()` — copies from `final_nonce_buffs[1-ping]`; returns empty (nonce=0) when `!warmed_up`; flips `ping` at the end.

**Template change — constant memory safety:**

CUDA `__constant__` symbols are module-global; both streams share `matrix`, `hash_header`, and `target`. Uploading new constants while a kernel is running would produce a data race. The `drain()` method (new on the `Worker` trait) synchronizes both streams before the upload:

```rust
fn drain(&mut self) -> Result<(), Error> {
    self.stop_events[0].synchronize()?;
    self.stop_events[1].synchronize()?;
    self.warmed_up = false;
    Ok(())
}
```

`miner.rs` calls `gpu_work.drain()` immediately after detecting a template change and before calling `load_to_gpu()`.

**Memory cost:** one additional xoshiro buffer of `4 × workload × 8` bytes. For a typical RTX 3090 workload (~200 K nonces): ~6.4 MB — negligible.

---

## Worker Trait Changes

```rust
pub trait Worker {
    // existing:
    fn id(&self) -> String;
    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]);
    fn calculate_hash(&mut self, nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64);
    fn sync(&self) -> Result<(), Error>;
    fn get_workload(&self) -> usize;
    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error>;

    // new:
    fn drain(&mut self) -> Result<(), Error>;  // drain all in-flight work before constant re-upload
}
```

The OpenCL worker provides a trivial `drain()` (single stream, just calls existing `sync()`).

---

## Logging

Extend the existing 10-second hashrate log in `miner.rs` to include per-worker dispatch metrics:

```
[GPU #0 RTX 3090] 247.1 MH/s  (batches/s: 1207  kernel: 0.83ms  uploads: 2/12070)
```

- `uploads: 2/12070` — number of actual constant re-uploads vs. total batches. Confirms lazy upload is working (typically equals the number of new block templates received in the window).

---

## Testing

### Consensus correctness

The kernel is unchanged, so the hash algorithm is identical. The risk is the lazy upload sending stale constants. Verification:

- Extend the existing `test_serialize_header` pattern: construct two `pow::State` values with different templates (distinct IDs), simulate a template swap, confirm the CPU-side `calculate_pow()` result matches the nonce returned by the GPU after the swap.
- The `state.id` field is monotonically increasing and serves as the ground-truth oracle.

### Hashrate regression

1. Measure baseline MH/s with the current code on the target GPU.
2. Apply Improvement A alone; measure again.
3. Apply Improvement B; measure again.
4. Compare all three. Each step should be equal or better — never regress.

### Double-buffer stress test

Run with `--mine-when-not-synced` against a local node configured to produce new templates every 200 ms (faster than normal). Verify:
- No `"Something is wrong in GPU results"` warnings in the log.
- `uploads` counter in logs matches template-change frequency.
- Miner does not deadlock or stall during rapid template cycling.

---

## Implementation Order

1. **A1** — Remove `Arc` allocation in `load_block_constants` (one-liner, zero risk).
2. **A2** — Lazy upload (`last_uploaded_id`) in `miner.rs`.
3. **A3** — Non-blocking template refresh after `sync()`.
4. Measure: deploy to test rig, record MH/s and `uploads/batches` ratio.
5. **B** — Double-buffered streams in `CudaGPUWorker` + `drain()` on Worker trait.
6. Measure again and compare.
