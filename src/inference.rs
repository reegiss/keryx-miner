/// Phase-1 OPoI inference module.
///
/// Detects AiRequest TXs in block templates and runs synthetic MLP inference.
/// The result is embedded in the next coinbase via extra_data so the indexer
/// can link responses back to their originating request TXs.

const KRX_AI_PREFIX: &[u8] = b"KRX:AI:1:";

/// MODEL_SEED shared with keryxd — "KeryxOP\0" as little-endian u64.
const MODEL_SEED: u64 = 0x4B_65_72_79_78_4F_50_00;

// ---------------------------------------------------------------------------
// LCG weight generator — matches keryx-node/inference/src/model.rs exactly.
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005)
                 .wrapping_add(1_442_695_040_888_963_407);
    let mantissa = (*state >> 41) as f32;
    mantissa / (1u32 << 23) as f32 * 2.0 - 1.0
}

fn make_weights(rows: usize, cols: usize, layer_id: u64) -> Vec<f32> {
    let mut state = MODEL_SEED.wrapping_add(layer_id.wrapping_mul(0xDEAD_BEEF_CAFE_1337));
    let scale = (2.0_f32 / cols as f32).sqrt(); // He initialization
    (0..rows * cols).map(|_| lcg_next(&mut state) * scale).collect()
}

/// Linear layer: out[i] = sum_j(w[i*cols + j] * x[j]).  Biases are zero.
fn linear(x: &[f32], w: &[f32], in_size: usize, out_size: usize) -> Vec<f32> {
    (0..out_size)
        .map(|i| {
            let row = &w[i * in_size..(i + 1) * in_size];
            row.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn relu(v: f32) -> f32 {
    v.max(0.0)
}

/// Forward pass of the 3-layer OPoI MLP (32→256→128→32).
///
/// Matches `keryx-node/inference/src/model.rs::forward()` exactly.
fn mlp_forward(input: &[u8; 32]) -> [u8; 32] {
    // Normalise input bytes to [-0.5, 0.5]
    let x: Vec<f32> = input.iter().map(|&b| b as f32 / 255.0 - 0.5).collect();

    // Layer 1: 32 → 256, ReLU
    let w1 = make_weights(256, 32, 0);
    let h1: Vec<f32> = linear(&x, &w1, 32, 256).into_iter().map(relu).collect();

    // Layer 2: 256 → 128, ReLU
    let w2 = make_weights(128, 256, 1);
    let h2: Vec<f32> = linear(&h1, &w2, 256, 128).into_iter().map(relu).collect();

    // Layer 3: 128 → 32, no activation
    let w3 = make_weights(32, 128, 2);
    let out: Vec<f32> = linear(&h2, &w3, 128, 32);

    // Map through tanh to [0, 255]
    let mut result = [0u8; 32];
    for (i, &v) in out.iter().enumerate().take(32) {
        result[i] = ((v.tanh() * 0.5 + 0.5) * 255.0).round() as u8;
    }
    result
}

/// Compute the Phase-1 OPoI tag for a coinbase.
///
/// Parses the 16-char hex nonce, builds the 32-byte MLP input, runs the
/// forward pass and returns the first 8 output bytes as 16 lowercase hex chars.
/// The node validates that `/ai:v1:{16 hex chars}` is present in every coinbase.
pub fn compute_opoi_tag(nonce_hex: &str) -> String {
    // Parse nonce as u64
    let nonce = u64::from_str_radix(nonce_hex, 16).unwrap_or(0);

    // Build 32-byte input: nonce LE bytes + repeating MODEL_SEED bytes
    let mut input = [0u8; 32];
    input[..8].copy_from_slice(&nonce.to_le_bytes());
    let seed_bytes = MODEL_SEED.to_le_bytes();
    for chunk in input[8..].chunks_mut(8) {
        let n = chunk.len();
        chunk.copy_from_slice(&seed_bytes[..n]);
    }

    let output = mlp_forward(&input);
    hex::encode(&output[..8])
}

// ---------------------------------------------------------------------------
// AiRequest TX scanning helpers.
// ---------------------------------------------------------------------------

/// Extract an AI inference request from a hex-encoded transaction payload.
///
/// Returns `(payload_prefix_16, prompt, max_tokens)` if the payload starts
/// with the KRX:AI:1: marker and contains a valid JSON body.
/// `payload_prefix_16` is the first 8 bytes of blake2b-256(payload_bytes) as
/// 16 lowercase hex chars — a deterministic identifier that does not require
/// the transaction ID (unavailable in GetBlockTemplateResponse).
pub fn extract_ai_request(payload_hex: &str) -> Option<(String, String, usize)> {
    if payload_hex.is_empty() {
        return None;
    }
    let bytes = hex::decode(payload_hex).ok()?;
    if bytes.len() <= KRX_AI_PREFIX.len() {
        return None;
    }
    if !bytes.starts_with(KRX_AI_PREFIX) {
        return None;
    }
    let json_str = std::str::from_utf8(&bytes[KRX_AI_PREFIX.len()..]).ok()?;
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let prompt = v["p"].as_str()?.to_string();
    let max_tokens = v["n"].as_u64().unwrap_or(128) as usize;
    // Compute stable prefix from blake2b(payload_bytes) — matches the indexer's payload_prefix.
    let hash = blake2b_simd::blake2b(&bytes);
    let prefix = hex::encode(&hash.as_bytes()[..8]);
    Some((prefix, prompt, max_tokens))
}

/// Run synthetic Phase-1 inference on a prompt.
///
/// Computes blake2b(prompt) and returns the first 8 bytes as 16 lowercase hex
/// chars.  Deterministic and cheap — proves the miner processed the prompt
/// without requiring a real neural network in Phase 1.
pub fn run_synthetic_inference(prompt: &str) -> String {
    use blake2b_simd::blake2b;
    let hash = blake2b(prompt.as_bytes());
    hex::encode(&hash.as_bytes()[..8])
}
