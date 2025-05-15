override embedding_size: u32; // Not directly used here but part of the constants
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32; // Max context length for KV cache, or current sequence length for logits
override rope_theta: u32;

@group(0) @binding(0) var<storage, read>  q    : array<f32>; // Query: [num_tokens, num_heads, head_size]
@group(0) @binding(1) var<storage, read>  k    : array<f32>; // Key:   [num_tokens, num_kv_heads, head_size]
@group(0) @binding(2) var<storage, read_write> logits : array<f32>; // Output: [num_heads, num_tokens, context_length (num_tokens_kv)]

struct NumToksData {
    num_tokens: u32     // Current number of query tokens
    // num_tokens_kv: u32; // Current number of key/value tokens (from KV cache) - assumed same as num_tokens for this fix if not provided
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

const ATTENTION_SCALE: f32 = 1.0; // Placeholder if 1/sqrt(head_size) is needed, set it here or pass as constant

@compute @workgroup_size(8,8) // Dispatch (num_heads, num_tokens_q)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h_q = gid.x; // Query head index
  let t_q = gid.y; // Query token index (sequence position for Q)

  if (h_q >= num_heads || t_q >= num_toks_data.num_tokens) { return; }

  let q_offset = (t_q * num_heads + h_q) * head_size;

  // Determine the corresponding KV head for this Q head (GQA)
  let num_q_heads_per_kv_group = num_heads / num_kv_heads;
  let h_kv = h_q / num_q_heads_per_kv_group; // Integer division

  // Calculate scaling factor (optional, add if Llama3 uses it)
  let scale = 1.0 / sqrt(f32(head_size)); // Common scaling factor

  var max_logit: f32 = -3.402823e+38; // Smallest f32

  // Calculate dot products and find max_logit
  // Assuming num_tokens_kv is available, for now using num_toks_data.num_tokens for key sequence length
  let num_tokens_kv = num_toks_data.num_tokens; // Or pass separately if KV cache has different length

  for (var t_kv: u32 = 0u; t_kv < num_tokens_kv; t_kv = t_kv + 1u) {
    // Only consider keys up to the current query token's position for causal attention
    if (t_kv > t_q) { break; }

    let k_offset = (t_kv * num_kv_heads + h_kv) * head_size;
    var dotp: f32 = 0.0;
    for (var d: u32 = 0u; d < head_size; d = d + 1u) {
      dotp = dotp + q[q_offset + d] * k[k_offset + d];
    }
    dotp = dotp * scale; // Apply scaling

    // Store temporarily or recompute; here storing in logits then finding max
    // Logits shape: [num_heads, num_tokens_q, num_tokens_kv]
    // Index for logits: (h_q * num_toks_data.num_tokens + t_q) * context_length + t_kv
    // Note: context_length here should be max_kv_sequence_length
    logits[(h_q * num_toks_data.num_tokens + t_q) * context_length + t_kv] = dotp;
    if (dotp > max_logit) {
      max_logit = dotp;
    }
  }

  // Subtract max_logit, exponentiate, and calculate sum_exp
  var sum_exp: f32 = 0.0;
  for (var t_kv: u32 = 0u; t_kv < num_tokens_kv; t_kv = t_kv + 1u) {
    if (t_kv > t_q) { // Apply causal mask for sum_exp and final normalization
        // Set masked logits to 0 before normalization, or handle by not summing
        logits[(h_q * num_toks_data.num_tokens + t_q) * context_length + t_kv] = 0.0; // Or some indicator
        continue;
    }
    let logit_idx = (h_q * num_toks_data.num_tokens + t_q) * context_length + t_kv;
    let val = exp(logits[logit_idx] - max_logit);
    logits[logit_idx] = val; // Store exp(logit - max_logit)
    sum_exp = sum_exp + val;
  }

  // Normalize
  if (sum_exp > 0.0) { // Avoid division by zero if all masked or zero
    for (var t_kv: u32 = 0u; t_kv < num_tokens_kv; t_kv = t_kv + 1u) {
       if (t_kv > t_q) { continue; } // Only normalize unmasked values
      let logit_idx = (h_q * num_toks_data.num_tokens + t_q) * context_length + t_kv;
      logits[logit_idx] = logits[logit_idx] / sum_exp;
    }
  }
}