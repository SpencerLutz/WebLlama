override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32; // Max sequence length for attn scores & V buffer
override rope_theta: u32;

// Group 0: For Weighted Sum part
@group(0) @binding(0) var<storage, read>  attn  : array<f32>; // Attn scores: [num_heads, num_tokens_q, context_length (num_tokens_kv)]
@group(0) @binding(1) var<storage, read>  v     : array<f32>; // V buffer: [context_length (num_tokens_kv), num_kv_heads, head_size]
@group(0) @binding(2) var<storage, read_write> ctx : array<f32>; // Intermediate context: [num_tokens_q, num_heads, head_size] (concatenated form: [num_tokens_q, num_heads * head_size])

// Group 1: For Output Projection part
// ctx_in should ideally be the same as ctx after all heads have written to it.
// The current TS binds a separate outContext2 here. This fix assumes ctx_in IS the data from the first stage.
@group(1) @binding(0) var<storage, read>  ctx_in : array<f32>; // This should be the ctx buffer: [num_tokens_q, num_heads * head_size]
@group(1) @binding(1) var<storage, read>  wo     : array<f32>; // Output weights: [embedding_size, num_heads * head_size]
@group(1) @binding(2) var<storage, read_write> out  : array<f32>; // Final output: [num_tokens_q, embedding_size]

struct NumToksData {
    num_tokens: u32     // Current number of query tokens (num_tokens_q)
    // num_tokens_kv: u32; // Current number of key/value tokens (context_length effectively)
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(8,8) // Dispatch (num_tokens_q, num_heads_q)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t_q = gid.x;   // Query token index
  let h_q = gid.y;   // Query head index

  if (t_q >= num_toks_data.num_tokens || h_q >= num_heads) { return; }

  // --- Part 1: Weighted Sum (Attn * V) ---
  // Each invocation (t_q, h_q) calculates one head's contribution to the context vector.
  let num_q_heads_per_kv_group = num_heads / num_kv_heads;
  let h_kv = h_q / num_q_heads_per_kv_group; // Corresponding KV head index

  let ctx_head_offset = (t_q * num_heads + h_q) * head_size;
  let num_tokens_kv = context_length; // Assuming attn scores and V are for up to context_length

  for (var d_idx: u32 = 0u; d_idx < head_size; d_idx = d_idx + 1u) { // Loop over each dimension of the head
    var weighted_sum_d: f32 = 0.0;
    for (var t_kv: u32 = 0u; t_kv < num_tokens_kv; t_kv = t_kv + 1u) {
      // Causal mask: only sum if t_kv <= t_q.
      // The attention scores 'attn' should already be causally masked (e.g., 0 for t_kv > t_q).
      // If not, an explicit check 'if (t_kv > t_q) { continue; }' might be needed here or when attn was computed.
      // We assume 'attn' contains valid (possibly 0 for masked) scores.

      let attn_score_idx = (h_q * num_toks_data.num_tokens + t_q) * context_length + t_kv;
      let a = attn[attn_score_idx];

      let v_val_idx = (t_kv * num_kv_heads + h_kv) * head_size + d_idx;
      weighted_sum_d = weighted_sum_d + a * v[v_val_idx];
    }
    ctx[ctx_head_offset + d_idx] = weighted_sum_d;
  }

  // WORKGROUP BARRIER WOULD BE NEEDED HERE if Part 2 relies on writes from other invocations
  // in the SAME workgroup to 'ctx'. Given the dispatch (num_tokens, num_heads), different heads
  // for the same token are likely in different workgroups or independent invocations.
  // Thus, this barrier is more conceptual for the split of logic.
  // The current `AttentionBlock.ts` pass structure implies Part 1 & 2 are in the same dispatch.
  // If `ctx_in` is truly a separate buffer `outContext2`, then `outContext` must be copied to `outContext2`
  // *between passes*, or this shader needs to be split into two passes.

  // --- Part 2: Output Projection (Wo * ctx_concat) ---
  // Each invocation (t_q, h_q) will compute a slice of the final embedding_size output.
  // The number of output elements this invocation computes:
  let output_slice_size = embedding_size / num_heads;
  let output_start_dim = h_q * output_slice_size;

  let concatenated_ctx_token_offset = t_q * num_heads * head_size;
  let final_out_token_offset = t_q * embedding_size;

  for (var out_d_slice: u32 = 0u; out_d_slice < output_slice_size; out_d_slice = out_d_slice + 1u) {
    let current_out_dim = output_start_dim + out_d_slice; // Index in the final embedding_size vector
    if (current_out_dim >= embedding_size) { continue; } // Boundary check for safety

    var proj_sum: f32 = 0.0;
    // Wo weights: [embedding_size, num_heads * head_size] (row-major: rows are output_dims, cols are input_dims)
    // Weight row for current_out_dim starts at: current_out_dim * (num_heads * head_size)
    let wo_row_offset = current_out_dim * (num_heads * head_size);

    for (var concat_d: u32 = 0u; concat_d < num_heads * head_size; concat_d = concat_d + 1u) {
      // Read from ctx_in (which should hold the fully formed concatenated context for token t_q)
      proj_sum = proj_sum + ctx_in[concatenated_ctx_token_offset + concat_d] * wo[wo_row_offset + concat_d];
    }
    out[final_out_token_offset + current_out_dim] = proj_sum;
  }
}