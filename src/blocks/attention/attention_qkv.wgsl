override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32;
override rope_theta: u32;
// context_length is not directly used in QKV projection per token,
// but might be kept for consistency with other shaders if buffer sizes depend on it.
// override context_length: u32;
// rope_theta is not used in this shader.

@group(0) @binding(0) var<storage, read>       x     : array<f32>; // Input: [num_tokens, embedding_size]
@group(0) @binding(1) var<storage, read>       wq    : array<f32>; // Weights: [num_heads * head_size, embedding_size]
@group(0) @binding(2) var<storage, read>       wk    : array<f32>; // Weights: [num_kv_heads * head_size, embedding_size]
@group(0) @binding(3) var<storage, read>       wv    : array<f32>; // Weights: [num_kv_heads * head_size, embedding_size]
@group(0) @binding(4) var<storage, read_write> q_out : array<f32>; // Output: [num_tokens, num_heads, head_size]
@group(0) @binding(5) var<storage, read_write> k_out : array<f32>; // Output: [num_tokens, num_kv_heads, head_size]
@group(0) @binding(6) var<storage, read_write> v_out : array<f32>; // Output: [num_tokens, num_kv_heads, head_size]

struct NumToksData {
    num_tokens: u32
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(8,8) // Workgroup size might need tuning. Dispatch is (num_tokens, num_heads/num_kv_heads)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;          // token index
  let h_idx = gid.y;      // head index for Q, or KV_head index if dispatched accordingly

  if (t >= num_toks_data.num_tokens) { return; }

  // Q projection
  if (h_idx < num_heads) {
    let q_head_offset = (t * num_heads + h_idx) * head_size;
    for (var d: u32 = 0u; d < head_size; d = d + 1u) { // Loop over head_size
      var sum_q: f32 = 0.0;
      let wq_row_offset = (h_idx * head_size + d) * embedding_size; // Start of row in Wq for this output element
      for (var i: u32 = 0u; i < embedding_size; i = i + 1u) { // Dot product
        sum_q = sum_q + x[t * embedding_size + i] * wq[wq_row_offset + i];
      }
      q_out[q_head_offset + d] = sum_q;
    }
  }

  // K projection
  // Assuming gid.y (h_idx) is dispatched up to num_kv_heads for K/V,
  // or we check h_idx < num_kv_heads if gid.y goes up to num_heads.
  // For simplicity, let's assume the dispatch for K/V might be separate or h_idx is the kv_head_idx.
  // If gid.y goes up to num_heads, we need to ensure K/V are only computed for h_idx < num_kv_heads.
  if (h_idx < num_kv_heads) { // h_idx here is treated as kv_head_index
    let k_head_offset = (t * num_kv_heads + h_idx) * head_size;
    let v_head_offset = (t * num_kv_heads + h_idx) * head_size;

    for (var d: u32 = 0u; d < head_size; d = d + 1u) { // Loop over head_size
      var sum_k: f32 = 0.0;
      var sum_v: f32 = 0.0;
      let wk_row_offset = (h_idx * head_size + d) * embedding_size; // Start of row in Wk
      let wv_row_offset = (h_idx * head_size + d) * embedding_size; // Start of row in Wv

      for (var i: u32 = 0u; i < embedding_size; i = i + 1u) { // Dot product
        let x_val = x[t * embedding_size + i];
        sum_k = sum_k + x_val * wk[wk_row_offset + i];
        sum_v = sum_v + x_val * wv[wv_row_offset + i];
      }
      k_out[k_head_offset + d] = sum_k;
      v_out[v_head_offset + d] = sum_v;
    }
  }
}