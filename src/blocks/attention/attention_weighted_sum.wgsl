// attention_weighted_sum.wgsl
// Computes: ctx = Attn * V
// ctx is the context vector for each head, before concatenation and final projection.

override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32; // Max sequence length for attn scores & V buffer
override rope_theta: u32;// Max sequence length for attn scores & V buffer (num_tokens_kv)

// Group 0: Input Buffers
@group(0) @binding(0) var<storage, read> attn_scores : array<f32>; // Attn scores: [num_heads, num_tokens_q, context_length (num_tokens_kv)]
@group(0) @binding(1) var<storage, read> v_buffer    : array<f32>; // V buffer: [context_length (num_tokens_kv), num_kv_heads, head_size]

// Group 1: Output Buffer
@group(1) @binding(0) var<storage, read_write> ctx_out : array<f32>; // Output context: [num_tokens_q, num_heads, head_size]
                                                                   // This will be reshaped/viewed as [num_tokens_q, num_heads * head_size] for the next projection stage.

struct NumToksData {
    num_tokens: u32 // Current number of query tokens (num_tokens_q)
                     // num_tokens_kv is implicitly context_length for this shader from overrides.
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(8, 8) // Dispatch (num_tokens_q, num_heads_q)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t_q = global_id.x; // Query token index
    let h_q = global_id.y; // Query head index

    if (t_q >= num_toks_data.num_tokens || h_q >= num_heads) {
        return;
    }

    // Determine the corresponding KV head for this Q head (GQA)
    let num_q_heads_per_kv_group = num_heads / num_kv_heads;
    let h_kv = h_q / num_q_heads_per_kv_group; // Integer division, maps Q head to its KV head

    // Output offset for the current token and query head
    // ctx_out is [num_tokens_q, num_heads, head_size]
    // The data for token t_q, head h_q starts at (t_q * num_heads + h_q) * head_size
    let ctx_head_output_offset = (t_q * num_heads + h_q) * head_size;

    let num_tokens_kv = context_length; // Max sequence length for V buffer and attention scores

    // Iterate over each dimension of the head_size for the output context vector
    for (var d_idx: u32 = 0u; d_idx < head_size; d_idx = d_idx + 1u) {
        var weighted_sum_d: f32 = 0.0; // Accumulator for the d_idx-th dimension

        // Iterate over the key/value sequence length
        for (var t_kv: u32 = 0u; t_kv < num_tokens_kv; t_kv = t_kv + 1u) {
            // Attention scores are shaped [num_heads, num_tokens_q, num_tokens_kv (context_length)]
            // Index for attn_scores: (h_q * num_tokens_q_total + t_q) * num_tokens_kv_total + t_kv
            // Assuming num_tokens_q_total is num_toks_data.num_tokens
            let attn_score_idx = (h_q * num_toks_data.num_tokens + t_q) * num_tokens_kv + t_kv;
            let score = attn_scores[attn_score_idx];

            // V buffer is shaped [num_tokens_kv (context_length), num_kv_heads, head_size]
            // Index for v_buffer: (t_kv * num_kv_heads + h_kv) * head_size + d_idx
            let v_value_idx = (t_kv * num_kv_heads + h_kv) * head_size + d_idx;
            let v_val = v_buffer[v_value_idx];

            weighted_sum_d = weighted_sum_d + score * v_val;
        }
        ctx_out[ctx_head_output_offset + d_idx] = weighted_sum_d;
    }
}

