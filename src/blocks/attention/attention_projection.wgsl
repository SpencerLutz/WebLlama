// attention_projection.wgsl
// Computes: final_out = Wo * ctx_concatenated
// ctx_concatenated is the result from attention_weighted_sum, viewed as [num_tokens, num_heads * head_size]

override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32; // Max sequence length for attn scores & V buffer
override rope_theta: u32;
// context_length (max sequence length) is not directly used here for indexing Wo,
// but is implicitly part of the input ctx_in buffer's total size.

// Group 0: Input Buffers
@group(0) @binding(0) var<storage, read> ctx_in : array<f32>; // Concatenated context from previous pass: [num_tokens, num_heads * head_size]
@group(0) @binding(1) var<storage, read> wo     : array<f32>; // Output projection weights Wo: [embedding_size, num_heads * head_size]

// Group 1: Output Buffer
@group(1) @binding(0) var<storage, read_write> final_out : array<f32>; // Final output: [num_tokens, embedding_size]

struct NumToksData {
    num_tokens: u32 // Current number of query tokens
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

// Workgroup size for a typical MatMul.
// Here, each invocation calculates one element of the final_out matrix.
// Dispatch will be (num_tokens, embedding_size), potentially tiled.
// For simplicity with current block structure, let's keep dispatch (num_tokens, embedding_size / some_factor)
// and loop inside if needed, or dispatch (num_tokens * embedding_size / workgroup_size_x)
// A common approach is to have each workgroup compute a tile of the output.
// Let's make each invocation compute one output element (t_idx, out_dim_idx).
@compute @workgroup_size(16, 16) // Dispatch (ceil(num_tokens/16), ceil(embedding_size/16))
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t_idx = global_id.x;         // Token index
    let out_dim_idx = global_id.y;   // Dimension index in the final embedding_size output

    if (t_idx >= num_toks_data.num_tokens || out_dim_idx >= embedding_size) {
        return;
    }

    let concatenated_hidden_dim = num_heads * head_size;

    // Offset for the input context vector for the current token
    let ctx_token_offset = t_idx * concatenated_hidden_dim;

    // Offset for the row in Wo corresponding to the current output dimension
    // Wo is [embedding_size, concatenated_hidden_dim]
    // Row out_dim_idx starts at out_dim_idx * concatenated_hidden_dim
    let wo_row_offset = out_dim_idx * concatenated_hidden_dim;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < concatenated_hidden_dim; k = k + 1u) {
        sum = sum + ctx_in[ctx_token_offset + k] * wo[wo_row_offset + k];
    }

    // final_out is [num_tokens, embedding_size]
    final_out[t_idx * embedding_size + out_dim_idx] = sum;
}
