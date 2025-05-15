// rmsNorm.wgsl

override embedding_size: u32;
override epsilon: f32;
override context_length: u32; // Sequence length

// Group 0: Input Buffers
@group(0) @binding(0) var<storage, read> input_buffer: array<f32>; // Input hidden state [context_length, embedding_size]
@group(0) @binding(1) var<storage, read> gamma_buffer: array<f32>; // Scaling weights [embedding_size]

// Group 1: Output Buffer
@group(1) @binding(0) var<storage, read_write> result_buffer: array<f32>; // Output normalized hidden state [context_length, embedding_size]

struct NumToksData {
    num_tokens: u32
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x; // Each invocation handles one token position in the sequence

    // Boundary check
    if (token_idx >= num_toks_data.num_tokens) {
        return;
    }

    // 1. Calculate the sum of squares for the current token's vector
    var sum_sq: f32 = 0.0;
    let start_idx = token_idx * embedding_size;
    for (var i: u32 = 0; i < embedding_size; i = i + 1) {
        let val = input_buffer[start_idx + i];
        sum_sq += val * val;
    }

    // 2. Calculate the RMS normalization factor
    let mean_sq = sum_sq / f32(embedding_size);
    let norm_factor = 1.0 / sqrt(mean_sq + epsilon);

    // 3. Apply normalization and scaling (gamma)
    for (var i: u32 = 0; i < embedding_size; i = i + 1) {
        let current_idx = start_idx + i;
        let normalized_val = input_buffer[current_idx] * norm_factor;
        result_buffer[current_idx] = normalized_val * gamma_buffer[i]; // Apply gamma scaling
    }
}