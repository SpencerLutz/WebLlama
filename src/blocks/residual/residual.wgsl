override embedding_size: u32;
override context_length: u32;

@group(0) @binding(0) var<storage, read> hidden_input: array<f32>;
@group(0) @binding(1) var<storage, read> residual_input: array<f32>;

@group(1) @binding(0) var<storage, read_write> output_buffer: array<f32>;

struct NumToksData {
    num_tokens: u32
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let element_idx = global_id.x;
    let total_elements = num_toks_data.num_tokens * embedding_size;

    if (element_idx >= total_elements) {
        return;
    }

    output_buffer[element_idx] = hidden_input[element_idx] + residual_input[element_idx];
}