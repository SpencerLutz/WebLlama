override embedding_size: u32;
override chunk_size: u32;
override context_length: u32;

@group(0) @binding(0) var<storage, read> input_tokens: array<u32>;
@group(0) @binding(1) var<storage, read> embeddings: array<f32>;
struct Metadata {
    chunk_index: u32
}
@group(0) @binding(2) var<uniform> metadata: Metadata;

@group(1) @binding(0) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= context_length) {
        return;
    }

    let token = input_tokens[token_idx];
    if (token < metadata.chunk_index * chunk_size 
        || token >= (metadata.chunk_index + 1) * chunk_size) {
        return;
    }

    for (var i: u32 = 0; i < embedding_size; i = i + 1) {
        let embed_index = (token - metadata.chunk_index * chunk_size) * embedding_size + i;
        let result_index = token_idx * embedding_size + i;
        result[result_index] = embeddings[embed_index];
    }
}