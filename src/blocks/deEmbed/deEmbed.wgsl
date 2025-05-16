override embedding_size: u32;
override chunk_size:     u32;
override context_length: u32;
override vocab_size:     u32;

@group(0) @binding(0) var<storage, read> hidden:  array<f32>;
@group(0) @binding(1) var<storage, read> deEmbed: array<f32>;
struct Metadata { chunk_index: u32 }

@group(0) @binding(2) var<uniform> metadata: Metadata;
@group(1) @binding(0) var<storage, read_write> result: array<f32>;

struct NumToksData {
    num_tokens: u32
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= num_toks_data.num_tokens || col >= chunk_size) {
    return;
  }

  // dot(hidden[row, :], deEmbed[:, col])
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < embedding_size; k = k + 1u) {
    let h = hidden[row * embedding_size + k];
    let w = deEmbed[k * chunk_size + col];
    acc = acc + h * w;
  }

  // write into the tile: index = row * chunk_size + col
  let idx = row * chunk_size + col;
  result[idx] = acc;
}
