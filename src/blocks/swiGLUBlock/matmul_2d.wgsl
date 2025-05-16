override M: u32;
override K: u32;
override N: u32;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct NumToksData {
    num_tokens: u32
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; let col = gid.y;
  if (row >= num_toks_data.num_tokens || col >= N) { return; }
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    sum = sum + A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}
