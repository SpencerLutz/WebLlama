// elementwise: C[i] = silu(A[i]) * B[i]
override N1: u32; 

@group(0) @binding(0) var<storage, read_write> A: array<f32>; // gate
@group(0) @binding(1) var<storage, read> B: array<f32>;      // up

struct NumToksData {
    num_tokens: u32
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= num_toks_data.num_tokens * N1) { return; }
  let x = A[i];
  // silu(x) = x * sigmoid(x)
  let s = 1.0 / (1.0 + exp(-x));
  A[i] = x * s * B[i];
}
