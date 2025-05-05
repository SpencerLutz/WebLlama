// elementwise: C[i] = silu(A[i]) * B[i]
override L: u32; // total elements = M×N

@group(0) @binding(0) var<storage, read_write> A: array<f32>; // gate
@group(0) @binding(1) var<storage, read> B: array<f32>;      // up

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= L) { return; }
  let x = A[i];
  // silu(x) = x * sigmoid(x)
  let s = 1.0 / (1.0 + exp(-x));
  A[i] = x * s * B[i];
}
