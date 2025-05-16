override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32;
override rope_theta: f32;

@group(0) @binding(0) var<storage, read>       x     : array<f32>;
@group(0) @binding(1) var<storage, read>       wq    : array<f32>;
@group(0) @binding(2) var<storage, read>       wk    : array<f32>;
@group(0) @binding(3) var<storage, read>       wv    : array<f32>;
@group(0) @binding(4) var<storage, read_write> q_out : array<f32>;
@group(0) @binding(5) var<storage, read_write> k_out : array<f32>;
@group(0) @binding(6) var<storage, read_write> v_out : array<f32>;

struct NumToksData {
    num_tokens: u32
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let h = gid.y;
  if (t >= num_toks_data.num_tokens) { return; }

  // Q projection
  if (h < num_heads) {
    let outBase = (t * num_heads + h) * head_size;
    let wBase   = h * head_size;
    var sum:f32 = 0.0;
    for(var i:u32=0; i<embedding_size; i++) {
      sum += x[t * embedding_size + i] * wq[wBase + i * (num_heads*head_size)];
    }
    q_out[outBase] = sum;
  }

  // K projection
  if (h < num_kv_heads) {
    let outBase = (t * num_kv_heads + h) * head_size;
    let wBase   = h * head_size;
    var sumk:f32 = 0.0;
    for(var i:u32=0; i<embedding_size; i++) {
      sumk += x[t * embedding_size + i] * wk[wBase + i * (num_kv_heads*head_size)];
    }
    k_out[outBase] = sumk;

    // V projection
    var sumv:f32 = 0.0;
    for(var i:u32=0; i<embedding_size; i++) {
      sumv += x[t * embedding_size + i] * wv[wBase + i * (num_kv_heads*head_size)];
    }
    v_out[outBase] = sumv;
  }
}
