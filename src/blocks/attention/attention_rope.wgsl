override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32;
override rope_theta: f32;

@group(0) @binding(0) var<storage, read_write> q : array<f32>;
@group(0) @binding(1) var<storage, read_write> k : array<f32>;
@group(0) @binding(2) var<storage, read>         pos : array<u32>;

struct NumToksData {
    num_tokens: u32
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

fn rotate2(v: vec2<f32>) -> vec2<f32> { return vec2<f32>(-v.y, v.x); }

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let h = gid.y;
  if (t >= num_toks_data.num_tokens || h >= num_heads) { return; }

  let base = (t * num_heads + h) * head_size;
  let p = f32(pos[t]);
  for(var i:u32=0; i<head_size; i+=2) {
    let angle = p * pow(rope_theta, f32(i)/f32(head_size));
    let c = cos(angle);
    let s = sin(angle);
    let vq = vec2<f32>(q[base+i], q[base+i+1]);
    let vk = vec2<f32>(k[base+i], k[base+i+1]);
    let rq = vec2<f32>(vq.x*c - vq.y*s, vq.x*s + vq.y*c);
    let rk = vec2<f32>(vk.x*c - vk.y*s, vk.x*s + vk.y*c);
    q[base+i]   = rq.x; q[base+i+1] = rq.y;
    k[base+i]   = rk.x; k[base+i+1] = rk.y;
  }
}
