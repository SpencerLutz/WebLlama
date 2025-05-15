override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32;
override rope_theta: f32;

@group(0) @binding(0) var<storage, read>  attn  : array<f32>;
@group(0) @binding(1) var<storage, read>  v     : array<f32>;
@group(0) @binding(2) var<storage, read_write> ctx : array<f32>;

@group(1) @binding(0) var<storage, read>  ctx_in : array<f32>;
@group(1) @binding(1) var<storage, read>  wo     : array<f32>;
@group(1) @binding(2) var<storage, read_write> out  : array<f32>;

struct NumToksData {
    num_tokens: u32
}
@group(2) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.y;
  let t = gid.x;
  if (h >= num_heads || t >= num_toks_data.num_tokens) { return; }

  // weighted sum: attn[h,t,*] · V[* , (h%num_kv_heads),:]
  var sum:f32 = 0.0;
  for(var s:u32=0; s<context_length; s++){
    let a = attn[(h*context_length+t)*context_length + s];
    let vBase = (s * num_heads + h) * head_size;
    sum += a * v[vBase];
  }
  ctx[(t*num_heads + h)*head_size] = sum;

  // final O projection
  let oBaseIn  = (t * num_heads + h) * head_size;
  let oBaseW   = h * head_size;
  var outSum:f32 = 0.0;
  for(var i:u32=0; i<head_size; i++){
    outSum += ctx[oBaseIn + i] * wo[oBaseW + i * (num_heads*head_size)];
  }
  out[oBaseIn] = outSum;
}
