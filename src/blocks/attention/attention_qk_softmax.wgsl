override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32;
override rope_theta: f32;

@group(0) @binding(0) var<storage, read>  q    : array<f32>;
@group(0) @binding(1) var<storage, read>  k    : array<f32>;
@group(0) @binding(2) var<storage, read_write> logits : array<f32>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  let t = gid.y;
  if (h >= num_heads || t >= context_length) { return; }

  let qBase = (t * num_heads + h) * head_size;
  // compute max for numeric stability
  var m = -1e9;
  for(var s:u32=0; s<context_length; s++){
    var dotp = 0.0;
    let kBase = (s * num_heads + (h % num_heads)) * head_size;
    for(var i:u32=0; i<head_size; i++){
      dotp += q[qBase+i] * k[kBase+i];
    }
    m = max(m, dotp);
  }
  // compute sum exp(...)
  var sumExp = 0.0;
  for(var s:u32=0; s<context_length; s++){
    let kBase = (s * num_heads + (h % num_heads)) * head_size;
    var dotp = 0.0;
    for(var i:u32=0; i<head_size; i++){
      dotp += q[qBase+i] * k[kBase+i];
    }
    let e = exp(dotp - m);
    sumExp += e;
    logits[(h * context_length + t)*context_length + s] = e;
  }
  // normalize
  for(var s:u32=0; s<context_length; s++){
    logits[(h * context_length + t)*context_length + s] /= sumExp;
  }
}
