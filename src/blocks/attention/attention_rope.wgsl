override embedding_size: u32;
override num_heads: u32;
override num_kv_heads: u32;
override head_size: u32;
override context_length: u32; // Max sequence length for attn scores & V buffer
override rope_theta: f32;
override concatenated_head_dim: u32;

// Placeholders for Llama3-specific RoPE scaling constants from config
// These would need to be passed in, likely via a uniform buffer or more overrides
override original_max_position_embeddings: f32 = 8192.0; // example default
override rope_scaling_factor: f32 = 32.0;             // example default, from "factor"
override rope_scaling_low_freq_factor: f32 = 1.0;     // example default
override rope_scaling_high_freq_factor: f32 = 4.0;    // example default


@group(0) @binding(0) var<storage, read_write> q : array<f32>; // Shape [num_tokens, num_heads, head_size]
@group(0) @binding(1) var<storage, read_write> k : array<f32>; // Shape [num_tokens, num_kv_heads, head_size]
@group(0) @binding(2) var<storage, read>         pos : array<u32>; // Positions for each token in sequence

struct NumToksData {
    num_tokens: u32
}
@group(1) @binding(0) var<uniform> num_toks_data: NumToksData;

@compute @workgroup_size(8,8) // workgroup_size x = for t, y = for h_idx (q_heads or kv_heads)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;             // Token index in the sequence
  let head_idx = gid.y;      // Head index (can be for Q or K heads)

  if (t >= num_toks_data.num_tokens) { return; }

  let p = f32(pos[t]); // Current token's position

  // Apply RoPE to Q heads
  if (head_idx < num_heads) {
    let q_base = (t * num_heads + head_idx) * head_size;
    for (var i: u32 = 0u; i < head_size; i = i + 2u) {
        let freq_exponent = f32(i) / f32(head_size);
        var inv_freq = 1.0 / pow(rope_theta, freq_exponent);

        // TODO: Apply Llama3 specific RoPE scaling to inv_freq if needed
        // inv_freq = apply_llama3_rope_scaling(inv_freq, i, p); // Pass dimension 'i' and position 'p'

        let angle = p * inv_freq;
        let c = cos(angle);
        let s = sin(angle);

        let q0 = q[q_base + i];
        let q1 = q[q_base + i + 1u];

        q[q_base + i]       = q0 * c - q1 * s;
        q[q_base + i + 1u]  = q0 * s + q1 * c;
    }
  }

  // Apply RoPE to K heads (num_kv_heads might be different from num_heads)
  // The dispatch for RoPE in AttentionBlock.ts is `numWorkgroups: [l => l, numHeads]`.
  // This means gid.y (head_idx) goes up to num_heads. We need to ensure we only process valid K heads.
  // K heads are typically a subset or repetition of Q heads for RoPE application if num_kv_heads < num_heads
  // Llama applies RoPE to the logical K heads.
  // If K heads are fewer, we might need to map q_head_idx to kv_head_idx for applying RoPE,
  // or ensure the K buffer is structured to align with the first num_kv_heads of Q for RoPE application.
  // Assuming RoPE is applied independently up to num_kv_heads:
  if (head_idx < num_kv_heads) { // Process if head_idx is a valid KV head index
    let k_base = (t * num_kv_heads + head_idx) * head_size; // Use num_kv_heads for indexing k_buffer
    for (var i: u32 = 0u; i < head_size; i = i + 2u) {
        let freq_exponent = f32(i) / f32(head_size);
        var inv_freq = 1.0 / pow(rope_theta, freq_exponent);

        // TODO: Apply Llama3 specific RoPE scaling to inv_freq
        // inv_freq = apply_llama3_rope_scaling(inv_freq, i, p);

        let angle = p * inv_freq;
        let c = cos(angle);
        let s = sin(angle);

        let k0 = k[k_base + i];
        let k1 = k[k_base + i + 1u];

        k[k_base + i]       = k0 * c - k1 * s;
        k[k_base + i + 1u]  = k0 * s + k1 * c;
    }
  }
}