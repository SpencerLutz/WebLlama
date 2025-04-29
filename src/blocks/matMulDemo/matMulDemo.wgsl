@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;

@group(1) @binding(0) var<storage, read_write> result: array<f32>;
@group(1) @binding(1) var<uniform, read> meta: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_index = global_id.x;
    let y_index = global_id.y;

    let n: u32 = meta[0];
    let k: u32 = meta[1];
    let m: u32 = meta[2];

    if (x_index >= n || y_index >= m) {
        return;
    }

    var sum: f32 = 0;
    for (var i: u32 = 0; i < k; i = i + 1) {
        let a_index = x_index * k + i;
        let b_index = i * m + y_index;
        sum = sum + matrixA[a_index] * matrixB[b_index];
    }
    let c_index = x_index * m + y_index;
    result[c_index] = sum;
}