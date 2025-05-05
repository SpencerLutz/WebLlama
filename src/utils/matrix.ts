export function zeros(dim: number): Float32Array {
    return new Float32Array(dim).fill(0);
}

export function transpose(
    array: Float32Array, 
    input_rows: number, 
    input_cols: number
): Float32Array {
    if (array.length !== input_rows * input_cols) {
        throw new Error(`Transpose dims incorrect: ${array.length}, ${input_rows}x${input_cols}`);
    }
  
    const transpose = [];
    for (let col = 0; col < input_cols; col++) {
        for (let row = 0; row < input_rows; row++) {
            transpose.push(array[row * input_cols + col]);
        }
    }
  
    return new Float32Array(transpose);
}