export const bufferUsageDict = {
  copy_from: GPUBufferUsage.COPY_SRC,
  copy_to: GPUBufferUsage.COPY_DST,
  storage: GPUBufferUsage.STORAGE,
  uniform: GPUBufferUsage.UNIFORM,
  map_read: GPUBufferUsage.MAP_READ,
};

export async function fetchBin(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return new Float32Array(buffer);
}

export function getBufferUsage(usage: string) {
  if (usage in bufferUsageDict) {
    return bufferUsageDict[usage as keyof typeof bufferUsageDict];
  }
  throw new Error(`Invalid buffer usage: ${usage}`);
}
  
const wgSize = (dim: number, size: number): number => Math.min(Math.ceil(dim / size), Infinity);

export function sampleFromDistribution(probs: Float32Array | number[]): number {
  const rand = Math.random();
  let cumulativeProb = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulativeProb += probs[i];
    if (rand < cumulativeProb) {
      return i;
    }
  }
  return probs.length - 1;
}

export function cpuSoftmax(logits: Float32Array, temperature: number = 1.0): Float32Array {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((logit) => Math.exp((logit - maxLogit) / temperature));
  const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
  return new Float32Array(expLogits.map((expLogit) => expLogit / sumExpLogits));
}

export function selectTopK(
  probs: Float32Array | number[], 
  top_k: number
): { topKIndices: number[]; topKProbs: number[] } {
  const sortedIndices = Array.from(probs)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .map(({ index }) => index);
  const topKIndices = sortedIndices.slice(0, top_k);
  const topKProbs = topKIndices.map((index) => probs[index]);
  return { topKIndices, topKProbs };
}

// ----------------------- Matrix Operations -----------------------

export const zeros = (dim: number): Float32Array => new Float32Array(dim).fill(0);

export function transpose(
  array: Float32Array, 
  input_rows: number, 
  input_cols: number
): Float32Array {
  if (array.length !== input_rows * input_cols) {
    console.log(array.length, input_rows, input_cols);
    throw new Error("Transpose dims failed");
  }

  const transpose: number[] = [];
  for (let col = 0; col < input_cols; col++) {
    for (let row = 0; row < input_rows; row++) {
      transpose.push(array[row * input_cols + col]);
    }
  }

  return new Float32Array(transpose);
}

export function leastPrimeFactor(n: number, start: number = 2): number {
  for (let i = start; i <= Math.sqrt(n); i++) {
    if (n % i === 0) return i;
  }
  return n;
}

export function formatAsMatrix(
  floatArray: Float32Array, 
  dimA: number, 
  dimB: number
): number[][] {
  const resultMatrix: number[][] = [];
  for (let i = 0; i < dimA; i++) {
    resultMatrix.push(Array.from(floatArray.slice(i * dimB, (i + 1) * dimB)));
  }
  return resultMatrix;
}