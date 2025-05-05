export function sampleFromLogits(logits: Float32Array, temperature: number = 1.0): number {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((logit) => Math.exp((logit - maxLogit) / temperature));
    const sumExpLogits = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((expLogit) => expLogit / sumExpLogits);

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