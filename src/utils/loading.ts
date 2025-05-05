export async function fetchBin(url: string): Promise<Float32Array> {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    return new Float32Array(buffer);
}