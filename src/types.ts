export type BufferTypeString = "read-only-storage" | "storage" | "uniform";

export type BindingConfig = {
    buffer: GPUBuffer;
    bufferType: BufferTypeString;
};

export type UsageString = "copy_src" | "copy_dst" | "storage" | "uniform" | "map_read" | "indirect";

export type WorkgroupDim = number | ((n: number) => number);

export type PassConfig = {
    pipeline: GPUComputePipeline;
    bindGroups: Array<GPUBindGroup>;
    numWorkgroups: Array<WorkgroupDim>;
}

export type LayerBuffers = {
    normAttentionGammaBuffer: GPUBuffer;    // Input Norm weight [n_embd]
    normLinearGammaBuffer: GPUBuffer;       // FFN Norm weight [n_embd]
    qWeightsBuffer: GPUBuffer;  // [n_embd, n_embd]
    kWeightsBuffer: GPUBuffer;  // [n_kv_head * head_size, n_embd]
    vWeightsBuffer: GPUBuffer;  // [n_kv_head * head_size, n_embd]
    oWeightsBuffer: GPUBuffer;  // [n_embd, n_embd]
    w1WeightsBuffer: GPUBuffer; // Gate proj [intermediate_size, n_embd]
    w3WeightsBuffer: GPUBuffer; // Up proj   [intermediate_size, n_embd]
    w2WeightsBuffer: GPUBuffer; // Down proj [n_embd, intermediate_size]
}

export type LoadedModelBuffers = {
    embeddingsBuffers: GPUBuffer[];
    deEmbeddingsBuffers: GPUBuffer[];   // Transposed [n_embd, vocab_chunk_size]
    layer_buffers: LayerBuffers[];
    normGammaBuffer: GPUBuffer;         // RMSNorm weight [n_embd]
}

export type ModelParams = {
    n_embd: number;
    n_layer: number;
    n_head: number;
    n_kv_head: number;
    head_size: number;
    vocab_size: number;
    intermediate_size: number;
    n_ctx: number;
    norm_eps: number;
    rope_theta: number;
    vocab_chunk_size: number;
    vocab_chunk_instances: number;
}