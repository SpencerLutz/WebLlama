export type BufferTypeString = "read-only-storage" | "storage" | "uniform";

export type BindingConfig = {
    buffer: GPUBuffer;
    bufferType: BufferTypeString;
};

export type UsageString = "copy_src" | "copy_dst" | "storage" | "uniform" | "map_read";

export type ComputePassConfig = {
    type: "compute";
    pipeline: GPUComputePipeline;
    bindGroups: Array<GPUBindGroup>;
    workgroups: Array<number>;
}

export type CopyPassConfig = {
    type: "copy";
    src: GPUBuffer;
    srcOffset: number;
    dst: GPUBuffer;
    dstOffset: number;
    size: number;
}

export type PassConfig = ComputePassConfig | CopyPassConfig;