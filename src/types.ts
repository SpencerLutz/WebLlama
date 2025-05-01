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