export type BufferTypeString = "read-only-storage" | "storage" | "uniform";

export type BindingConfig = {
    buffer: GPUBuffer;
    bufferType: BufferTypeString;
};

export type UsageString = "copy_src" | "copy_dst" | "storage" | "uniform" | "map_read" | "indirect";

export type FixedWorkgroupsConfig = {
    type: "fixed";
    workgroups: Array<number>;
}

export type IndirectWorkgroupsConfig = {
    type: "indirect";
    workgroups: GPUBuffer;
}

export type WorkgroupsConfig = FixedWorkgroupsConfig | IndirectWorkgroupsConfig;

export type PassConfig = {
    pipeline: GPUComputePipeline;
    bindGroups: Array<GPUBindGroup>;
    workgroupsConfig: WorkgroupsConfig;
}