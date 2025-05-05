import { BufferTypeString, BindingConfig, UsageString } from '../types';
import { bufferUsageDict, calcBufferSize } from '../utils/buffer';

export class Block {
    private device: GPUDevice;
    static bindGroupLayoutCache: Map<string, GPUBindGroupLayout> = new Map();

    constructor(device: GPUDevice) {
        this.device = device;
    }

    /**
     * Creates (or retrieves from cache) and returns a bind group layout 
     * for the provided buffer types.
     * 
     * @param bufferTypes - Array of buffer type strings ("read-only-storage" | "storage" | "uniform")
     * @returns Corresponding bind group layout
     */
    createBindGroupLayout(bufferTypes: Array<BufferTypeString>): GPUBindGroupLayout {
        const cacheKey = bufferTypes.join('_');

        if (Block.bindGroupLayoutCache.has(cacheKey)) {
            return Block.bindGroupLayoutCache.get(cacheKey)!;
        } else {
            const bindGroupLayout = this.device.createBindGroupLayout({
                entries: bufferTypes.map((bufferType, bindingIndex) => ({
                    binding: bindingIndex,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: bufferType },
                }))
            });
            Block.bindGroupLayoutCache.set(cacheKey, bindGroupLayout);
            return bindGroupLayout;
        }
    }

    /**
     * Creates and returns a bind group and its layout.
     * 
     * @param bindingConfigs - Array of binding configs (buffer and buffer type)
     * @returns Object containing the created bind group and its layout.
     */
    createBindGroup(bindingConfigs: Array<BindingConfig>, label?: string): {
        bindGroup: GPUBindGroup;
        bindGroupLayout: GPUBindGroupLayout;
    } {
        const bufferTypes = bindingConfigs.map(config => config.bufferType);
        const bindGroupLayout = this.createBindGroupLayout(bufferTypes);

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: bindingConfigs.map(({ buffer }, bindingIndex) => ({
                binding: bindingIndex,
                resource: { buffer },
            })),
            label
        });
        return { bindGroup, bindGroupLayout };
    }

    /**
     * Creates a pipeline.
     * 
     * @param bindGroupLayouts - Array of bind group layouts (from createBindGroup())
     * @param shaderCode - WGSL shader code as string
     * @param constants - Optional shader constants
     * @returns The created compute pipeline
     */
    createPipeline(bindGroupLayouts: Array<GPUBindGroupLayout>, shaderCode: string, constants = {}, label: string): 
        GPUComputePipeline
    {
        const pipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts }),
            compute: {
                module: this.device.createShaderModule({ code: shaderCode, label: label}),
                entryPoint: "main",
                constants,
            }
        });

        return pipeline
    }

    /**
     * Creates a new buffer with the specified shape and usage flags.
     * 
     * @param shape - 1 or 2 element array containing buffer dimensions
     * @param usages - Array of buffer usage flag strings
     * @returns The created GPU buffer
     */
    createBuffer(shape: Array<number>, usages: Array<UsageString>, label: string): GPUBuffer {
        return this.device.createBuffer({
            size: calcBufferSize(shape[0], shape[1]),
            usage: usages.map((usage) => bufferUsageDict[usage]).reduce((a, b) => a | b),
            label: label
        });
    }

    /**
     * Writes data into buffer.
     * 
     * @param buffer - The buffer to write to
     * @param data - The data to write to the buffer
     * @param offset - Optional offset into the buffer
     */
    writeBuffer(buffer: GPUBuffer, data: Array<number>, offset: number = 0): void {
        const u32Data = new Uint32Array(data);
        this.device.queue.writeBuffer(buffer, offset, u32Data);
    }
}