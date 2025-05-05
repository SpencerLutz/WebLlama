import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./embed.wgsl?raw";

export default class EmbedBlock extends Block {
    /* Only caching 1 pipeline because each one has
    the same bindGroupLayout, shader, and constants */
    private pipeline?: GPUComputePipeline

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputTokens: GPUBuffer, embeddingsBuffers: Array<GPUBuffer>,
        embeddingSize: number, chunkSize: number, numChunks: number, contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        const metadataBuffers = Array.from({ length: numChunks }, (_, i) => 
            this.createBuffer([1], ["uniform", "copy_dst"]));
        metadataBuffers.forEach((buffer, i) => this.writeBuffer(buffer, [i]));

        const inputBindGroupsAndLayouts = Array.from({ length: numChunks }, (_, i) => 
            this.createBindGroup([
                { buffer: inputTokens, bufferType: "read-only-storage" },
                { buffer: embeddingsBuffers[i], bufferType: "read-only-storage" },
                { buffer: metadataBuffers[i], bufferType: "uniform" },
            ]));
        const inputBindGroupLayout = inputBindGroupsAndLayouts[0].bindGroupLayout;
        const inputBindGroups = inputBindGroupsAndLayouts.map(bg => bg.bindGroup);

        const resultBuffer = this.createBuffer([contextLength, embeddingSize], ["storage", "copy_src"]);
        const outputBindGroupConfig: BindingConfig[] = [
            { buffer: resultBuffer, bufferType: "storage" }
        ];
        const { bindGroup: outputBindGroup, bindGroupLayout: outputBindGroupLayout } 
            = this.createBindGroup(outputBindGroupConfig);

        const constants = { 
            embedding_size: embeddingSize, 
            chunk_size: chunkSize, 
            context_length: contextLength 
        };

        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, outputBindGroupLayout], 
                shaderCode,
                constants
            );
        }

        const workgroupSize = 64;
        const passes: Array<PassConfig> = inputBindGroups.map(inputBindGroup => ({
            pipeline: this.pipeline,
            bindGroups: [inputBindGroup, outputBindGroup],
            numWorkgroups: [(numTokens) => Math.ceil(numTokens / workgroupSize)]
        } as PassConfig));

        return { resultBuffer, passes };
    }
}