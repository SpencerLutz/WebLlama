import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./embed.wgsl?raw";

export default class EmbedBlock extends Block {
    private pipeline?: GPUComputePipeline

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputTokens: GPUBuffer, numTokensBuffer: GPUBuffer, embeddingsBuffers: Array<GPUBuffer>,
        embeddingSize: number, chunkSize: number, numChunks: number, maxContextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        const metadataBuffers = Array.from({ length: numChunks }, (_, i) => 
            this.createBuffer([1], ["uniform", "copy_dst"], "metadataBuffer"));
        metadataBuffers.forEach((buffer, i) => this.writeBuffer(buffer, [i]));

        const inputBindGroupsAndLayouts = Array.from({ length: numChunks }, (_, i) => 
            this.createBindGroup([
                { buffer: inputTokens, bufferType: "read-only-storage" },
                { buffer: embeddingsBuffers[i], bufferType: "read-only-storage" },
                { buffer: metadataBuffers[i], bufferType: "uniform" },
            ], "inputBuf"));
        const inputBindGroupLayout = inputBindGroupsAndLayouts[0].bindGroupLayout;
        const inputBindGroups = inputBindGroupsAndLayouts.map(bg => bg.bindGroup);

        const resultBuffer = this.createBuffer([maxContextLength, embeddingSize], ["storage", "copy_src"], "resultBuffer_embed");
        const outputBindGroupConfig: BindingConfig[] = [
            { buffer: resultBuffer, bufferType: "storage" }
        ];
        const { bindGroup: outputBindGroup, bindGroupLayout: outputBindGroupLayout } 
            = this.createBindGroup(outputBindGroupConfig, "embedBuf");

        const numTokBindGroupConfig: BindingConfig[] = [
            { buffer: numTokensBuffer, bufferType: "uniform" }
        ];
        const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
            = this.createBindGroup(numTokBindGroupConfig, "numTokBuf");

        const constants = { 
            embedding_size: embeddingSize, 
            chunk_size: chunkSize, 
            max_context_length: maxContextLength 
        };

        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, outputBindGroupLayout, numTokBindGroupLayout], 
                shaderCode,
                constants,
                "embed"
            );
        }

        const workgroupSize = 64;
        const passes: Array<PassConfig> = inputBindGroups.map(inputBindGroup => ({
            pipeline: this.pipeline,
            bindGroups: [inputBindGroup, outputBindGroup, numTokBindGroup],
            numWorkgroups: [(numTokens) => Math.ceil(numTokens / workgroupSize)]
        } as PassConfig));

        return { resultBuffer, passes };
    }
}