import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";

import shaderCode from "./rmsNorm.wgsl?raw";

export default class RMSNormBlock extends Block {
    
    private pipeline?: GPUComputePipeline;
    private static readonly WORKGROUP_SIZE = 64; 

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputBuffer: GPUBuffer, 
        numTokensBuffer: GPUBuffer,
        gammaBuffer: GPUBuffer, 
        embeddingSize: number,  
        epsilon: number,        
        contextLength: number   
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        const inputBindings: BindingConfig[] = [
            { buffer: inputBuffer, bufferType: "read-only-storage" }, 
            { buffer: gammaBuffer, bufferType: "read-only-storage" }, 
        ];
        const { bindGroup: inputBindGroup, bindGroupLayout: inputBindGroupLayout } =
            this.createBindGroup(inputBindings, "RmsNormInputGroup");

        const resultBuffer = this.createBuffer(
            [contextLength, embeddingSize], 
            ["storage", "copy_src"],         
            "resultBuffer_norm"
        );
        const outputBindings: BindingConfig[] = [
            { buffer: resultBuffer, bufferType: "storage" } 
        ];
        const { bindGroup: outputBindGroup, bindGroupLayout: outputBindGroupLayout } =
            this.createBindGroup(outputBindings, "RmsNormOutputGroup");

        const numTokBindGroupConfig: BindingConfig[] = [
            { buffer: numTokensBuffer, bufferType: "uniform" }
        ];
        const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
            = this.createBindGroup(numTokBindGroupConfig, "numTokBuf");

        
        const constants = {
            embedding_size: embeddingSize,
            epsilon: epsilon,
            context_length: contextLength
        };

        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, outputBindGroupLayout, numTokBindGroupLayout], 
                shaderCode,
                constants,
                "RMSNorm"
            );
        }

        const numWorkgroupsX = Math.ceil(contextLength / RMSNormBlock.WORKGROUP_SIZE);

        const passConfig: PassConfig = {
            pipeline: this.pipeline!, 
            bindGroups: [inputBindGroup, outputBindGroup, numTokBindGroup], 
            numWorkgroups: [(numTokens: number) => Math.ceil(numTokens / RMSNormBlock.WORKGROUP_SIZE)] 
        };

        return { resultBuffer, passes: [passConfig] };
    }
}