import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./residual.wgsl?raw";

export default class ResidualBlock extends Block {
    private pipeline?: GPUComputePipeline;
    private static readonly WORKGROUP_SIZE = 64; 

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        hiddenBuffer: GPUBuffer,    
        numTokensBuffer: GPUBuffer,
        residualBuffer: GPUBuffer,  
        embeddingSize: number,
        contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        const resultBuffer = this.createBuffer(
            [contextLength, embeddingSize],
            ["storage", "copy_src"], 
            "resultBuffer_residual"
        );

        const inputBindings: BindingConfig[] = [
            { buffer: hiddenBuffer, bufferType: "read-only-storage" },   
            { buffer: residualBuffer, bufferType: "read-only-storage" }, 
        ];
        const { bindGroup: inputBindGroup, bindGroupLayout: inputBindGroupLayout } =
            this.createBindGroup(inputBindings, 'ResidualInputGroup');

        
        const outputBindings: BindingConfig[] = [
            { buffer: resultBuffer, bufferType: "storage" } 
        ];
        const { bindGroup: outputBindGroup, bindGroupLayout: outputBindGroupLayout } =
            this.createBindGroup(outputBindings, 'ResidualOutputGroup');

        const numTokBindGroupConfig: BindingConfig[] = [
            { buffer: numTokensBuffer, bufferType: "uniform" }
        ];
        const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
            = this.createBindGroup(numTokBindGroupConfig, "numTokBuf");

        
        const constants = {
            embedding_size: embeddingSize,
            context_length: contextLength
        };

        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, outputBindGroupLayout, numTokBindGroupLayout], 
                shaderCode,
                constants,
                "residual"
            );
        }

        const totalElements = contextLength * embeddingSize;
        const numWorkgroups = Math.ceil(totalElements / ResidualBlock.WORKGROUP_SIZE);

        const passConfig: PassConfig = {
            pipeline: this.pipeline!,
            bindGroups: [inputBindGroup, outputBindGroup, numTokBindGroup],
            
            numWorkgroups: [(numTokens: number) => Math.ceil((numTokens * embeddingSize) / ResidualBlock.WORKGROUP_SIZE)]
        };

        return { resultBuffer, passes: [passConfig] };
    }
}