import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
// Make sure the path to the shader file is correct
import shaderCode from "./rmsNorm.wgsl?raw";

export default class RMSNormBlock extends Block {
    // Cache the pipeline to avoid recompilation
    private pipeline?: GPUComputePipeline;
    private static readonly WORKGROUP_SIZE = 64; // Match the shader

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputBuffer: GPUBuffer, // Hidden state from previous layer
        gammaBuffer: GPUBuffer, // Learnable scaling weights
        embeddingSize: number,  // n_embd
        epsilon: number,        // norm_eps
        contextLength: number   // n_ctx
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        // --- 1. Define Bind Group Layouts and Bind Groups ---

        // Input Bind Group (Group 0)
        const inputBindings: BindingConfig[] = [
            { buffer: inputBuffer, bufferType: "read-only-storage" }, // Input hidden state
            { buffer: gammaBuffer, bufferType: "read-only-storage" }, // Gamma weights
        ];
        const { bindGroup: inputBindGroup, bindGroupLayout: inputBindGroupLayout } =
            this.createBindGroup(inputBindings, "RmsNormInputGroup");

        // Output Bind Group (Group 1)
        // Create the result buffer where the normalized output will be written
        const resultBuffer = this.createBuffer(
            [contextLength, embeddingSize], // Shape: [seq_length, embedding_dim]
            ["storage", "copy_src"],         // Usage: Shader writes to it, can be copied from
            "resultBuffer_norm"
        );
        const outputBindings: BindingConfig[] = [
            { buffer: resultBuffer, bufferType: "storage" } // Output normalized hidden state
        ];
        const { bindGroup: outputBindGroup, bindGroupLayout: outputBindGroupLayout } =
            this.createBindGroup(outputBindings, "RmsNormOutputGroup");

        // --- 2. Define Shader Constants ---
        const constants = {
            embedding_size: embeddingSize,
            epsilon: epsilon,
            context_length: contextLength
        };

        // --- 3. Create Compute Pipeline (Cache if possible) ---
        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, outputBindGroupLayout], // Order matches @group in shader
                shaderCode,
                constants,
                "RMSNorm"
            );
        }

        // --- 4. Configure Compute Pass ---
        // We dispatch one workgroup invocation per token in the sequence.
        // The shader then handles the normalization across the embedding dimension for that token.
        const numWorkgroupsX = Math.ceil(contextLength / RMSNormBlock.WORKGROUP_SIZE);

        const passConfig: PassConfig = {
            pipeline: this.pipeline!, // Use the cached or newly created pipeline
            bindGroups: [inputBindGroup, outputBindGroup], // Assign bind groups to their slots (0 and 1)
            numWorkgroups: [(numTokens: number) => Math.ceil(numTokens / RMSNormBlock.WORKGROUP_SIZE)] // Dynamic workgroup calculation based on actual input tokens
        };

        // --- 5. Return Result Buffer and Pass Configuration ---
        // RMSNorm typically only needs one pass
        return { resultBuffer, passes: [passConfig] };
    }
}