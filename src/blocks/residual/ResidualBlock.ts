import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
// Ensure the path to the shader file is correct
import shaderCode from "./residual.wgsl?raw";

export default class ResidualBlock extends Block {
    // Cache the pipeline
    private pipeline?: GPUComputePipeline;
    private static readonly WORKGROUP_SIZE = 64; // Must match the shader

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        hiddenBuffer: GPUBuffer,    // Output of the preceding block (e.g., Attention or SwiGLU)
        residualBuffer: GPUBuffer,  // Input that was fed *into* the preceding block
        embeddingSize: number,
        contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        // --- 1. Create Output Buffer ---
        // The result buffer has the same dimensions as the inputs
        const resultBuffer = this.createBuffer(
            [contextLength, embeddingSize],
            ["storage", "copy_src"], // Shader writes to it, can be copied from
            "resultBuffer_residual"
        );

        // --- 2. Define Bind Group Layouts and Bind Groups ---

        // Input Bind Group (Group 0)
        const inputBindings: BindingConfig[] = [
            { buffer: hiddenBuffer, bufferType: "read-only-storage" },   // Binding 0
            { buffer: residualBuffer, bufferType: "read-only-storage" }, // Binding 1
        ];
        const { bindGroup: inputBindGroup, bindGroupLayout: inputBindGroupLayout } =
            this.createBindGroup(inputBindings, 'ResidualInputGroup');

        // Output Bind Group (Group 1)
        const outputBindings: BindingConfig[] = [
            { buffer: resultBuffer, bufferType: "storage" } // Binding 0
        ];
        const { bindGroup: outputBindGroup, bindGroupLayout: outputBindGroupLayout } =
            this.createBindGroup(outputBindings, 'ResidualOutputGroup');

        // --- 3. Define Shader Constants ---
        const constants = {
            embedding_size: embeddingSize,
            context_length: contextLength
        };

        // --- 4. Create Compute Pipeline (Cache if possible) ---
        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, outputBindGroupLayout], // Layouts for Group 0 and Group 1
                shaderCode,
                constants,
                "residual"
            );
        }

        // --- 5. Configure Compute Pass ---
        // We need enough invocations to cover every element in the buffers.
        const totalElements = contextLength * embeddingSize;
        const numWorkgroups = Math.ceil(totalElements / ResidualBlock.WORKGROUP_SIZE);

        const passConfig: PassConfig = {
            pipeline: this.pipeline!,
            bindGroups: [inputBindGroup, outputBindGroup],
            // Calculate workgroups dynamically based on actual token count if needed
            numWorkgroups: [(numTokens: number) => Math.ceil((numTokens * embeddingSize) / ResidualBlock.WORKGROUP_SIZE)]
        };

        // --- 6. Return Result Buffer and Pass Configuration ---
        return { resultBuffer, passes: [passConfig] };
    }
}