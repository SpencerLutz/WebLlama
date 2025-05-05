import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./rmsNorm.wgsl?raw";

export default class RMSNormBlock extends Block {
    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputBuffer: GPUBuffer, gammaBuffer: GPUBuffer,
        embeddingSize: number, epsilon: number, contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        // Implement here
        return { resultBuffer: new GPUBuffer(), passes: [] };
    }
}