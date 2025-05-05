import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./swiGLU.wgsl?raw";

export default class SwiGLUBlock extends Block {
    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputBuffer: GPUBuffer, w1WeightsBuffer: GPUBuffer,
        w3WeightsBuffer: GPUBuffer, w2WeightsBuffer: GPUBuffer,
        embeddingSize: number, intermediateSize: number, contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        return { resultBuffer: new GPUBuffer(), passes: [] };
    }
}