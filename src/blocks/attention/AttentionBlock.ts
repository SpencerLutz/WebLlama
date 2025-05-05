import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./attention.wgsl?raw";

export default class AttentionBlock extends Block {
    constructor(device: GPUDevice) {
        super(device);
    }
    newInstance(
        inputBuffer: GPUBuffer, positionBuffer: GPUBuffer,
        qWieghtsBuffer: GPUBuffer, kWeightsBuffer: GPUBuffer,
        vWeightsBuffer: GPUBuffer, oWeightsBuffer: GPUBuffer,
        embeddingSize: number, numHeads: number, numKVHeads: number,
        headSize: number, roPETheta: number, contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        return { resultBuffer: new GPUBuffer(), passes: [] };
    }
}