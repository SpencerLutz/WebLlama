import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./residual.wgsl?raw";

export default class ResidualBlock extends Block {
    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        hiddenBuffer: GPUBuffer, residualBuffer: GPUBuffer,
        embeddingSize: number, contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        return { resultBuffer: new GPUBuffer(), passes: [] };
    }
}