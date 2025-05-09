import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./matMul.wgsl?raw";

export default class MatMulBlock extends Block {
    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(

    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        return { resultBuffer: new GPUBuffer(), passes: [] };
    }
}