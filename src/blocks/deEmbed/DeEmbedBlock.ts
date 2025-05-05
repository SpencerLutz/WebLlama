import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./deEmbed.wgsl?raw";

export default class DeEmbedBlock extends Block {
    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        inputBuffer: GPUBuffer, deEmbeddingsBuffers: Array<GPUBuffer>,
        embeddingSize: number, vocabSize: number, chunkSize: number,
        numChunks: number, contextLength: number
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        return { resultBuffer: new GPUBuffer(), passes: [] };
    }
}