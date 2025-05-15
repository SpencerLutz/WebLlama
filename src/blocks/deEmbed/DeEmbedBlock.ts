import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";
import shaderCode from "./deEmbed.wgsl?raw";

export default class DeEmbedBlock extends Block {
  private pipeline?: GPUComputePipeline;

  constructor(device: GPUDevice) {
    super(device);
  }

  newInstance(
    inputBuffer: GPUBuffer,
    numTokensBuffer: GPUBuffer,
    deEmbeddingsBuffers: Array<GPUBuffer>,
    embeddingSize: number,
    vocabSize: number,
    chunkSize: number,
    numChunks: number,
    contextLength: number
  ): {
    // now: one small resultBuffer per chunk
    resultBuffers: GPUBuffer[];
    passes: Array<PassConfig>;
  } {
    // 1) make per-chunk result buffers of size [contextLength × chunkSize]
    const resultBuffers = Array.from({ length: numChunks }, (_, i) =>
      this.createBuffer(
        [contextLength, chunkSize],
        ["storage", "copy_src"],
        `resultBuffer_deembed_chunk${i}`
      )
    );

    // 2) uniform metadata buffers for chunk_index
    const metadataBuffers = Array.from({ length: numChunks }, (_, i) => {
      const buf = this.createBuffer([1], ["uniform", "copy_dst"], "metadataBuffer");
      this.writeBuffer(buf, [i]);
      return buf;
    });

    // 3) bind groups for input (hidden, deEmbed chunk, metadata)
    const inputBGs = deEmbeddingsBuffers.map((embBuf, i) => {
      const { bindGroup, bindGroupLayout } = this.createBindGroup([
        { buffer: inputBuffer, bufferType: "read-only-storage" },
        { buffer: embBuf,    bufferType: "read-only-storage" },
        { buffer: metadataBuffers[i], bufferType: "uniform" },
      ], "deEmbedBuffer");
      return { bindGroup, bindGroupLayout };
    });

    // 4) bind groups for each chunk's resultBuffer
    const resultBGs = resultBuffers.map(buf =>
      this.createBindGroup([{ buffer: buf, bufferType: "storage" }], "resultBuffer")
    );

    const numTokBindGroupConfig: BindingConfig[] = [
        { buffer: numTokensBuffer, bufferType: "uniform" }
    ];
    const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
        = this.createBindGroup(numTokBindGroupConfig, "numTokBuf");

    // 5) compile the pipeline once, using the first layouts
    const constants = {
      embedding_size: embeddingSize,
      chunk_size:     chunkSize,
      context_length: contextLength,
      vocab_size:     vocabSize,
    };
    if (!this.pipeline) {
      this.pipeline = this.createPipeline(
        [ inputBGs[0].bindGroupLayout, resultBGs[0].bindGroupLayout, numTokBindGroupLayout ],
        shaderCode,
        constants,
        "deEmbed"
      );
    }

    // 6) dispatch sizes
    const workgroupSizeX = 16, workgroupSizeY = 16;
    const dispatchX = (numTokens: number) => Math.ceil(numTokens / workgroupSizeX);
    const dispatchY = Math.ceil(chunkSize     / workgroupSizeY);

    // 7) one compute‐pass per chunk, binding its own result buffer
    const passes: PassConfig[] = inputBGs.map(({ bindGroup }, i) => ({
      pipeline: this.pipeline!,
      bindGroups: [ bindGroup, resultBGs[i].bindGroup, numTokBindGroup ],
      numWorkgroups: [dispatchX, dispatchY],
    }));

    return {
      resultBuffers,
      passes
    };
  }
}
