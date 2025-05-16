import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";

import qkvShader from "./attention_qkv.wgsl?raw";
import ropeShader from "./attention_rope.wgsl?raw";
import attnShader from "./attention_qk_softmax.wgsl?raw";
import outShader  from "./attention_out.wgsl?raw";

export default class AttentionBlock extends Block {
  private qkvPipeline?: GPUComputePipeline;
  private ropePipeline?: GPUComputePipeline;
  private attnPipeline?: GPUComputePipeline;
  private outPipeline?: GPUComputePipeline;

  constructor(device: GPUDevice) {
    super(device);
  }

  newInstance(
    inputBuffer: GPUBuffer,
    numTokensBuffer: GPUBuffer,
    positionBuffer: GPUBuffer,
    qWeights: GPUBuffer,
    kWeights: GPUBuffer,
    vWeights: GPUBuffer,
    oWeights: GPUBuffer,
    embeddingSize: number,
    numHeads: number,
    numKVHeads: number,
    headSize: number,
    ropeTheta: number,
    contextLength: number
  ): { resultBuffer: GPUBuffer; passes: PassConfig[] } {
    const qBuffer     = this.createBuffer([contextLength, numHeads * headSize], ["storage"], "qBuffer");
    const kBuffer     = this.createBuffer([contextLength, numKVHeads * headSize], ["storage"], "kBuffer");
    const vBuffer     = this.createBuffer([contextLength, numKVHeads * headSize], ["storage"], "vBuffer");
    const attnBuffer  = this.createBuffer([numHeads, contextLength, contextLength], ["storage"], "attnBuffer");
    const outContext  = this.createBuffer([contextLength, numHeads * headSize], ["storage"], "outContext");
    const outContext2  = this.createBuffer([contextLength, numHeads * headSize], ["storage"], "outContext2");
    const resultBuffer = this.createBuffer([contextLength, numHeads * headSize], ["storage", "copy_src"], "resultBuffer_attn");

    const qkvBinds: BindingConfig[] = [
      { buffer: inputBuffer,  bufferType: "read-only-storage" },
      { buffer: qWeights,     bufferType: "read-only-storage" },
      { buffer: kWeights,     bufferType: "read-only-storage" },
      { buffer: vWeights,     bufferType: "read-only-storage" },
      { buffer: qBuffer,      bufferType: "storage" },
      { buffer: kBuffer,      bufferType: "storage" },
      { buffer: vBuffer,      bufferType: "storage" },
    ];
    const { bindGroup: qkvGroup, bindGroupLayout: qkvLayout } 
      = this.createBindGroup(qkvBinds, "qkv");

    const ropeBinds: BindingConfig[] = [
      { buffer: qBuffer,    bufferType: "storage" },
      { buffer: kBuffer,    bufferType: "storage" },
      { buffer: positionBuffer, bufferType: "read-only-storage" },
    ];
    const { bindGroup: ropeGroup, bindGroupLayout: ropeLayout }
      = this.createBindGroup(ropeBinds, "rope");

    const attnBinds: BindingConfig[] = [
      { buffer: qBuffer,   bufferType: "read-only-storage" },
      { buffer: kBuffer,   bufferType: "read-only-storage" },
      { buffer: attnBuffer, bufferType: "storage" },
    ];
    const { bindGroup: attnGroup, bindGroupLayout: attnLayout }
      = this.createBindGroup(attnBinds, "attn");

    const outBinds1: BindingConfig[] = [
      { buffer: attnBuffer, bufferType: "read-only-storage" },
      { buffer: vBuffer,    bufferType: "read-only-storage" },
      { buffer: outContext, bufferType: "storage" },
    ];
    const { bindGroup: outGroup1, bindGroupLayout: outLayout1 }
      = this.createBindGroup(outBinds1, "weightedSum");

    const outBinds2: BindingConfig[] = [
      { buffer: outContext2, bufferType: "read-only-storage" },
      { buffer: oWeights,   bufferType: "read-only-storage" },
      { buffer: resultBuffer, bufferType: "storage" },
    ];
    const { bindGroup: outGroup2, bindGroupLayout: outLayout2 }
       = this.createBindGroup(outBinds2, "outProj");

    const numTokBindGroupConfig: BindingConfig[] = [
        { buffer: numTokensBuffer, bufferType: "uniform" }
    ];
    const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
        = this.createBindGroup(numTokBindGroupConfig, "numTokBuf");

    const constants = {
      embedding_size:    embeddingSize,
      num_heads:         numHeads,
      num_kv_heads:      numKVHeads,
      head_size:         headSize,
      context_length:    contextLength,
      rope_theta:        ropeTheta,
    };

    if (!this.qkvPipeline) {
      this.qkvPipeline = this.createPipeline(
        [qkvLayout, numTokBindGroupLayout], qkvShader, constants, "qkv_proj"
      );
      this.ropePipeline = this.createPipeline(
        [ropeLayout, numTokBindGroupLayout], ropeShader, constants, "rope"
      );
      this.attnPipeline = this.createPipeline(
        [attnLayout, numTokBindGroupLayout], attnShader, constants, "attn"
      );
      this.outPipeline  = this.createPipeline(
        [outLayout1, outLayout2, numTokBindGroupLayout], outShader, constants, "out"
      );
    }

    const passes: PassConfig[] = [
      { pipeline: this.qkvPipeline!, bindGroups: [qkvGroup, numTokBindGroup],      numWorkgroups: [l => l, numHeads] },
      { pipeline: this.ropePipeline!, bindGroups: [ropeGroup, numTokBindGroup],    numWorkgroups: [l => l, numHeads] },
      { pipeline: this.attnPipeline!, bindGroups: [attnGroup, numTokBindGroup],    numWorkgroups: [numHeads, l => l] },
      { pipeline: this.outPipeline!,  bindGroups: [outGroup1, outGroup2, numTokBindGroup], numWorkgroups: [l => l, numHeads] },
    ];

    return { resultBuffer, passes };
  }
}
