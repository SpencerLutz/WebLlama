import { Block } from '../Block';
import { BindingConfig, PassConfig } from '../../types';
import matmulShader from './matmul_2d.wgsl?raw';
import siluMulShader from './silu_mul.wgsl?raw';

export default class SwiGLUBlock extends Block {
  private matmul1?: GPUComputePipeline;
  private matmul2?: GPUComputePipeline;
  private siluMul?: GPUComputePipeline;
  private matmul3?: GPUComputePipeline;

  constructor(device: GPUDevice) {
    super(device);
  }

  newInstance(
    inputBuffer: GPUBuffer,
    numTokensBuffer: GPUBuffer,
    w1WeightsBuffer: GPUBuffer,
    w3WeightsBuffer: GPUBuffer,
    w2WeightsBuffer: GPUBuffer,
    embeddingSize: number,
    intermediateSize: number,
    contextLength: number
  ): { resultBuffer: GPUBuffer; passes: PassConfig[] } {
    const M = contextLength;
    const K1 = embeddingSize, N1 = intermediateSize;
    const L = M * N1;
    const K2 = intermediateSize, N2 = embeddingSize;

    // allocate intermediate + output buffers
    const gateBuffer = this.createBuffer([M, N1], ['storage', 'copy_src'], "gateBuffer");
    const upBuffer   = this.createBuffer([M, N1], ['storage', 'copy_src'], "upBuffer");
    // reuse gateBuffer for activated
    const resultBuffer = this.createBuffer([M, N2], ['storage', 'copy_src'], "resultBuffer_mlp");

    const numTokBindGroupConfig: BindingConfig[] = [
        { buffer: numTokensBuffer, bufferType: "uniform" }
    ];
    const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
        = this.createBindGroup(numTokBindGroupConfig, "numTokBuf");

    // 1) input × W1 → gateBuffer
    const bg1 = this.createBindGroup([
      { buffer: inputBuffer, bufferType: 'read-only-storage' },
      { buffer: w1WeightsBuffer, bufferType: 'read-only-storage' },
      { buffer: gateBuffer, bufferType: 'storage' }
    ], "weightMultiplyGroup1");
    if (!this.matmul1) {
      this.matmul1 = this.createPipeline(
        [bg1.bindGroupLayout, numTokBindGroupLayout],
        matmulShader,
        { M, K: K1, N: N1 },
        "matmul_swiglu"
      );
    }

    // 2) input × W3 → upBuffer
    const bg2 = this.createBindGroup([
      { buffer: inputBuffer, bufferType: 'read-only-storage' },
      { buffer: w3WeightsBuffer, bufferType: 'read-only-storage' },
      { buffer: upBuffer, bufferType: 'storage' }
    ], "weightMultiplyGroup2");
    if (!this.matmul2) {
      this.matmul2 = this.createPipeline(
        [bg2.bindGroupLayout, numTokBindGroupLayout],
        matmulShader,
        { M, K: K1, N: N1 },
        "matmul_swiglu"
      );
    }

    // 3) silu(gate) * up → gateBuffer
    const bg3 = this.createBindGroup([
      { buffer: gateBuffer, bufferType: 'storage' },
      { buffer: upBuffer,   bufferType: 'read-only-storage' }
    ], "SiluGroup");
    if (!this.siluMul) {
      this.siluMul = this.createPipeline(
        [bg3.bindGroupLayout, numTokBindGroupLayout],
        siluMulShader,
        { N1 },
        "silu_mul"
      );
    }

    // 4) activated (in gateBuffer) × W2 → resultBuffer
    const bg4 = this.createBindGroup([
      { buffer: gateBuffer, bufferType: 'read-only-storage' },
      { buffer: w2WeightsBuffer, bufferType: 'read-only-storage' },
      { buffer: resultBuffer, bufferType: 'storage' }
    ], "outMLPGroup");
    if (!this.matmul3) {
      this.matmul3 = this.createPipeline(
        [bg4.bindGroupLayout, numTokBindGroupLayout],
        matmulShader,
        { M, K: K2, N: N2 },
        "matmul_swiglu"
      );
    }

    // build passes
    const passes: PassConfig[] = [
      {
        pipeline: this.matmul1!,
        bindGroups: [bg1.bindGroup, numTokBindGroup],
        numWorkgroups: [
          (numTokens) => Math.ceil(numTokens / 16),
          Math.ceil(N1 / 16)
        ]
      },
      {
        pipeline: this.matmul2!,
        bindGroups: [bg2.bindGroup, numTokBindGroup],
        numWorkgroups: [
          (numTokens) => Math.ceil(numTokens / 16),
          Math.ceil(N1 / 16)
        ]
      },
      {
        pipeline: this.siluMul!,
        bindGroups: [bg3.bindGroup, numTokBindGroup],
        numWorkgroups: [(numTokens) => Math.ceil(numTokens * N1 / 64)]
      },
      {
        pipeline: this.matmul3!,
        bindGroups: [bg4.bindGroup, numTokBindGroup],
        numWorkgroups: [
          (numTokens) => Math.ceil(numTokens / 16),
          Math.ceil(N2 / 16)
        ]
      }
    ];

    return { resultBuffer, passes };
  }
}
