import { Block } from "../Block";
import { BindingConfig, PassConfig } from "../../types";

// import raw WGSL shaders
import qkvShader from "./attention_qkv.wgsl?raw";
import ropeShader from "./attention_rope.wgsl?raw";
import attnSoftmaxShader from "./attention_qk_softmax.wgsl?raw"; // Renamed for clarity
import weightedSumShader from "./attention_weighted_sum.wgsl?raw"; // New shader
import projectionShader from "./attention_projection.wgsl?raw";   // New shader

export default class AttentionBlock extends Block {
  private qkvPipeline?: GPUComputePipeline;
  private ropePipeline?: GPUComputePipeline;
  private attnSoftmaxPipeline?: GPUComputePipeline;
  private weightedSumPipeline?: GPUComputePipeline; // New pipeline
  private projectionPipeline?: GPUComputePipeline;  // New pipeline

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
    oWeights: GPUBuffer, // This is Wo for the final projection
    embeddingSize: number,
    numHeads: number,
    numKVHeads: number,
    headSize: number,
    ropeTheta: number,
    contextLength: number // This is n_ctx, max sequence length
  ): { resultBuffer: GPUBuffer; passes: PassConfig[] } {
    // 1) Allocate intermediate buffers
    // qBuffer: [contextLength, numHeads, headSize] -> flat [contextLength * numHeads * headSize]
    const qBuffer = this.createBuffer([contextLength, numHeads * headSize], ["storage"], "qBuffer_attn");
    // kBuffer: [contextLength, numKVHeads, headSize] -> flat [contextLength * numKVHeads * headSize]
    const kBuffer = this.createBuffer([contextLength, numKVHeads * headSize], ["storage"], "kBuffer_attn");
    // vBuffer: [contextLength, numKVHeads, headSize] -> flat [contextLength * numKVHeads * head_size]
    const vBuffer = this.createBuffer([contextLength, numKVHeads * headSize], ["storage"], "vBuffer_attn");
    
    // attnBuffer (logits/scores): [numHeads, contextLength (q_seq), contextLength (kv_seq)]
    // Flattened size: numHeads * contextLength * contextLength
    const attnBuffer  = this.createBuffer([numHeads * contextLength * contextLength], ["storage"], "attnScoresBuffer");

    // intermediateContextBuffer: output of (Attn*V), input to (Wo * intermediateContext)
    // Shape: [contextLength (num_tokens_q), numHeads, headSize] -> flat [contextLength * numHeads * headSize]
    const intermediateContextBuffer = this.createBuffer([contextLength, numHeads * headSize], ["storage"], "intermediateCtxBuffer");
    
    // resultBuffer: final output of attention block
    // Shape: [contextLength (num_tokens_q), embeddingSize]
    const resultBuffer = this.createBuffer([contextLength, embeddingSize], ["storage", "copy_src"], "resultBuffer_attn");

    // --- Bind Group Layouts and Bind Groups ---

    // Uniform buffer for number of tokens
    const numTokBindGroupConfig: BindingConfig[] = [
        { buffer: numTokensBuffer, bufferType: "uniform" }
    ];
    const { bindGroup: numTokBindGroup, bindGroupLayout: numTokBindGroupLayout } 
        = this.createBindGroup(numTokBindGroupConfig, "numTokUniformBG_attn");

    // 2) QKV projection pass
    const qkvBindings: BindingConfig[] = [
      { buffer: inputBuffer,  bufferType: "read-only-storage" }, // x
      { buffer: qWeights,     bufferType: "read-only-storage" }, // Wq
      { buffer: kWeights,     bufferType: "read-only-storage" }, // Wk
      { buffer: vWeights,     bufferType: "read-only-storage" }, // Wv
      { buffer: qBuffer,      bufferType: "storage" },           // q_out
      { buffer: kBuffer,      bufferType: "storage" },           // k_out
      { buffer: vBuffer,      bufferType: "storage" },           // v_out
    ];
    const { bindGroup: qkvGroup, bindGroupLayout: qkvLayout } 
      = this.createBindGroup(qkvBindings, "qkvBG");

    // 3) RoPE pass
    const ropeBindings: BindingConfig[] = [
      { buffer: qBuffer,        bufferType: "storage" }, // q (in/out)
      { buffer: kBuffer,        bufferType: "storage" }, // k (in/out)
      { buffer: positionBuffer, bufferType: "read-only-storage" }, // pos
    ];
    const { bindGroup: ropeGroup, bindGroupLayout: ropeLayout }
      = this.createBindGroup(ropeBindings, "ropeBG");

    // 4) Attention Softmax pass (Q*K^T / sqrt(d_k))
    const attnSoftmaxBindings: BindingConfig[] = [
      { buffer: qBuffer,    bufferType: "read-only-storage" }, // q
      { buffer: kBuffer,    bufferType: "read-only-storage" }, // k
      { buffer: attnBuffer, bufferType: "storage" },           // logits (scores out)
    ];
    const { bindGroup: attnSoftmaxGroup, bindGroupLayout: attnSoftmaxLayout }
      = this.createBindGroup(attnSoftmaxBindings, "attnSoftmaxBG");

    // 5) Weighted Sum pass (AttnSoftmax * V)
    const weightedSumBindingsInput: BindingConfig[] = [
      { buffer: attnBuffer, bufferType: "read-only-storage" }, // attn_scores
      { buffer: vBuffer,    bufferType: "read-only-storage" }, // v_buffer
    ];
    const { bindGroup: weightedSumInputGroup, bindGroupLayout: weightedSumInputLayout }
      = this.createBindGroup(weightedSumBindingsInput, "weightedSumInputBG");
      
    const weightedSumBindingsOutput: BindingConfig[] = [
      { buffer: intermediateContextBuffer, bufferType: "storage" }, // ctx_out
    ];
    const { bindGroup: weightedSumOutputGroup, bindGroupLayout: weightedSumOutputLayout }
      = this.createBindGroup(weightedSumBindingsOutput, "weightedSumOutputBG");


    // 6) Output Projection pass (Wo * intermediateContext)
    const projectionBindingsInput: BindingConfig[] = [
      { buffer: intermediateContextBuffer, bufferType: "read-only-storage" }, // ctx_in
      { buffer: oWeights,                  bufferType: "read-only-storage" }, // Wo
    ];
    const { bindGroup: projectionInputGroup, bindGroupLayout: projectionInputLayout }
       = this.createBindGroup(projectionBindingsInput, "projectionInputBG");

    const projectionBindingsOutput: BindingConfig[] = [
         { buffer: resultBuffer, bufferType: "storage" }, // final_out
    ];
    const { bindGroup: projectionOutputGroup, bindGroupLayout: projectionOutputLayout }
       = this.createBindGroup(projectionBindingsOutput, "projectionOutputBG");


    // --- Compile Pipelines (once per instance of AttentionBlock) ---
    const constants = {
      embedding_size:    embeddingSize,
      num_heads:         numHeads,
      num_kv_heads:      numKVHeads,
      head_size:         headSize,
      context_length:    contextLength, // Max sequence length
      rope_theta:        ropeTheta,
    };

    if (!this.qkvPipeline) {
      this.qkvPipeline = this.createPipeline(
        [qkvLayout, numTokBindGroupLayout], qkvShader, constants, "qkv_proj_pipeline"
      );
    }
    if (!this.ropePipeline) {
      this.ropePipeline = this.createPipeline(
        [ropeLayout, numTokBindGroupLayout], ropeShader, constants, "rope_pipeline"
      );
    }
    if (!this.attnSoftmaxPipeline) {
      this.attnSoftmaxPipeline = this.createPipeline(
        [attnSoftmaxLayout, numTokBindGroupLayout], attnSoftmaxShader, constants, "attn_softmax_pipeline"
      );
    }
    if (!this.weightedSumPipeline) {
      this.weightedSumPipeline = this.createPipeline(
        [weightedSumInputLayout, weightedSumOutputLayout, numTokBindGroupLayout], 
        weightedSumShader, 
        constants, 
        "weighted_sum_pipeline"
      );
    }
    if (!this.projectionPipeline) {
      this.projectionPipeline = this.createPipeline(
        [projectionInputLayout, projectionOutputLayout, numTokBindGroupLayout], 
        projectionShader, 
        constants, 
        "projection_pipeline"
      );
    }

    // --- Build PassConfig list ---
    // Workgroup counts need to be functions of actual num_tokens if dynamic, or constants if fixed.
    // (l => l) means num_tokens, which is passed to the function when dispatching.
    // Workgroup size for qkv: (8,8). Dispatch: (ceil(num_tokens/8), ceil(num_heads/8))
    // Workgroup size for rope: (8,8). Dispatch: (ceil(num_tokens/8), ceil(num_heads/8))
    // Workgroup size for attn_softmax: (8,8). Dispatch: (ceil(num_heads/8), ceil(num_tokens/8))
    // Workgroup size for weighted_sum: (8,8). Dispatch: (ceil(num_tokens/8), ceil(num_heads/8))
    // Workgroup size for projection: (16,16). Dispatch: (ceil(num_tokens/16), ceil(embeddingSize/16))

    const passes: PassConfig[] = [
      { 
        pipeline: this.qkvPipeline!, 
        bindGroups: [qkvGroup, numTokBindGroup],      
        numWorkgroups: [
            (numTokens: number) => Math.ceil(numTokens / 8), 
            Math.ceil(numHeads / 8) // numHeads for q, numKVHeads for k/v implicitly handled by shader logic
        ] 
      },
      { 
        pipeline: this.ropePipeline!, 
        bindGroups: [ropeGroup, numTokBindGroup],    
        numWorkgroups: [
            (numTokens: number) => Math.ceil(numTokens / 8), 
            Math.ceil(numHeads / 8)
        ] 
      },
      { 
        pipeline: this.attnSoftmaxPipeline!, 
        bindGroups: [attnSoftmaxGroup, numTokBindGroup],    
        numWorkgroups: [
            Math.ceil(numHeads / 8),
            (numTokens: number) => Math.ceil(numTokens / 8) 
        ] 
      },
      { 
        pipeline: this.weightedSumPipeline!,  
        bindGroups: [weightedSumInputGroup, weightedSumOutputGroup, numTokBindGroup], 
        numWorkgroups: [
            (numTokens: number) => Math.ceil(numTokens / 8), 
            Math.ceil(numHeads / 8)
        ]
      },
      {
        pipeline: this.projectionPipeline!,
        bindGroups: [projectionInputGroup, projectionOutputGroup, numTokBindGroup],
        numWorkgroups: [
            (numTokens: number) => Math.ceil(numTokens / 16),
            Math.ceil(embeddingSize / 16)
        ]
      }
    ];

    return { resultBuffer, passes };
  }
}
