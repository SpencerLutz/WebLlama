import { MatMulDemoBlock } from "./blocks/matMulDemo/MatMulDemoBlock";
import { PassConfig } from "./types";
import { Tokenizer } from "./tokenizers/Tokenizer";
import * as global from "./global";

interface LoadedModelBuffers {
  // Token Embeddings (potentially chunked for DeEmbed)
  embeddingsBuffer: GPUBuffer;     // Single buffer for lookup [vocab_size, n_embd]
  deEmbeddingsBuffers: GPUBuffer[]; // Transposed [n_embd, vocab_chunk_size] possibly chunked for final matmul

  // Layer Buffers
  layer_buffers: LayerBuffers[];

  // Final Norm (RMSNorm)
  normGammaBuffer: GPUBuffer; // RMSNorm weight [n_embd]

  // Output projection (tied to embeddings in Llama3) - uses deEmbeddingsBuffers
}

// Buffers specific to a single Llama 3 Transformer Layer
interface LayerBuffers {
  // Norms (RMSNorm weights)
  normAttentionGammaBuffer: GPUBuffer; // Input Norm weight [n_embd]
  normLinearGammaBuffer: GPUBuffer;    // FFN Norm weight [n_embd]

  // Attention Weights (GQA - No Biases)
  qWeightsBuffer: GPUBuffer; // [n_embd, n_embd]
  kWeightsBuffer: GPUBuffer; // [n_kv_head * head_size, n_embd]
  vWeightsBuffer: GPUBuffer; // [n_kv_head * head_size, n_embd]
  oWeightsBuffer: GPUBuffer; // [n_embd, n_embd]

  // MLP Weights (SwiGLU - No Biases)
  w1WeightsBuffer: GPUBuffer; // Gate proj [intermediate_size, n_embd]
  w3WeightsBuffer: GPUBuffer; // Up proj   [intermediate_size, n_embd]
  w2WeightsBuffer: GPUBuffer; // Down proj [n_embd, intermediate_size]
}

interface ModelParams {
  n_embd: number;         // Mapped from hidden_size
  n_layer: number;        // Mapped from num_hidden_layers
  n_head: number;         // Mapped from num_attention_heads
  n_kv_head: number;      // Mapped from num_key_value_heads
  head_size: number;      // Mapped from head_dim or calculated (n_embd / n_head)
  vocab_size: number;     // Mapped from vocab_size
  intermediate_size: number; // Mapped from intermediate_size (for SwiGLU MLP)
  n_ctx: number;          // Context size (max sequence length)
  norm_eps: number;       // Mapped from rms_norm_eps
  rope_theta: number;     // Mapped from rope_theta

  // Derived parameters for buffer splitting / WebGPU implementation details
  vocab_chunk_size: number; // For potential chunking of DeEmbed
  vocab_chunk_instances: number;
}

export class Model {
  private initialized: boolean = false;

  private device?: GPUDevice;
  private model?: LoadedModelBuffers;
  private tokenizer?: Tokenizer;
  private params?: ModelParams;
  private minBufferOffset: number = 256; // Default, updated on init

  private defaultPrompt: string = "";
  private defaultTopK: number = 5;
  private defaultTemperature: number = 0.7;
  private defaultTokens: number = 100;

  private passes?: Array<PassConfig>;
  private unloadDeletionStack: GPUBuffer[] = [];

  async init() {
    if (this.initialized) {
        console.warn("Model already initialized. Call unloadBuffers() first if re-initialization is needed.");
        return;
    }
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("Failed to get GPU adapter.");
    }
    // Request device with required limits if known (optional)
    this.device = await adapter.requestDevice();
    if (!this.device) {
        throw new Error("Failed to get GPU device.");
    }

    this.minBufferOffset = this.device.limits.minStorageBufferOffsetAlignment || 256;
    console.log("WebGPU Device Acquired:", this.device.label);
    console.log("Device Limits Relevant to Buffers:");
    console.log(`  maxStorageBufferBindingSize: ${this.device.limits.maxStorageBufferBindingSize}`);
    console.log(`  maxBufferSize: ${this.device.limits.maxBufferSize}`);
    console.log(`  minStorageBufferOffsetAlignment: ${this.minBufferOffset}`);

    //initializeOperations(this.device);

    const weightsFolder = `weights/`;

    // Load parameters first to know model dimensions
    const paramsFromJson = await this.loadLlamaParameters(weightsFolder);
    if (!paramsFromJson) {
        throw new Error("Failed to load model parameters (config.json).");
    }
    this.params = this.deriveInternalParams(paramsFromJson, 2048);

    
    // Load model weights into GPU buffers
    [this.model] = await this.loadModelWeights(weightsFolder, this.params);

    // Load tokenizer
    this.tokenizer = new Tokenizer();

    // Set defaults
    this.defaultPrompt = `What is the answer to life, the universe, and everything?\n`; // Or a better Llama prompt start
    this.defaultTopK = 5;
    this.defaultTemperature = 0.6; // Common for Llama
    this.defaultTokens = 128;

    this.initialized = true;
    console.log("Llama 3 Model initialized successfully.");
  }

  async *generate(
    prompt: string,
    max_new_tokens: number = this.defaultTokens,
    top_k: number = this.defaultTopK,
    temperature: number = this.defaultTemperature
  ): AsyncGenerator<string, void, undefined> {
    if (!this.initialized || !this.tokenizer || !this.params || !this.model || !this.device) {
      console.error("Model not properly initialized. Call initialize() first.");
      return;
    }
    if (prompt === "") {
        console.warn("Prompt is empty.");
        return;
    }

    let history = this.tokenizer.encode(prompt);
    // Llama often uses BOS token 128000 at the start, check if tokenizer adds it automatically.
    // if (history[0] !== 128000) { history.unshift(128000); } // Example: Prepend BOS if needed
    console.log(`Prompt (${history.length} tokens):\n${prompt}`);

    const warmupRuns = 1;
    let totalTime = 0;
    let generatedTokens = 0;
    const eos_token_id = 128001; // Standard Llama3 EOS token

    // --- KV Cache Handling (Placeholder) ---
    // You need to manage KV cache buffers here. Create them before the loop.
    // Their size depends on n_layer, n_ctx, n_kv_head, head_size.
    // Example structure (needs actual buffer creation and management):
    for (let i = 0; i < max_new_tokens; i++) {
        const currentLength = history.length;
        if (currentLength >= this.params.n_ctx) {
            console.warn(`Context length limit (${this.params.n_ctx}) reached. Truncating history.`);
            // Implement sliding window or truncation if needed
            history = history.slice(-this.params.n_ctx + 1); // Keep space for next token
        }

        // Prepare input for this step
        const isPrefill = i === 0; // Or determined by kvCache state
        const inputTokens = isPrefill ? history : [history[history.length - 1]]; // Process full prompt or just the last token
        const positions = isPrefill
            ? Array.from({ length: inputTokens.length }, (_, k) => k) // 0, 1, 2...
            : [currentLength - 1]; // Position of the single new token

        /*    
        // --- Run Inference ---
        const startTime = performance.now();
        const logits = await this.run(inputTokens, positions); // Pass current KV cache
        const endTime = performance.now();

        if (!logits) {
            console.error("Logits calculation failed for step", i);
            return;
        }

        const lapsedTime = endTime - startTime;
        if (i >= warmupRuns) {
            totalTime += lapsedTime;
            generatedTokens++;
        }
        const tps = 1000 / lapsedTime;
        console.log(`Token ${i + 1}/${max_new_tokens} (${isPrefill ? 'Prefill' : 'Decode'}) | ${tps.toFixed(1)} tok/s | Kernel time: ${lapsedTime.toFixed(2)} ms`);


        // --- Sample Next Token ---
        // Logits buffer contains logits for *all* input tokens. We only need the last one.
        const lastTokenLogitsOffset = (inputTokens.length - 1) * this.params.vocab_size;
        const lastTokenLogits = logits.slice(lastTokenLogitsOffset, lastTokenLogitsOffset + this.params.vocab_size);

        // TODO: Implement top-p sampling? Llama often uses it.
        const { topKIndices, topKProbs } = selectTopK(lastTokenLogits, top_k);
        const topKProbs_float = new Float32Array(topKProbs);
        const probs = cpuSoftmax(topKProbs_float, temperature);
        const idx_next = topKIndices[sampleFromDistribution(probs)];

        // Stop if EOS token is generated
        if (idx_next === eos_token_id) {
            console.log("EOS token detected.");
            break;
        }

        // Update history and yield the new token
        history = history.concat(idx_next);
        const decodedToken = this.tokenizer.decode([idx_next]);
        yield decodedToken;

        // --- Update KV Cache (Placeholder) ---
        // The 'run' method should have updated the kvCache internally based on the 'positions' argument.
        // No explicit update needed here if 'run' handles it.

        */
    }

    if (generatedTokens > 0) {
        console.log(`Average Decode kernel execution time: ${(totalTime / generatedTokens).toFixed(2)} ms (${(1000 * generatedTokens / totalTime).toFixed(1)} tok/s)`);
    } else if (max_new_tokens > warmupRuns) {
        console.log("Not enough tokens generated after warmup to calculate average time.");
    }
  }


  /**
   * Executes the Llama 3 inference pipeline on the GPU.
   * @param token_ids Input token IDs for this step.
   * @param positions Positional indices corresponding to token_ids, needed for RoPE.
   * @param kvCache Key-Value cache buffers (placeholder structure).
   * @returns A Float32Array containing the logits for the input sequence, or null on failure.
   */
  async run(
        token_ids: number[],
        positions: number[], // For RoPE
    ) {
    if (!this.device || !this.model || !this.params) {
        console.error("Cannot run inference: model not fully loaded or initialized.");
        return null;
    }

    /*

    const { layer_buffers, normGammaBuffer, embeddingsBuffer, deEmbeddingsBuffers } = this.model;
    const { n_embd, n_head, n_kv_head, head_size, n_layer, vocab_size, intermediate_size, norm_eps, rope_theta, n_ctx } = this.params;
    const seq_length = token_ids.length; // Length of the current input sequence step

    console.debug(`Running inference step. Input seq_length: ${seq_length}`);

    // --- Create GPU Buffers for this step's inputs ---
    // TODO: Manage these buffers efficiently (reuse, pool?)
    const tokenIdBuffer = this.initTensor(new Int32Array(token_ids), [seq_length], ["storage"], true, 'int32'); // Use int32 for token IDs
    const positionIdBuffer = this.initTensor(new Int32Array(positions), [seq_length], ["storage"], true, 'int32'); // Use int32 for positions

    // --- Setup Compute Passes ---
    this.computePasses = [];
    let residualBuffer: GPUBuffer; // Holds the input to the current layer block (output of previous + residual)
    let currentHiddenState: GPUBuffer; // Holds the output of the current block before residual addition

    // 1. Embedding Lookup
    console.debug(" Creating Embedding pass...");
    const embedResult = EmbedBlock.newInstance(
        tokenIdBuffer,          // Input token IDs for this step
        embeddingsBuffer,       // Full embedding weight matrix
        seq_length,
        n_embd
        // Output buffer containing embedding vectors [seq_length, n_embd]
    );
    currentHiddenState = embedResult.resultBuffer;
    residualBuffer = currentHiddenState; // Input to the first layer is the embedding output
    this.computePasses.push(...embedResult.passes);

    // 2. Transformer Layers
    for (let i = 0; i < n_layer; i++) {
        const layerName = `Layer ${i}`;
        const buffers = layer_buffers[i];

        // --- 2a. Pre-Attention RMSNorm ---
        console.debug(` ${layerName}: RMSNorm (Attention Input)`);
        const normAttnResult = RMSNormBlock.newInstance(
            residualBuffer,                 // Input: Output of previous layer (or embedding)
            buffers.normAttentionGammaBuffer, // RMSNorm weights
            seq_length,
            n_embd,
            norm_eps
            // Output buffer: Normalized hidden state [seq_length, n_embd]
        );
        currentHiddenState = normAttnResult.resultBuffer;
        this.computePasses.push(...normAttnResult.passes);

        // --- 2b. Attention (GQA + RoPE + KV Cache) ---
        console.debug(` ${layerName}: Attention (GQA, RoPE)`);
        const attentionResult = LlamaAttentionBlock.newInstance(
            currentHiddenState,         // Input: Normalized hidden state
            positionIdBuffer,           // Positions for RoPE calculation
            buffers.qWeightsBuffer,     // Weights Q
            buffers.kWeightsBuffer,     // Weights K
            buffers.vWeightsBuffer,     // Weights V
            buffers.oWeightsBuffer,     // Weights O
            kvCache.keys[i],            // KV Cache Key buffer for this layer
            kvCache.values[i],          // KV Cache Value buffer for this layer
            seq_length,                 // Current input sequence length
            n_ctx,                      // Max context length (for cache indexing)
            n_embd,
            n_head,
            n_kv_head,
            head_size,
            rope_theta
            // Output buffer: Attention output [seq_length, n_embd]
            // This block *must* update the provided kvCache buffers internally.
        );
        currentHiddenState = attentionResult.resultBuffer;
        this.computePasses.push(...attentionResult.passes);

        // --- 2c. Residual Connection 1 ---
        console.debug(` ${layerName}: Residual Add (Attention)`);
        const residual1Result = ResidualBlock.newInstance(
            currentHiddenState, // Input from Attention block
            residualBuffer,     // Input from before the layer norm (output of previous layer)
            seq_length,
            n_embd
            // Output buffer: Added result [seq_length, n_embd]
        );
        // The output of the residual add becomes the input for the next block's norm
        residualBuffer = residual1Result.resultBuffer;
        this.computePasses.push(...residual1Result.passes);

        // --- 2d. Post-Attention RMSNorm (Pre-FFN/MLP) ---
        console.debug(` ${layerName}: RMSNorm (FFN Input)`);
        const normFfnResult = RMSNormBlock.newInstance(
            residualBuffer,             // Input: Output of Residual Add 1
            buffers.normLinearGammaBuffer, // RMSNorm weights for FFN
            seq_length,
            n_embd,
            norm_eps
            // Output buffer: Normalized hidden state [seq_length, n_embd]
        );
        currentHiddenState = normFfnResult.resultBuffer;
        this.computePasses.push(...normFfnResult.passes);

        // --- 2e. MLP Block (SwiGLU) ---
        console.debug(` ${layerName}: MLP (SwiGLU)`);
        const swigluResult = SwiGLUBlock.newInstance(
            currentHiddenState,         // Input: Normalized hidden state from FFN norm
            buffers.w1WeightsBuffer,    // Gate projection weights
            buffers.w3WeightsBuffer,    // Up projection weights
            buffers.w2WeightsBuffer,    // Down projection weights
            seq_length,
            n_embd,
            intermediate_size
            // Output buffer: MLP output [seq_length, n_embd]
        );
        currentHiddenState = swigluResult.resultBuffer;
        this.computePasses.push(...swigluResult.passes);

        // --- 2f. Residual Connection 2 ---
        console.debug(` ${layerName}: Residual Add (MLP)`);
        const residual2Result = ResidualBlock.newInstance(
            currentHiddenState, // Input from SwiGLU block
            residualBuffer,     // Input from Residual Add 1 (before FFN norm)
            seq_length,
            n_embd
            // Output buffer: Added result [seq_length, n_embd]
        );
        // This is the final output of the layer, becomes the input residual for the next layer
        residualBuffer = residual2Result.resultBuffer;
        this.computePasses.push(...residual2Result.passes);

        console.debug(` ${layerName}: Completed`);
    } // End Layer Loop

    // 3. Final RMSNorm (Before Output Projection)
    console.debug(" Creating Final RMSNorm pass...");
    const finalNormResult = RMSNormBlock.newInstance(
        residualBuffer,     // Input from last layer's output
        normGammaBuffer,    // Final norm weights
        seq_length,
        n_embd,
        norm_eps
        // Output buffer: Final normalized hidden state [seq_length, n_embd]
    );
    currentHiddenState = finalNormResult.resultBuffer;
    this.computePasses.push(...finalNormResult.passes);

    // 4. DeEmbedding / Output Projection
    // Uses the (transposed) embedding weights. May be chunked.
    console.debug(" Creating DeEmbedding (Output Projection) pass...");
    const deembedResult = DeEmbedBlock.newInstance(
        currentHiddenState,     // Input from final norm
        deEmbeddingsBuffers,    // Transposed embedding weights (potentially chunked)
        seq_length,
        n_embd,
        vocab_size,
        this.params.vocab_chunk_size, // Pass chunk info if DeEmbedBlock handles chunking
        this.params.vocab_chunk_instances
        // Output buffer: Logits [seq_length, vocab_size]
    );
    const logitsBuffer = deembedResult.resultBuffer; // Final output buffer
    this.computePasses.push(...deembedResult.passes);

    // --- Execute Compute Passes ---
    console.debug(" Encoding and submitting GPU commands...");
    const commandEncoder = this.device.createCommandEncoder();
    for (const pass of this.passes?) {
      if (pass.flag === "compute") {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pass.pipeline);
        for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
        passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y, pass.workgroups.z);
        passEncoder.end();
      } else if (pass.flag === "copy") {
        // Copy passes might be needed for KV cache management or debugging
        commandEncoder.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
      }
    }
    this.device.queue.submit([commandEncoder.finish()]);
    console.debug(" Commands submitted.");

    // --- Read Results Back ---
    console.debug(" Mapping result buffer (Logits)...");
    // Create a temporary buffer for reading back the results.
    // Reuse readback buffers if possible for performance.
    const resultBufferSize = this.bufferSize(seq_length, vocab_size); // Calculate expected size
    const readbackBuffer = this.device.createBuffer({
        size: resultBufferSize,
        usage: ["map_read", "copy_to"].map(getBufferUsage).reduce((a, b) => a | b)
    });

    // Copy the final logits from the compute buffer to the readback buffer
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(logitsBuffer, 0, readbackBuffer, 0, resultBufferSize);
    this.device.queue.submit([copyEncoder.finish()]);

    // Wait for the GPU to finish and map the buffer
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const output = readbackBuffer.getMappedRange();
    // Slice ensures we only take the valid part of the buffer if it was padded
    const outputArray = new Float32Array(output.slice(0, seq_length * vocab_size * Float32Array.BYTES_PER_ELEMENT));
    // IMPORTANT: Create a copy before unmapping, as the underlying ArrayBuffer is detached.
    const outputCopy = outputArray.slice();
    readbackBuffer.unmap();
    readbackBuffer.destroy(); // Clean up the temporary readback buffer

    // Destroy temporary input buffers for this step
    tokenIdBuffer.destroy();
    positionIdBuffer.destroy();

    console.debug(" Results read back.");

    return outputCopy;
    */
  }

  // --- Model Loading Functions ---

  // Loads the model weights into GPU buffers
  async loadModelWeights(weightsFolder: string, params: ModelParams): Promise<[LoadedModelBuffers, ModelParams]> {
    if (!this.device) throw new Error("Device not initialized for model loading.");
    if (this.initialized) {
        console.warn("Model already loaded. Re-loading.");
        this.unloadBuffers();
    }

    console.log("Loading Llama3 model weights from:", weightsFolder);

    // Load Token Embeddings (and prepare for DeEmbedding/Output Projection)
    console.log(" Loading token embeddings (tok_embeddings.weight.bin)...");
    const { embeddingsBuffer, deEmbeddingsBuffers } = await this.loadEmbeddings(params, weightsFolder);

    // Load Transformer Layers
    console.log(" Loading transformer layers...");
    const layer_buffers = await this.loadLayers(params, weightsFolder);

    // Load Final Layer Norm (RMSNorm)
    console.log(" Loading final layer norm (norm.weight.bin)...");
    const normGammaBuffer = await this.fetchAndInitTensor(
        `${weightsFolder}norm.weight.bin`,
        [params.n_embd],
        ["storage"] // Read-only weight for final norm
    );

    const modelBuffers = {
        embeddingsBuffer,
        deEmbeddingsBuffers,
        layer_buffers,
        normGammaBuffer
    };

    console.log("Finished loading model weights.");
    return [modelBuffers, params];
  }

  // Loads Llama3 specific parameters from config.json
  async loadLlamaParameters(weightsFolder: string): Promise<any | null> {
      try {
          console.log("Loading Llama parameters from config.json...");
          const response = await fetch(`${weightsFolder}/config.json`);
          if (!response.ok) {
              throw new Error(`HTTP error loading config.json: ${response.status}`);
          }
          const params = await response.json();
          console.log(" Loaded Raw Llama Params:", params);
          return params;
      } catch (error) {
          console.error("Error loading Llama parameters:", error);
          return null;
      }
  }

  // Maps loaded Llama3 parameters and calculates derived values
  deriveInternalParams(llamaParams: any, contextSize: number): ModelParams {
      if (!this.device) throw new Error("Device not initialized for parameter derivation.");

      const n_embd = llamaParams.hidden_size;
      const n_layer = llamaParams.num_hidden_layers;
      const n_head = llamaParams.num_attention_heads;
      const n_kv_head = llamaParams.num_key_value_heads || n_head;
      const vocab_size = llamaParams.vocab_size;
      const intermediate_size = llamaParams.intermediate_size;
      const norm_eps = llamaParams.rms_norm_eps;
      const head_size = llamaParams.head_dim || n_embd / n_head;
      const rope_theta = llamaParams.rope_theta || 10000.0; // Default if not specified

      if (n_embd % n_head !== 0) {
          console.warn(`Warning: hidden_size (${n_embd}) is not perfectly divisible by num_attention_heads (${n_head}).`);
      }
      if(n_head % n_kv_head !== 0) {
          console.warn(`Warning: num_attention_heads (${n_head}) is not divisible by num_key_value_heads (${n_kv_head}). This is required for GQA grouping.`);
      }

      const n_ctx = Math.min(contextSize, llamaParams.max_position_embeddings || contextSize);
      console.log(` Using context size (n_ctx): ${n_ctx}`);

      // Calculate vocab chunk size if DeEmbedding needs chunking
      const { vocab_chunk_size, vocab_chunk_instances } = this.calculateVocabChunking(
            vocab_size, n_embd, this.device.limits.maxStorageBufferBindingSize
        );


      // --- Final Parameter Object ---
      const params: ModelParams = {
          n_embd,
          n_layer,
          n_head,
          n_kv_head,
          head_size,
          vocab_size,
          intermediate_size,
          n_ctx,
          norm_eps,
          rope_theta,
          vocab_chunk_size,
          vocab_chunk_instances,
      };

      // --- Parameter Size Checks (Optional but recommended) ---
       this.checkBufferLimits(params);

      console.log("Derived Internal Params:", params);
      return params;
  }

  // Calculates if vocab needs chunking for DeEmbedding and the chunk parameters
  calculateVocabChunking(vocab_size: number, n_embd: number, maxBindingSize: number): { vocab_chunk_size: number, vocab_chunk_instances: number } {
        // DeEmbedding matmul uses transposed weights [n_embd, vocab_size]
        const deEmbedSizeBytes = this.bufferSize(n_embd, vocab_size);
        let vocab_chunk_size = vocab_size;
        let vocab_chunk_instances = 1;

        if (deEmbedSizeBytes > maxBindingSize) {
            console.warn(`DeEmbedding buffer size (${deEmbedSizeBytes} B) exceeds maxStorageBufferBindingSize (${maxBindingSize} B). Splitting required.`);
            // Calculate minimum splits needed based on bytes per vocab item in the transposed matrix
            const bytesPerVocabItem = n_embd * Float32Array.BYTES_PER_ELEMENT;
            const maxVocabItemsPerChunk = Math.floor(maxBindingSize / bytesPerVocabItem);

            if (maxVocabItemsPerChunk === 0) {
                throw new Error(`Cannot fit even one vocab item (${bytesPerVocabItem} B) into maxStorageBufferBindingSize (${maxBindingSize} B)`);
            }

            vocab_chunk_instances = Math.ceil(vocab_size / maxVocabItemsPerChunk);
            vocab_chunk_size = Math.ceil(vocab_size / vocab_chunk_instances);
            // Ensure alignment if needed (e.g., align to 4 or 8 for vectorization)
            const alignment = 8;
            vocab_chunk_size = Math.ceil(vocab_chunk_size / alignment) * alignment;
            // Recalculate instances based on aligned chunk size
            vocab_chunk_instances = Math.ceil(vocab_size / vocab_chunk_size);

            console.log(` Calculated DeEmbedding Chunks: instances=${vocab_chunk_instances}, chunk_size (tokens)=${vocab_chunk_size}`);

             // Sanity check the recalculated chunk size
            const chunkSizeBytes = this.bufferSize(n_embd, vocab_chunk_size);
            if (chunkSizeBytes > maxBindingSize) {
                 console.error(`Error: Calculated chunk size ${vocab_chunk_size} still results in buffer size ${chunkSizeBytes} exceeding limit ${maxBindingSize}.`);
                 // This indicates a potential logic error in calculation.
                 throw new Error("Failed to calculate valid vocab chunk size.");
            }
        } else {
            console.log(`DeEmbedding buffer size (${deEmbedSizeBytes} B) fits within limits. No chunking applied.`);
            // Align vocab_size if needed, even if not chunking, might help shader performance
            const alignment = 8;
            vocab_chunk_size = Math.ceil(vocab_size / alignment) * alignment;
            if(vocab_chunk_size !== vocab_size) {
                 console.log(` Aligning vocab_size for DeEmbedding buffer from ${vocab_size} to ${vocab_chunk_size}.`);
            }
        }
        return { vocab_chunk_size, vocab_chunk_instances };
  }

   // Optional: Check common large buffer sizes against limits
   checkBufferLimits(params: ModelParams): void {
        if (!this.device) return;
        const maxStorageElements = this.device.limits.maxStorageBufferBindingSize / Float32Array.BYTES_PER_ELEMENT;
        const maxTotalElements = this.device.limits.maxBufferSize / Float32Array.BYTES_PER_ELEMENT;

        const check = (label: string, size: number, limit: number = maxStorageElements) => {
            if (size > limit) {
                console.warn(`Potential Buffer Limit Exceeded: ${label} (${size} elements) > Limit (${limit} elements)`);
            }
        };

        // Check individual weights
        check("Attention Q/O Weight", params.n_embd * params.n_embd);
        check("Attention K/V Weight", params.n_kv_head * params.head_size * params.n_embd);
        check("MLP W1/W3 Weight", params.intermediate_size * params.n_embd);
        check("MLP W2 Weight", params.n_embd * params.intermediate_size);
        check("Token Embedding", params.vocab_size * params.n_embd, maxTotalElements); // Check against maxBufferSize

        // Check intermediate activation tensors (can be large)
        check("Activations (Hidden State)", params.n_ctx * params.n_embd);
        check("Activations (MLP Intermediate)", params.n_ctx * params.intermediate_size);
        // Attention scores (n_ctx * n_ctx * n_head) - can be very large, often computed tile-by-tile
        // check("Attention Scores (Full)", params.n_ctx * params.n_ctx * params.n_head);
   }


   async loadEmbeddings(params: ModelParams, weightsFolder: string) {
    console.log("Loading token embeddings...");
    const embeddingWeights = await global.fetchBin(`${weightsFolder}/tok_embeddings.weight.bin`);

    // Chunks are stored in row-major order and are of dimensions n_embd x vocab_chunk_size.
    // Embedding weights are imported in column-major order and are of dimensions vocab_size x n_embd.
    // We pre-transpose the chunk for the deEmbedding process for the matmul. Could do this on GPU later.
    const embeddingsBuffers: GPUBuffer[] = [];
    const deEmbeddingsBuffers: GPUBuffer[] = [];
    for (let i = 0; i < params.vocab_chunk_instances; i++) {
      console.log(`Loading deEmbedding chunk ${i + 1}/${params.vocab_chunk_instances}...`);
      const offset = i * params.vocab_chunk_size;
      let size = params.vocab_chunk_size;

      const paddedArray = new Float32Array(params.vocab_chunk_size * params.n_embd);
      if (i === params.vocab_chunk_instances - 1) {
        size = params.vocab_size - offset;
        // First set the actual data
        paddedArray.set(embeddingWeights.subarray(offset * params.n_embd, offset * params.n_embd + size * params.n_embd));
        // Then set the zeros for padding at the correct offset
        paddedArray.set(
            global.zeros((params.vocab_chunk_size - size) * params.n_embd), 
            size * params.n_embd  // offset where to start writing zeros
        );
      } else {
          paddedArray.set(embeddingWeights.subarray(offset * params.n_embd, offset * params.n_embd + size * params.n_embd));
      }

      embeddingsBuffers.push(this.initTensor(paddedArray, [params.vocab_chunk_size, params.n_embd], ["copy_from"]));

      const chunk = global.transpose(paddedArray, params.vocab_chunk_size, params.n_embd); // Use GPU perhaps?
      deEmbeddingsBuffers.push(this.initTensor(chunk, [params.n_embd, params.vocab_chunk_size], ["storage"]));
    }

    return { embeddingsBuffers, deEmbeddingsBuffers };
  }
  
  // Loads all transformer layers concurrently
  async loadLayers(params: ModelParams, weightsFolder: string): Promise<LayerBuffers[]> {
    console.log(`Loading ${params.n_layer} transformer layers concurrently...`);
    const layerPromises: Promise<LayerBuffers>[] = [];

    for (let i = 0; i < params.n_layer; i++) {
      layerPromises.push(this.loadLayer(params, weightsFolder, i));
    }

    const layer_buffers = await Promise.all(layerPromises);
    console.log("All transformer layers loaded.");
    return layer_buffers;
  }

  // Loads weights for a single Llama 3 transformer layer
  async loadLayer(params: ModelParams, weightsFolder: string, layerIndex: number): Promise<LayerBuffers> {
    console.log(` Loading weights for layer ${layerIndex}...`);
    const prefix = `${weightsFolder}layers.${layerIndex}.`;
    const { n_embd, intermediate_size, n_head, n_kv_head, head_size } = params;

    // Define dimensions based on Llama parameters (weights are often stored [out_features, in_features])
    const qDim = [n_embd, n_embd]; // Wq maps input [n_embd] -> output [n_heads * head_dim = n_embd]
    const kDim = [n_kv_head * head_size, n_embd]; // Wk maps input [n_embd] -> output [n_kv_heads * head_dim]
    const vDim = [n_kv_head * head_size, n_embd]; // Wv maps input [n_embd] -> output [n_kv_heads * head_dim]
    const oDim = [n_embd, n_embd]; // Wo maps input [n_heads * head_dim = n_embd] -> output [n_embd]
    const w1Dim = [intermediate_size, n_embd]; // Gate proj W1
    const w3Dim = [intermediate_size, n_embd]; // Up proj W3
    const w2Dim = [n_embd, intermediate_size]; // Down proj W2

    const tensorPromises = [
      // Attention Norm
      this.fetchAndInitTensor(`${prefix}attention_norm.weight.bin`, [n_embd], ["storage"]),
      // Attention Weights
      this.fetchAndInitTensor(`${prefix}attention.wq.weight.bin`, qDim, ["storage"]),
      this.fetchAndInitTensor(`${prefix}attention.wk.weight.bin`, kDim, ["storage"]),
      this.fetchAndInitTensor(`${prefix}attention.wv.weight.bin`, vDim, ["storage"]),
      this.fetchAndInitTensor(`${prefix}attention.wo.weight.bin`, oDim, ["storage"]),
      // FFN Norm
      this.fetchAndInitTensor(`${prefix}ffn_norm.weight.bin`, [n_embd], ["storage"]),
      // MLP Weights (SwiGLU)
      this.fetchAndInitTensor(`${prefix}feed_forward.w1.weight.bin`, w1Dim, ["storage"]), // gate_proj
      this.fetchAndInitTensor(`${prefix}feed_forward.w3.weight.bin`, w3Dim, ["storage"]), // up_proj
      this.fetchAndInitTensor(`${prefix}feed_forward.w2.weight.bin`, w2Dim, ["storage"]), // down_proj
    ];

    // Wait for all tensors in this layer to be fetched and initialized
    const [
        normAttentionGammaBuffer,
        qWeightsBuffer,
        kWeightsBuffer,
        vWeightsBuffer,
        oWeightsBuffer,
        normLinearGammaBuffer,
        w1WeightsBuffer,
        w3WeightsBuffer,
        w2WeightsBuffer,
    ] = await Promise.all(tensorPromises);

    console.log(` Finished loading weights for layer ${layerIndex}`);
    return {
      normAttentionGammaBuffer,
      qWeightsBuffer,
      kWeightsBuffer,
      vWeightsBuffer,
      oWeightsBuffer,
      normLinearGammaBuffer,
      w1WeightsBuffer,
      w2WeightsBuffer,
      w3WeightsBuffer,
    };
  }


  // Helper to fetch binary data and initialize a GPU buffer tensor
  async fetchAndInitTensor(
      url: string,
      dims: readonly number[],
      usageOps: Array<keyof typeof global.bufferUsageDict>,
      isInt32: boolean = false
    ): Promise<GPUBuffer> {
    if (!this.device) throw new Error("Device not initialized");
    console.log(`  Fetching ${url}...`);
    try {
        // Assuming fetchBin returns Float32Array always, adjust if it can return other types
        const rawData = await global.fetchBin(url);
        let data: Float32Array | Int32Array;
        const bytesPerElement = isInt32 ? Int32Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT;
        const expectedElements = dims.reduce((a, b) => a * b, 1);
        const expectedBytes = expectedElements * bytesPerElement;

        if (rawData.byteLength !== expectedBytes) {
             console.error(`Tensor size mismatch for ${url}: Expected ${expectedBytes} bytes (${expectedElements} elements), Got ${rawData.byteLength} bytes`);
             throw new Error(`Tensor size mismatch for ${url}`);
        }

        data = isInt32 ? new Int32Array(rawData.buffer, rawData.byteOffset, expectedElements) : rawData;

        console.log(`  Initializing tensor ${url} [${dims.join(', ')}] (${isInt32 ? 'Int32' : 'Float32'})...`);
        return this.initTensor(data, dims, usageOps);

    } catch (error) {
         console.error(`Failed to fetch or init tensor ${url}:`, error);
         throw error;
    }
  }

  // Initializes a GPU buffer with the provided data (Float32 or Int32)
  initTensor(
      data: Float32Array | Int32Array,
      dims: readonly number[],
      usageOps: Array<keyof typeof global.bufferUsageDict>,
      mapAtCreation: boolean = true, // Default to mapping for weights/inputs
      dataType: 'float32' | 'int32' = 'float32' // Explicit type for clarity
    ): GPUBuffer {
    if (!this.device) throw new Error("Device not initialized");

    const bytesPerElement = dataType === 'int32' ? Int32Array.BYTES_PER_ELEMENT : Float32Array.BYTES_PER_ELEMENT;
    const elementCount = dims.reduce((a, b) => a * b, 1);
    const unalignedSize = elementCount * bytesPerElement;
    const bufferSizeBytes = this.bufferSize(...dims, bytesPerElement); // Pass bytesPerElement

    if (data.byteLength !== unalignedSize) {
         console.warn(`Data size (${data.byteLength}) != expected unaligned size (${unalignedSize}) for dims [${dims.join(', ')}]. Ensure data type matches.`);
    }
     if (data.byteLength > bufferSizeBytes) {
        console.warn(`Data size (${data.byteLength}) > calculated aligned buffer size (${bufferSizeBytes}). Check alignment logic or data.`);
        // Decide how to handle: truncate (dangerous) or error out?
    }

    const bufferDescriptor: GPUBufferDescriptor = {
        label: `Tensor[${dims.join(',')}]_${dataType}`, // Optional label
        size: bufferSizeBytes,
        usage: usageOps.map(global.getBufferUsage).reduce((a, b) => a | b),
        mappedAtCreation: mapAtCreation,
    };
    const buffer = this.device.createBuffer(bufferDescriptor);

    if (mapAtCreation) {
        const mappedRange = buffer.getMappedRange();
        const constructor = dataType === 'int32' ? Int32Array : Float32Array;
        // Create a typed array view of the correct type *and size* for the source data
        const sourceView = data;
         // Create a typed array view for the destination mapped range
         // Use byteLength / bytesPerElement for the length of the destination view
         const destinationView = new constructor(mappedRange, 0, bufferSizeBytes / bytesPerElement);

        // Copy the data. If padding is needed, the extra space in destinationView remains 0.
        destinationView.set(sourceView);

        buffer.unmap();
    } else {
         // If not mapped at creation, queue a write operation
         this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
         console.warn(`Using queue.writeBuffer for tensor init. mappedAtCreation is generally preferred for weights.`);
    }

    this.unloadDeletionStack.push(buffer); // Add to stack for later cleanup
    return buffer;
  }

  // Calculates buffer size aligned to minimum offset requirements
  bufferSize(...dimsAndBytesPerElement: readonly number[]): number {
    const bytesPerElement = dimsAndBytesPerElement[dimsAndBytesPerElement.length - 1];
    const dims = dimsAndBytesPerElement.slice(0, -1);

    // If only one number provided, assume it's already the byte size
    if (dims.length === 0 && dimsAndBytesPerElement.length === 1) {
        return Math.ceil(bytesPerElement / this.minBufferOffset) * this.minBufferOffset;
    }
     // If bytesPerElement wasn't the last arg, default to float32
     const actualBytesPerElement = (bytesPerElement === Int32Array.BYTES_PER_ELEMENT || bytesPerElement === Float32Array.BYTES_PER_ELEMENT)
            ? bytesPerElement
            : Float32Array.BYTES_PER_ELEMENT;
     const actualDims = (bytesPerElement === actualBytesPerElement) ? dims : dimsAndBytesPerElement;


    const elementCount = actualDims.reduce((a, b) => a * b, 1);
    const unalignedSize = elementCount * actualBytesPerElement;
    const alignedSize = Math.ceil(unalignedSize / this.minBufferOffset) * this.minBufferOffset;
    return alignedSize;
  }

  // Destroys GPU buffers created during loading
  unloadBuffers(): void {
    console.log(`Unloading ${this.unloadDeletionStack.length} GPU buffers...`);
    this.unloadDeletionStack.forEach((buffer) => {
        try {
            buffer.destroy();
        } catch (e) {
             console.warn("Error destroying buffer:", e);
        }
    });
    this.unloadDeletionStack = [];
    this.model = undefined; // Clear model references
    this.initialized = false; // Reset initialized status
    console.log("Buffers unloaded.");
  }
}