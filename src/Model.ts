import EmbedBlock from "./blocks/embed/EmbedBlock";
import RMSNormBlock from "./blocks/rmsNorm/RMSNormBlock";
import AttentionBlock from "./blocks/attention/AttentionBlock";
import ResidualBlock from "./blocks/residual/ResidualBlock";
import SwiGLUBlock from "./blocks/swiGLUBlock/SwiGLUBlock";
import DeEmbedBlock from "./blocks/deEmbed/DeEmbedBlock";

import { PassConfig, UsageString, LayerBuffers, LoadedModelBuffers, ModelParams } from "./types";

import Tokenizer from "./tokenizers/Tokenizer";

import { selectTopK, sampleFromLogits } from "./utils/sampling";
import { fetchBin } from "./utils/loading";
import { transpose, zeros } from "./utils/matrix";
import { bufferUsageDict } from "./utils/buffer";

export class Model {
    private initialized: boolean = false;

    private device?: GPUDevice;
    private model?: LoadedModelBuffers;
    private tokenizer?: Tokenizer;
    private params?: ModelParams;
    private minBufferOffset: number = 256;

    private defaultTopK: number = 5;
    private defaultTemperature: number = 0.6;
    private defaultTokens: number = 128;

    private passes?: Array<PassConfig>;

    private inputBuffer?: GPUBuffer;
    private positionBuffer?: GPUBuffer;
    private resultBuffer?: GPUBuffer;
    private stagingBuffer?: GPUBuffer;

    private unloadDeletionStack: GPUBuffer[] = [];

    /**
     * Initializes the model with the provided weights.
     * @param weightsDir: Directory in which model weights are stored
     */
    async init(weightsDir: string) {
        if (this.initialized) {
            console.warn("Model already initialized. " + 
                "Call unloadBuffers() first if re-initialization is needed.");
            return;
        }
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported on this browser.");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("Failed to get GPU adapter.");
        }
        
        this.device = await adapter.requestDevice();
        if (!this.device) {
            throw new Error("Failed to get GPU device.");
        }
        this.minBufferOffset = this.device.limits.minStorageBufferOffsetAlignment || 256;

        console.log(`WebGPU Device Acquired: ${this.device.label}`);
        
        const paramsFromJson = await this.loadLlamaParameters(weightsDir);
        if (!paramsFromJson) {
            throw new Error("Failed to load model parameters (config.json).");
        }
        this.params = this.deriveInternalParams(paramsFromJson, 2048);
        [this.model] = await this.loadModelWeights(weightsDir, this.params);

        this.inputBuffer = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT * this.params.n_ctx,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // this.initTensor(new Int32Array(token_ids), [seq_length], ["storage"], true, 'int32');

        this.positionBuffer = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT * this.params.n_ctx,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // this.initTensor(new Int32Array(token_ids), [seq_length], ["storage"], true, 'int32');


        ({ passes: this.passes, resultBuffer: this.resultBuffer } = this.getModelPasses());

        this.tokenizer = new Tokenizer();

        this.initialized = true;
        console.log("Model initialized successfully.");
    }

    private getModelPasses(): {
        passes: Array<PassConfig>;
        resultBuffer: GPUBuffer;
    } {
        const { layer_buffers, normGammaBuffer, embeddingsBuffers, deEmbeddingsBuffers } = this.model!;
        const { n_embd, n_head, n_kv_head, head_size, n_layer, vocab_size, vocab_chunk_size, 
                vocab_chunk_instances, intermediate_size, norm_eps, rope_theta, n_ctx } = this.params!;

        const embedBlock = new EmbedBlock(this.device!);
        const rmsNormBlock = new RMSNormBlock(this.device!);
        const attentionBlock = new AttentionBlock(this.device!);
        const residualBlock = new ResidualBlock(this.device!);
        const swiGLUBlock = new SwiGLUBlock(this.device!);
        const deEmbedBlock = new DeEmbedBlock(this.device!);

        const modelPasses: Array<PassConfig> = [];

        let hiddenBuffer: GPUBuffer;
        let residualBuffer: GPUBuffer;

        console.log(embeddingsBuffers.length);
        console.log(embeddingsBuffers[0].size);

        {
            const { passes, resultBuffer } = embedBlock.newInstance(
                this.inputBuffer!, embeddingsBuffers,
                n_embd, vocab_chunk_size, vocab_chunk_instances, n_ctx
            );
            hiddenBuffer = residualBuffer = resultBuffer;
            modelPasses.push(...passes);
        }

        for (let i = 0; i < n_layer; i++) {
            const buffers = layer_buffers[i];

            {
                const { passes, resultBuffer } = rmsNormBlock.newInstance(
                    residualBuffer, buffers.normAttentionGammaBuffer, // RMSNorm weights
                    n_embd, norm_eps, n_ctx
                );
                hiddenBuffer = resultBuffer;
                modelPasses.push(...passes);
            }

            {
                const { passes, resultBuffer } = attentionBlock.newInstance(
                    hiddenBuffer, this.positionBuffer!,
                    buffers.qWeightsBuffer, buffers.kWeightsBuffer,
                    buffers.vWeightsBuffer, buffers.oWeightsBuffer,
                    n_embd, n_head, n_kv_head,
                    head_size, rope_theta, n_ctx
                );
                hiddenBuffer = resultBuffer;
                modelPasses.push(...passes);
            }

            {
                const { passes, resultBuffer } = residualBlock.newInstance(
                    hiddenBuffer, residualBuffer,
                    n_embd, n_ctx
                );
                residualBuffer = resultBuffer;
                modelPasses.push(...passes);
            }

            {
                const { passes, resultBuffer } = rmsNormBlock.newInstance(
                    residualBuffer, buffers.normLinearGammaBuffer,
                    n_embd, norm_eps, n_ctx
                );
                hiddenBuffer = resultBuffer;
                modelPasses.push(...passes);
            }

            {
                const { passes, resultBuffer } = swiGLUBlock.newInstance(
                    hiddenBuffer, buffers.w1WeightsBuffer, // Gate projection weights
                    buffers.w3WeightsBuffer, buffers.w2WeightsBuffer, // Up & Down projection weights
                    n_embd, intermediate_size, n_ctx
                );
                hiddenBuffer = resultBuffer;
                modelPasses.push(...passes);
            }

            {
                const { passes, resultBuffer } = residualBlock.newInstance(
                    hiddenBuffer, residualBuffer,
                    n_embd, n_ctx
                );
                residualBuffer = resultBuffer;
                modelPasses.push(...passes);
            }
        }

        {
            const { passes, resultBuffer } = rmsNormBlock.newInstance(
                residualBuffer, normGammaBuffer,
                n_embd, norm_eps, n_ctx
            );
            hiddenBuffer = resultBuffer;
            modelPasses.push(...passes);
        }

        {
            const { passes, resultBuffer } = deEmbedBlock.newInstance(
                hiddenBuffer, deEmbeddingsBuffers, // Transposed embedding weights
                n_embd, vocab_size, vocab_chunk_size,
                vocab_chunk_instances, n_ctx
            ); // Output buffer: Logits [seq_length, vocab_size]
            hiddenBuffer = resultBuffer;
            modelPasses.push(...passes);
        }

        return { passes: modelPasses, resultBuffer: hiddenBuffer };
    }

    /**
     * Yields new tokens from the prompt. 
     * @param prompt Input string
     * @param max_new_tokens (Optional) Number of new tokens at which to stop generation
     * @param top_k (Optional) Number of top logits to sample from
     * @param temperature (Optional) Generation temperature
     * @yields Successive new tokens
     */
    async *generate(
        prompt: string,
        max_new_tokens: number = this.defaultTokens,
        top_k: number = this.defaultTopK,
        temperature: number = this.defaultTemperature
    ): AsyncGenerator<string, void, undefined> {
        // Add performance stuff back
        if (!this.initialized) {
            console.error("Model not properly initialized. Call initialize() first.");
            return;
        }
        if (prompt === "") {
            console.warn("Prompt is empty.");
            return;
        }

        let history = this.tokenizer!.encode(prompt);

        const eos_token_id = 128001;

        for (let i = 0; i < max_new_tokens; i++) {
            const inputTokens = history.slice(-this.params!.n_ctx);
            const positions = Array.from({ length: inputTokens.length }, (_, k) => k)

            // TODO: KV Cache
            const logits = await this.run(inputTokens, positions);

            // TODO: Implement top-p sampling? Llama often uses it.
            const { topKIndices, topKProbs } = selectTopK(logits, top_k);
            const topKProbs_float = new Float32Array(topKProbs);
            const idx_next = topKIndices[sampleFromLogits(topKProbs_float, temperature)];

            if (idx_next === eos_token_id) break;

            history = history.concat(idx_next);
            yield this.tokenizer!.decode([idx_next]);;
        }
    }

    /**
     * Executes the inference pipeline on the GPU.
     * @param history Input token IDs for this step
     * @param positions Positional indices corresponding to token_ids
     * @returns A Float32Array containing the logits for the input sequence
     */
    private async run(inputTokens: number[], positions: number[]): Promise<Float32Array> {

        this.device!.queue.writeBuffer(this.inputBuffer!, 0, new Uint32Array(inputTokens));
        this.device!.queue.writeBuffer(this.positionBuffer!, 0, new Uint32Array(positions));

        const commandEncoder = this.device!.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        for (const pass of this.passes!) {
            passEncoder.setPipeline(pass.pipeline);
            for (let i = 0; i < pass.bindGroups.length; i++) {
                passEncoder.setBindGroup(i, pass.bindGroups[i]);
            }
            const wgDims = pass.numWorkgroups.map(dim => 
                typeof dim === "function" ? dim(inputTokens.length) : dim
            );
            passEncoder.dispatchWorkgroups(wgDims[0], wgDims[1], wgDims[2]);
        }

        passEncoder.end();
        commandEncoder.copyBufferToBuffer(
            this.resultBuffer!, 0, this.stagingBuffer!, 0, this.resultBuffer!.size
        );

        this.device!.queue.submit([commandEncoder.finish()]);

        await this.stagingBuffer!.mapAsync(GPUMapMode.READ);
        const output = this.stagingBuffer!.getMappedRange();
        const data = new Float32Array(output).slice(); // check for padding?
        this.stagingBuffer!.unmap();

        return data;
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
        const { embeddingsBuffers, deEmbeddingsBuffers } = await this.loadEmbeddings(params, weightsFolder);

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
            embeddingsBuffers,
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
        const embeddingWeights = await fetchBin(`${weightsFolder}/tok_embeddings.weight.bin`);

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
                    zeros((params.vocab_chunk_size - size) * params.n_embd), 
                    size * params.n_embd  // offset where to start writing zeros
                );
            } else {
                paddedArray.set(embeddingWeights.subarray(offset * params.n_embd, offset * params.n_embd + size * params.n_embd));
            }

            embeddingsBuffers.push(this.initTensor(paddedArray, [params.vocab_chunk_size, params.n_embd], ["storage", "copy_src"]));

            const chunk = transpose(paddedArray, params.vocab_chunk_size, params.n_embd); // Use GPU perhaps?
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
        usageOps: Array<UsageString>,
        isInt32: boolean = false
    ): Promise<GPUBuffer> {
        if (!this.device) throw new Error("Device not initialized");
        console.log(`  Fetching ${url}...`);
        try {
            // Assuming fetchBin returns Float32Array always, adjust if it can return other types
            const rawData = await fetchBin(url);
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
        usageOps: Array<UsageString>,
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
            usage: usageOps.map((usage) => bufferUsageDict[usage]).reduce((a, b) => a | b),
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