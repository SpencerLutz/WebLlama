import { MatMulDemoBlock } from "./blocks/matMulDemo/MatMulDemoBlock";
import { PassConfig } from "./types";

export class Model {
    private device?: GPUDevice;
    private initialized: boolean = false;

    private model?: { [key: string]: any }; // probably solidify these types once
    private params?: { [key: string]: any }; // model loading is developed

    private inputIdxBuffer?: GPUBuffer;
    private resultBuffer?: GPUBuffer;
    private stagingBuffer?: GPUBuffer;

    private passes?: Array<PassConfig>;

    private maxSequenceLength: number = 1024;

    private matMulDemoBlock?: MatMulDemoBlock;
    // other blocks here

    async init() {
        if (this.initialized) return console.error("Model already initialized");
        if (!navigator.gpu) throw new Error("WebGPU is not supported");

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("Error: Could not get adapter");
        this.device = await adapter.requestDevice();

        this.loadModel(":)");

        this.inputIdxBuffer = this.device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT * this.maxSequenceLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const passResults = this.getModelPasses();
        this.passes = passResults.passes;
        this.resultBuffer = passResults.resultBuffer;

        this.stagingBuffer = this.device.createBuffer({
            size: this.resultBuffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        // TODO: load params, tokenizer, initialize hyperparameters

        this.initialized = true;
    }

    private async loadModel(dir: string): Promise<void> {
        // would load model stuff from dir here

        // begin placeholder
        const demoBuffer1 = this.device!.createBuffer({ 
            size: 4 * 64 * 1, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST 
        });
        const demoBuffer2 = this.device!.createBuffer({ 
            size: 4 * 1 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST 
        });
        const f32Data1 = new Float32Array(64 * 1);
        const f32Data2 = new Float32Array(1 * 64);
        for (let i = 0; i < 64 * 1; i++) {
            f32Data1[i] = Math.random();
            f32Data2[i] = Math.random();
        }
        this.device!.queue.writeBuffer(demoBuffer1, 0, f32Data1);
        this.device!.queue.writeBuffer(demoBuffer2, 0, f32Data2);
        this.model = { demoBuffer1, demoBuffer2 };
        this.params = { placeholder: "placeholder" };
        // end placeholder
    }

    private getModelPasses(): {
        passes: Array<PassConfig>;
        resultBuffer: GPUBuffer;
    } {
        const { demoBuffer1, demoBuffer2 } = this.model!;
        const { placeholder } = this.params!;

        this.matMulDemoBlock = new MatMulDemoBlock(this.device!);
        // other blocks here

        const modelPasses: Array<PassConfig> = [];
        let intermediateBuffer: GPUBuffer;
        {
            const { passes, resultBuffer } = this.matMulDemoBlock.newInstance(
                64, 1, this.maxSequenceLength, 
                demoBuffer1, this.inputIdxBuffer!
            );
            intermediateBuffer = resultBuffer;
            modelPasses.push(...passes);
        }
        {
            const { passes, resultBuffer } = this.matMulDemoBlock.newInstance(
                1, 64, this.maxSequenceLength,
                demoBuffer2, intermediateBuffer
            );
            intermediateBuffer = resultBuffer;
            modelPasses.push(...passes);
        }

        return { passes: modelPasses, resultBuffer: intermediateBuffer };
    }

    async *generate(prompt: string, max_new_tokens: number): AsyncGenerator<string, void, void>{
        if (!this.initialized) {
            console.error("Model not yet initialized.");
            return;
        }

        // tokenize prompt (placeholder tokenizer used here)
        const history = Array.from(prompt, c => c.charCodeAt(0) % 128);

        for (let i = 0; i < max_new_tokens; i++) {
            const historyArray = Uint32Array.from(history.slice(-this.maxSequenceLength));
            const logits = await this.run(historyArray);

            // TODO: do actual sampling 
            const next_idx = logits.indexOf(Math.max(...logits)) % 94 + 33;
            history.push(next_idx);

            const next_tok = String.fromCharCode(next_idx);
            yield next_tok;
        }
    }

    private async run(history: Uint32Array): Promise<Float32Array> {
        // this is just because we wouldn't normally be doing a matmul on the token idx
        const floatHistory = new Float32Array(history.length);
        for (let i = 0; i < history.length; i++) {
            floatHistory[i] = history[i];
        }

        this.device!.queue.writeBuffer(this.inputIdxBuffer!, 0, floatHistory);

        const commandEncoder = this.device!.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        for (const pass of this.passes!) {
            passEncoder.setPipeline(pass.pipeline);
            for (let i = 0; i < pass.bindGroups.length; i++) {
                passEncoder.setBindGroup(i, pass.bindGroups[i]);
            }
            const wgDims = pass.numWorkgroups.map(dim => 
                typeof dim === "function" ? dim(history.length) : dim
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
        const data = new Float32Array(output).slice();
        this.stagingBuffer!.unmap();

        return data;
    }
}