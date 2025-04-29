import { MatMulDemoBlock } from "./blocks/matMulDemo/MatMulDemoBlock";

export class Model {
    private device?: GPUDevice;
    private initialized: boolean = false;

    private matMulDemoBlock?: MatMulDemoBlock;
    // other blocks here

    async init() {
        if (this.initialized) return console.error("Model already initialized");
        if (!navigator.gpu) throw new Error("WebGPU is not supported");

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("Error: Could not get adapter");
        this.device = await adapter.requestDevice();

        this.matMulDemoBlock = new MatMulDemoBlock(this.device);
        // other blocks here

        // TODO: load params, tokenizer, initialize hyperparameters

        this.initialized = true;
    }

    async *generate(prompt: string, max_new_tokens: number): AsyncGenerator<string, void, void>{
        if (!this.initialized) {
            console.error("Model not yet initialized.");
            return;
        }

        // tokenize prompt (placeholder tokenizer used here)
        const history = Array.from(prompt, c => c.charCodeAt(0) % 128);

        for (let i = 0; i < max_new_tokens; i++) {
            // TODO: trim to context length
            const logits = await this.run(history)
            // TODO: do actual sampling 
            const next_idx = logits.indexOf(Math.max(...logits));
            history.push(next_idx);

            const next_tok = String.fromCharCode(next_idx);
            yield next_tok;
        }
    }

    async run(history: Array<number>): Promise<Array<number>> {
        // actual stuff here

        return Array.from({length: 128}, (_, i) => 
            (32 <= i && i <= 126) ? Math.random() : 0
        );
    }
}