import { BindingConfig, PassConfig } from "../../types";
import { Block } from "../Block";
import shaderCode from "./matMulDemo.wgsl?raw";

export class MatMulDemoBlock extends Block {
    /* Cache map is useful if your pipeline has variations, e.g. different
    checks in the kernel. Otherwise can just cache a single pipeline. */
    // pipelineCache: Map<string, GPUComputePipeline>

    pipeline?: GPUComputePipeline

    constructor(device: GPUDevice) {
        super(device);
    }

    newInstance(
        n: number, k: number, m: number, 
        matrixABuffer: GPUBuffer, matrixBBuffer: GPUBuffer
    ): {
        resultBuffer: GPUBuffer;
        passes: Array<PassConfig>;
    } {
        // define buffers and types for input bindings
        const inputBindGroupConfigs: BindingConfig[] = [
            { buffer: matrixABuffer, bufferType: "read-only-storage" },
            { buffer: matrixBBuffer, bufferType: "read-only-storage" }
        ];
        const { bindGroup: inputBindGroup, bindGroupLayout: inputBindGroupLayout } 
            = this.createBindGroup(inputBindGroupConfigs);

        // create other buffers
        const resultBuffer = this.createBuffer([n, m], ["storage", "copy_src"]);
        const metaBuffer = this.createBuffer([3], ["uniform", "copy_dst"]);
        this.writeBuffer(metaBuffer, [n, k, m]);
        
        // define buffers and types for other bindings
        const opBindGroupConfigs: Array<BindingConfig> = [
            { buffer: resultBuffer, bufferType: "storage" },
            { buffer: metaBuffer, bufferType: "uniform" }
        ];
        const { bindGroup: opBindGroup, bindGroupLayout: opBindGroupLayout } 
            = this.createBindGroup(opBindGroupConfigs);

        // if the pipeline isn't cached, create and cache it
        if (!this.pipeline) {
            this.pipeline = this.createPipeline(
                [inputBindGroupLayout, opBindGroupLayout], 
                shaderCode
            );
        }

        // configure passes (just one in this case)
        const passes: Array<PassConfig> = [{
            pipeline: this.pipeline,
            bindGroups: [inputBindGroup, opBindGroup],
            numWorkgroups: [Math.ceil(n / 8), (l) => Math.ceil(l / 8)]
        }];

        return { resultBuffer, passes };
    }
}