import { BufferBase } from "./BufferBase";

interface IComputeBufferOptions {
    type?: string;
    usage: GPUFlagsConstant;
    buffer?: Float32Array;
}

class ComputeBuffer extends BufferBase {

    public type?: string = ComputeBuffer.BUFFER_TYPE_STORAGE;
    needsUpdate: boolean = true;

    constructor(options: IComputeBufferOptions) {
        super();
        this.type = options.type;
        this.usage = options.usage;
        this.buffer = options.buffer;
    }

    public initialize(gpuDevice: GPUDevice): void {
        this._resource = gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: this.buffer!.byteLength,
            usage: this.usage
        });
        new Float32Array(this._resource.getMappedRange()).set(this.buffer!);
        this._resource.unmap();

        this.initialized = true;
    }

    public setBuffer(buffer: Float32Array, usage: GPUFlagsConstant = GPUBufferUsage.STORAGE) {
        this.buffer = buffer;
        this.usage = usage;
    }
}

export { ComputeBuffer };