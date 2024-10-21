import { BufferBase } from "./BufferBase";

interface IComputeBufferOptions {
    type?: string;
    usage: GPUFlagsConstant;
    buffer?: Float32Array;
}

class ComputeBuffer extends BufferBase {

    public type?: string = ComputeBuffer.BUFFER_TYPE_STORAGE;

    constructor(options: IComputeBufferOptions) {
        super();
        this.type = options.type;
        this.usage = options.usage;
        this.buffer = options.buffer;
    }
}

export { ComputeBuffer };