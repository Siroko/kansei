import { BufferBase } from "./BufferBase";

interface IComputeBufferOptions {
    type?: string;
    usage: GPUFlagsConstant;
    buffer?: Float32Array;
    shaderLocation?: number;
    offset?: number;
    stride?: number;
    format?: GPUVertexFormat;
}

class ComputeBuffer extends BufferBase {

    public type?: string = ComputeBuffer.BUFFER_TYPE_STORAGE;

    constructor(options: IComputeBufferOptions) {
        super();
        this.type = options.type;
        this.usage = options.usage;
        this.buffer = options.buffer;
        this.shaderLocation = options.shaderLocation;
        this.offset = options.offset;
        this.stride = options.stride;
        this.format = options.format;
    }
}

export { ComputeBuffer };