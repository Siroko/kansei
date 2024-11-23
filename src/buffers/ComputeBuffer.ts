import { BufferBase } from "./BufferBase";

/**
 * Configuration options for creating a compute buffer
 */
interface IComputeBufferOptions {
    /** Type of buffer (e.g. storage, uniform) */
    type?: string;
    /** GPU buffer usage flags */
    usage: GPUFlagsConstant;
    /** Initial buffer data */
    buffer?: Float32Array;
    /** Binding location in shader */
    shaderLocation?: number;
    /** Byte offset within buffer */
    offset?: number;
    /** Byte stride between elements */
    stride?: number;
    /** Vertex format for the buffer */
    format?: GPUVertexFormat;
}

/**
 * Represents a GPU buffer used for compute operations
 * @extends BufferBase
 */
class ComputeBuffer extends BufferBase {
    /** Default buffer type for compute operations */
    public type?: string = ComputeBuffer.BUFFER_TYPE_STORAGE;

    /**
     * Creates a new compute buffer
     * @param options - Configuration options for the buffer
     */
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

    /**
     * Clones the current buffer
     * @returns A new buffer instance with the same data
     */
    public clone(): ComputeBuffer {
        return new ComputeBuffer({
            type: this.type,
            usage: this.usage,
            buffer: this.buffer?.slice(),
            shaderLocation: this.shaderLocation,
            offset: this.offset,
            stride: this.stride,
            format: this.format
        });
    }
}

export { ComputeBuffer };
