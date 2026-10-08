import { BufferBase } from "./BufferBase";

/**
 * Describes a single vertex attribute within a compute buffer.
 */
interface IComputeBufferAttribute {
    shaderLocation: number;
    offset: number;
    format: GPUVertexFormat;
}

/**
 * Configuration options for creating a compute buffer
 */
interface IComputeBufferOptions {
    /** Type of buffer (e.g. storage, uniform) */
    type?: string;
    /** GPU buffer usage flags */
    usage: GPUFlagsConstant;
    /** Initial buffer data */
    buffer?: Float32Array | Uint32Array | Int32Array;
    /** Binding location in shader */
    shaderLocation?: number;
    /** Byte offset within buffer */
    offset?: number;
    /** Byte stride between elements */
    stride?: number;
    /** Vertex format for the buffer */
    format?: GPUVertexFormat;
    /** Multiple vertex attributes from one buffer (overrides shaderLocation/offset/format). */
    attributes?: IComputeBufferAttribute[];
}

/**
 * Represents a GPU buffer used for compute operations
 * @extends BufferBase
 */
class ComputeBuffer extends BufferBase {
    /** Default buffer type for compute operations */
    public type?: string = ComputeBuffer.BUFFER_TYPE_STORAGE;
    /** Multiple vertex attributes from one buffer. */
    public attributes?: IComputeBufferAttribute[];

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
        this.attributes = options.attributes;
    }

    /**
     * Wrap a buffer created elsewhere (another system's output, an indirect-args buffer) so it
     * binds like any other buffer, as `type` (`storage`, `read-only-storage` or `uniform`) over
     * `size` bytes from `offset` (the rest of the buffer when `size` is unset). Nothing is
     * uploaded: its owner writes it.
     */
    public static fromExternal(buffer: GPUBuffer, type: string, range: { offset?: number; size?: number } = {}): ComputeBuffer {
        const wrapped = new ComputeBuffer({ type, usage: buffer.usage });
        wrapped._resource = buffer;
        if (range.offset !== undefined || range.size !== undefined) {
            wrapped.bindingRange = { offset: range.offset ?? 0, size: range.size };
        }
        wrapped.initialized = true;
        return wrapped;
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
            format: this.format,
            attributes: this.attributes?.map(a => ({ ...a })),
        });
    }
}

export { ComputeBuffer };
