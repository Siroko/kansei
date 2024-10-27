import { IBindable } from "./IBindable";

/**
 * Base class for GPU buffer management implementing the IBindable interface.
 * Provides core functionality for creating and managing WebGPU buffers.
 */
class BufferBase implements IBindable {
    /** Identifies a storage buffer type */
    public static BUFFER_TYPE_STORAGE: string = 'storage';
    /** Identifies a uniform buffer type */
    public static BUFFER_TYPE_UNIFORM: string = 'uniform';
    /** Identifies a read-only storage buffer type */
    public static BUFFER_TYPE_READ_ONLY_STORAGE: string = 'read-only-storage';

    /** Buffer can be mapped for reading */
    public static BUFFER_USAGE_MAP_READ: GPUFlagsConstant = 0x0001;
    /** Buffer can be mapped for writing */
    public static BUFFER_USAGE_MAP_WRITE: GPUFlagsConstant = 0x0002;
    /** Buffer can be used as a copy source */
    public static BUFFER_USAGE_COPY_SRC: GPUFlagsConstant = 0x0004;
    /** Buffer can be used as a copy destination */
    public static BUFFER_USAGE_COPY_DST: GPUFlagsConstant = 0x0008;
    /** Buffer can be used as an index buffer */
    public static BUFFER_USAGE_INDEX: GPUFlagsConstant = 0x0010;
    /** Buffer can be used as a vertex buffer */
    public static BUFFER_USAGE_VERTEX: GPUFlagsConstant = 0x0020;
    /** Buffer can be used as a uniform buffer */
    public static BUFFER_USAGE_UNIFORM: GPUFlagsConstant = 0x0040;
    /** Buffer can be used as a storage buffer */
    public static BUFFER_USAGE_STORAGE: GPUFlagsConstant = 0x0080;
    /** Buffer can be used for indirect draws */
    public static BUFFER_USAGE_INDIRECT: GPUFlagsConstant = 0x0100;
    /** Buffer can be used for query resolves */
    public static BUFFER_USAGE_QUERY_RESOLVE: GPUFlagsConstant = 0x0200;

    /** Default usage flags for uniform buffer with copy destination capability */
    public usage: GPUFlagsConstant = BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST;
    /** Buffer type identifier */
    public type?: string | undefined;
    /** Indicates if the buffer has been initialized */
    public initialized: boolean = false;
    /** Indicates if the buffer needs to be updated */
    public needsUpdate: boolean = false;
    /** Shader location binding point */
    public shaderLocation?: number;
    /** Byte offset into the buffer */
    public offset?: number;
    /** Stride between elements in bytes */
    public stride?: number;
    /** Vertex format for vertex buffers */
    public format?: GPUVertexFormat;

    /** 
     * Returns the GPU buffer binding configuration
     * @returns GPUBufferBinding object
     */
    get resource(): GPUBufferBinding {
        return {
            buffer: this._resource!
        };
    }
    /** Internal GPU buffer reference */
    protected _resource?: GPUBuffer;
    /** Internal buffer data storage */
    protected buffer?: Float32Array;

    /**
     * Initializes the GPU buffer with the current data
     * @param gpuDevice The GPU device to create the buffer on
     */
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

    /**
     * Updates the GPU buffer with the current data
     * @param gpuDevice The GPU device to update the buffer on
     */
    public update(gpuDevice: GPUDevice) {
        if (this._resource) {
            gpuDevice!.queue.writeBuffer(this._resource, 0, this.buffer!);
        }
        this.needsUpdate = false;
    }
}

export { BufferBase };
