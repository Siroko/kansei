import { IBindable } from "./IBindable";

class BufferBase implements IBindable {
    public static BUFFER_TYPE_STORAGE: string = 'storage';
    public static BUFFER_TYPE_UNIFORM: string = 'uniform';
    public static BUFFER_TYPE_READ_ONLY_STORAGE: string = 'read-only-storage';

    public static BUFFER_USAGE_MAP_READ: GPUFlagsConstant = 0x0001;
    public static BUFFER_USAGE_MAP_WRITE: GPUFlagsConstant = 0x0002;
    public static BUFFER_USAGE_COPY_SRC: GPUFlagsConstant = 0x0004;
    public static BUFFER_USAGE_COPY_DST: GPUFlagsConstant = 0x0008;
    public static BUFFER_USAGE_INDEX: GPUFlagsConstant = 0x0010;
    public static BUFFER_USAGE_VERTEX: GPUFlagsConstant = 0x0020;
    public static BUFFER_USAGE_UNIFORM: GPUFlagsConstant = 0x0040;
    public static BUFFER_USAGE_STORAGE: GPUFlagsConstant = 0x0080;
    public static BUFFER_USAGE_INDIRECT: GPUFlagsConstant = 0x0100;
    public static BUFFER_USAGE_QUERY_RESOLVE: GPUFlagsConstant = 0x0200;

    public usage: GPUFlagsConstant = BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST;
    public type?: string | undefined;
    public initialized: boolean = false;
    public needsUpdate: boolean = false;
    public shaderLocation?: number;
    public offset?: number;
    public stride?: number;
    public format?: GPUVertexFormat;

    get resource(): GPUBufferBinding {
        return {
            buffer: this._resource!
        };
    }
    protected _resource?: GPUBuffer;
    protected buffer?: Float32Array;

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

    public update(gpuDevice: GPUDevice) {
        if (this._resource) {
            gpuDevice!.queue.writeBuffer(this._resource, 0, this.buffer!);
        }
        this.needsUpdate = false;
    }
}

export { BufferBase };