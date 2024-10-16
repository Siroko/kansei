import { IBindable } from "./IBindable";

class BufferBase implements IBindable {
    public static BUFFER_TYPE_STORAGE: string = 'storage';
    public static BUFFER_TYPE_UNIFORM: string = 'uniform';
    public static BUFFER_TYPE_READ_ONLY_STORAGE: string = 'read-only-storage';

    public usage: GPUFlagsConstant = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    public type?: string | undefined;
    public initialized: boolean = false;
    public needsUpdate: boolean = false;

    get resource(): GPUBufferBinding {
        return { buffer: this._resource! };
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
    }

    public update(gpuDevice: GPUDevice) {
        if (this._resource) {
            gpuDevice!.queue.writeBuffer(this._resource, 0, this.buffer!);
        }
        this.needsUpdate = false;
    }
}

export { BufferBase };