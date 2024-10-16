import { BindGroupDescriptor } from "./Material";

class UniformGroup {
    public bindGroupLayout?: GPUBindGroupLayout;
    public bindGroup?: GPUBindGroup;
    public initialized: boolean = false;

    constructor(
        public uniforms: BindGroupDescriptor[]
    ) {
    }

    public createBindGroupLayout(gpuDevice: GPUDevice) {
        const entries = [];
        for (const uniform of this.uniforms) {
            const entry: GPUBindGroupLayoutEntry = {
                binding: uniform.binding!,
                visibility: uniform.visibility!
            };
            switch (uniform.value!.type) {
                case 'storage':
                case 'read-only-storage':
                case 'uniform':
                    entry.buffer = {
                        type: uniform.value!.type as GPUBufferBindingType
                    };
                    break;
                case 'sampler':
                    entry.sampler = { type: 'filtering' };
                    break;
                case 'texture':
                    entry.texture = { sampleType: 'float' };
                    break;
                case 'storage-texture':
                    entry.storageTexture = {
                        access: 'write-only',
                        format: 'rgba8unorm'
                    };
                    break;
                case 'external-texture':
                    entry.externalTexture = {
                        sampleType: 'float'
                    };
                    break;
                default:
                    console.error(`Unknown binding type: ${uniform.value!.type}`);
                    continue;
            }
            entries.push(entry);
        }

        this.bindGroupLayout = gpuDevice.createBindGroupLayout({
            entries
        });
    }

    public async getBindGroup(gpuDevice: GPUDevice, pipeline: GPURenderPipeline, bindingGroupLayoutPosition: number = 0): Promise<GPUBindGroup> {
        const entries: GPUBindGroupEntry[] = [];

        if (!this.bindGroupLayout) {
            this.createBindGroupLayout(gpuDevice);
        }

        for (const uniform of this.uniforms) {
            if (!uniform.value?.initialized) {
                await uniform.value?.initialize(gpuDevice);
            }

            if (uniform.value?.needsUpdate) {
                await uniform.value?.update(gpuDevice);
            }

            entries.push({
                binding: uniform.binding ?? 0,
                resource: uniform.value!.resource!,
            });
        }

        this.bindGroup = gpuDevice.createBindGroup({
            label: 'UniformGroup ' + bindingGroupLayoutPosition,
            layout: pipeline.getBindGroupLayout(bindingGroupLayoutPosition),
            entries: entries
        });

        return new Promise((resolve) => {
            resolve(this.bindGroup!);
        });
    }
}

export { UniformGroup };