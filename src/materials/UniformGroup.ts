import { IBindable } from "../buffers/IBindable";

export class BindGroupDescriptor {
    binding?: number;
    visibility?: GPUFlagsConstant;
    value?: IBindable;
}
class UniformGroup {
    public bindGroupLayout?: GPUBindGroupLayout;
    public bindGroup?: GPUBindGroup;
    public initialized: boolean = false;
    public pipelineBindGroupLayout?: GPUPipelineLayout;
    private uniformGroupLayout?: GPUBindGroupLayout;
    public cameraUniformsGroupLayout?: GPUBindGroupLayout;
    public meshUniformsGroupLayout?: GPUBindGroupLayout;

    constructor(
        public uniforms: BindGroupDescriptor[],
        public isCompute: boolean = false
    ) {
    }

    public createRenderingMaterialUniformsBindGroupLayout(gpuDevice: GPUDevice) {
        this.cameraUniformsGroupLayout = gpuDevice.createBindGroupLayout({
            label: 'Camera BindGroupLayout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: 'uniform'
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: 'uniform'
                    }
                }
            ]
        });

        this.meshUniformsGroupLayout = gpuDevice.createBindGroupLayout({
            label: 'Mesh BindGroupLayout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: 'uniform'
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: 'uniform'
                    }
                }
            ]
        });
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
            label: 'UniformGroup BindGroupLayout',
            entries
        });
    }

    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];
        let needsRebind = false;

        for (const uniform of this.uniforms) {
            if (!uniform.value?.initialized) {
                uniform.value?.initialize(gpuDevice);
            }

            if (uniform.value?.needsUpdate) {
                uniform.value?.update(gpuDevice);
                needsRebind = true;
            }

            if (!this.uniformGroupLayout) {
                this.createBindGroupLayout(gpuDevice)
            }
            if (!this.pipelineBindGroupLayout) {
                this.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
                    bindGroupLayouts: [this.bindGroupLayout!]
                });
            }

            entries.push({
                binding: uniform.binding ?? 0,
                resource: uniform.value!.resource!,
            });
        }

        if (this.bindGroup && !needsRebind) return this.bindGroup!;

        this.bindGroup = gpuDevice.createBindGroup({
            label: 'UniformGroup',
            layout: this.bindGroupLayout!,
            entries: entries
        });

        return this.bindGroup!;
    }
}

export { UniformGroup };