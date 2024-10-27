import { IBindable } from "../buffers/IBindable";

/**
 * Represents a descriptor for a bind group, which includes binding, visibility, and value.
 */
export class BindGroupDescriptor {
    binding?: number;
    visibility?: GPUFlagsConstant;
    value?: IBindable;
}

/**
 * Represents a group of uniforms for rendering or compute operations.
 */
class UniformGroup {
    public bindGroupLayout?: GPUBindGroupLayout;
    public bindGroup?: GPUBindGroup;
    public initialized: boolean = false;
    public pipelineBindGroupLayout?: GPUPipelineLayout;
    private uniformGroupLayout?: GPUBindGroupLayout;
    public cameraUniformsGroupLayout?: GPUBindGroupLayout;
    public meshUniformsGroupLayout?: GPUBindGroupLayout;

    /**
     * Constructs a new UniformGroup.
     * @param uniforms - An array of BindGroupDescriptor objects.
     * @param isCompute - A boolean indicating if the group is for compute operations.
     */
    constructor(
        public uniforms: BindGroupDescriptor[],
        public isCompute: boolean = false
    ) {
    }

    /**
     * Creates the bind group layout for rendering material uniforms.
     * @param gpuDevice - The GPU device used to create the bind group layout.
     */
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

    /**
     * Creates the bind group layout based on the provided uniforms.
     * @param gpuDevice - The GPU device used to create the bind group layout.
     */
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

    /**
     * Retrieves or creates the bind group for the uniforms.
     * @param gpuDevice - The GPU device used to create or update the bind group.
     * @returns The created or existing GPUBindGroup.
     */
    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];

        if (this.bindGroup) {
            for (const uniform of this.uniforms) {
                if (uniform.value?.needsUpdate) {
                    uniform.value?.update(gpuDevice);
                }
            }
            return this.bindGroup!;
        }
        for (const uniform of this.uniforms) {
            if (!uniform.value?.initialized) {
                uniform.value?.initialize(gpuDevice);
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
                binding: uniform.binding!,
                resource: uniform.value!.resource!,
            });
        }

        if (this.bindGroup) return this.bindGroup!;

        this.bindGroup = gpuDevice.createBindGroup({
            label: 'UniformGroup',
            layout: this.bindGroupLayout!,
            entries: entries
        });

        return this.bindGroup!;
    }
}

export { UniformGroup };
