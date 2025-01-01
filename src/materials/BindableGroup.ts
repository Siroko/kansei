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
 * Represents a group of bindables for rendering or compute operations.
 */
class BindableGroup {
    public bindGroupLayout?: GPUBindGroupLayout;
    public bindGroup?: GPUBindGroup;
    public initialized: boolean = false;
    public pipelineBindGroupLayout?: GPUPipelineLayout;
    private bindableGroupLayout?: GPUBindGroupLayout;
    public cameraBindablesGroupLayout?: GPUBindGroupLayout;
    public meshBindablesGroupLayout?: GPUBindGroupLayout;

    /**
     * Constructs a new BindableGroup.
     * @param bindables - An array of BindGroupDescriptor objects.
     * @param isCompute - A boolean indicating if the group is for compute operations.
     */
    constructor(
        public bindables: BindGroupDescriptor[],
        public isCompute: boolean = false
    ) {
    }

    /**
     * Creates the bind group layout for rendering material bindables.
     * @param gpuDevice - The GPU device used to create the bind group layout.
     */
    public createRenderingBindGroupLayout(gpuDevice: GPUDevice) {
        this.cameraBindablesGroupLayout = gpuDevice.createBindGroupLayout({
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

        this.meshBindablesGroupLayout = gpuDevice.createBindGroupLayout({
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
     * Creates the bind group layout based on the provided bindables.
     * @param gpuDevice - The GPU device used to create the bind group layout.
     */
    public createBindGroupLayout(gpuDevice: GPUDevice) {
        const entries = [];
        for (const bindable of this.bindables) {
            const entry: GPUBindGroupLayoutEntry = {
                binding: bindable.binding!,
                visibility: bindable.visibility!
            };
            switch (bindable.value!.type) {
                case 'storage':
                case 'read-only-storage':
                case 'uniform':
                    entry.buffer = {
                        type: bindable.value!.type as GPUBufferBindingType
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
                        sampleType: 'float',
                    };
                    break;
                default:
                    console.error(`Unknown binding type: ${bindable.value!.type}`);
                    continue;
            }
            entries.push(entry);
        }

        this.bindGroupLayout = gpuDevice.createBindGroupLayout({
            label: 'BindableGroup BindGroupLayout',
            entries
        });
    }

    /**
     * Retrieves or creates the bind group for the bindables.
     * @param gpuDevice - The GPU device used to create or update the bind group.
     * @returns The created or existing GPUBindGroup.
     */
    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];

        if (this.bindGroup) {
            let isExternalTexture = false;
            for (const bindable of this.bindables) {
                if (bindable.value?.needsUpdate) {
                    bindable.value?.update(gpuDevice);
                }
                if (bindable.value?.type === 'external-texture') {
                    isExternalTexture = true;
                }
            }
            if (!isExternalTexture) {
                return this.bindGroup!;
            }
        }
        for (const bindable of this.bindables) {
            if (!bindable.value?.initialized) {
                bindable.value?.initialize(gpuDevice);
            }
            if (!this.bindableGroupLayout) {
                this.createBindGroupLayout(gpuDevice)
            }
            if (!this.pipelineBindGroupLayout) {
                this.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
                    bindGroupLayouts: [this.bindGroupLayout!]
                });
            }

            entries.push({
                binding: bindable.binding!,
                resource: bindable.value!.resource!,
            });
        }

        this.bindGroup = gpuDevice.createBindGroup({
            label: 'BindableGroup',
            layout: this.bindGroupLayout!,
            entries: entries
        });

        return this.bindGroup!;
    }
}

export { BindableGroup };
