import { IBindable } from "../buffers/IBindable";
import { cameraBindGroupLayoutEntries, meshBindGroupLayoutEntries } from "../renderers/SharedLayouts";

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
    public shadowBindablesGroupLayout?: GPUBindGroupLayout;

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
     * Creates the shared layouts of a render pipeline's groups 1-3 (see renderers/SharedLayouts).
     * @param gpuDevice - The GPU device used to create the bind group layouts.
     */
    public createRenderingBindGroupLayout(gpuDevice: GPUDevice) {
        // Group 1: camera (view, projection, scene lights, temporal data).
        this.cameraBindablesGroupLayout = gpuDevice.createBindGroupLayout({
            label: 'Camera BindGroupLayout',
            entries: cameraBindGroupLayoutEntries(),
        });

        // Group 2: per-object normal and world matrices. hasDynamicOffset lets the
        // renderer select each object's slice of one large shared buffer, so all
        // objects' matrices upload in 2 writeBuffer calls per frame.
        this.meshBindablesGroupLayout = gpuDevice.createBindGroupLayout({
            label: 'Mesh BindGroupLayout',
            entries: meshBindGroupLayoutEntries(),
        });

        this.shadowBindablesGroupLayout = gpuDevice.createBindGroupLayout({
            label: 'Shadow BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
            ],
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
