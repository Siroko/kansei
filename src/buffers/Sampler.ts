import { IBindable } from "./IBindable";
import type { BindingLayout } from "../materials/Binding";

/** Sampler settings beyond filters, address mode and anisotropy. */
export interface SamplerOptions {
    /** Makes a comparison sampler (`sampler_comparison`), as shadow maps read depth. */
    compare?: GPUCompareFunction;
    /** Defaults to the minification filter. */
    mipmapFilter?: GPUMipmapFilterMode;
    lodMinClamp?: number;
    lodMaxClamp?: number;
    /**
     * The binding type. Defaults to `comparison` with `compare`, otherwise `filtering`; use
     * `non-filtering` beside an `unfilterable-float` texture.
     */
    bindingType?: GPUSamplerBindingType;
}

/**
 * A wrapper class for GPU sampler bindings that implements the IBindable interface.
 * Handles creation and management of GPU samplers with specified filtering and address modes.
 */
class Sampler implements IBindable {
    /** The underlying GPU sampler resource */
    sampler?: GPUBindingResource | undefined;
    /** Type identifier for the binding */
    type: string = 'sampler';
    /** Indicates if the sampler has been initialized */
    initialized: boolean = false;
    /** Flag to indicate if the sampler needs to be updated */
    needsUpdate: boolean = false;
    /** Unique identifier for this sampler instance */
    uuid: string;
    /** Reference to the GPU device used for initialization */
    private gpuDevice?: GPUDevice;

    /**
     * Creates a new Sampler instance.
     * @param magFilter - The magnification filter mode to use when sampling the texture
     * @param minFilter - The minification filter mode to use when sampling the texture
     * @param repeatMode - The address mode determining how texture coordinates outside [0, 1] are handled, on all three axes
     * @param maxAnisotropy - Anisotropic filtering clamp (needs linear filters)
     * @param options - Comparison, mip filter, LOD clamps and binding type
     */
    constructor(
        private magFilter: GPUFilterMode,
        private minFilter: GPUFilterMode,
        private repeatMode: GPUAddressMode = 'repeat',
        private maxAnisotropy: number = 1,
        private options: SamplerOptions = {}
    ) {
        this.uuid = crypto.randomUUID();
    }

    /**
     * Updates the sampler if needed.
     * Reinitializes the sampler if the needsUpdate flag is set and the sampler was previously initialized.
     */
    public async update(): Promise<void> {
        if (this.needsUpdate) {
            if (this.initialized) {
                this.initialize(this.gpuDevice!);
                this.needsUpdate = false;
            }
        }
    }

    /**
     * Initializes the GPU sampler with the specified parameters.
     * @param gpuDevice - The GPU device to create the sampler on
     */
    public initialize(gpuDevice: GPUDevice): void {
        this.gpuDevice = gpuDevice;
        const sampler = gpuDevice.createSampler({
            magFilter: this.magFilter,
            minFilter: this.minFilter,
            mipmapFilter: this.options.mipmapFilter ?? this.minFilter,
            addressModeU: this.repeatMode,
            addressModeV: this.repeatMode,
            addressModeW: this.repeatMode,
            maxAnisotropy: this.maxAnisotropy > 1 ? this.maxAnisotropy : undefined,
            compare: this.options.compare,
            lodMinClamp: this.options.lodMinClamp,
            lodMaxClamp: this.options.lodMaxClamp,
        });

        this.sampler = sampler;
        this.initialized = true;
    }

    /**
     * Gets the underlying GPU sampler resource.
     * @returns The GPU binding resource for this sampler
     * @throws Will throw an error if accessed before initialization
     */
    get resource(): GPUBindingResource {
        return this.sampler!;
    }

    public getBindingLayout(_gpuDevice: GPUDevice, requested?: BindingLayout): BindingLayout {
        if (requested) return requested;
        const type = this.options.bindingType ?? (this.options.compare ? 'comparison' : 'filtering');
        return { sampler: { type } };
    }
}

export { Sampler };
