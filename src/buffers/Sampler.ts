import { IBindable } from "./IBindable";

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
     * @param repeatMode - The address mode determining how texture coordinates outside [0, 1] are handled
     */
    constructor(
        private magFilter: GPUFilterMode,
        private minFilter: GPUFilterMode,
        private repeatMode: GPUAddressMode = 'repeat'
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
            addressModeU: this.repeatMode,
            addressModeV: this.repeatMode,
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
}

export { Sampler };
