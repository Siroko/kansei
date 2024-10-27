/**
 * Interface for GPU-bindable resources that can be used in a WebGPU binding layout
 */
export interface IBindable {
    /** Optional type identifier for the bindable resource */
    type?: string;

    /** Indicates whether the resource has been initialized */
    initialized: boolean;

    /** Flag indicating if the resource needs to be updated on the GPU */
    needsUpdate: boolean;

    /**
     * Returns the underlying GPU binding resource
     * @returns {GPUBindingResource} The WebGPU binding resource
     */
    get resource(): GPUBindingResource;

    /**
     * Initializes the bindable resource on the GPU
     * @param gpuDevice The GPU device to initialize the resource on
     */
    initialize(gpuDevice: GPUDevice): void;

    /**
     * Updates the resource data on the GPU if needed
     * @param gpuDevice The GPU device to update the resource on
     */
    update(gpuDevice: GPUDevice): void;
}
