import { IBindable } from "./IBindable";

/**
 * Represents a video texture that can be bound to a GPU device.
 * Implements the IBindable interface.
 */
class VideoTexture implements IBindable {
    type: string = 'external-texture';
    private gpuDevice?: GPUDevice;
    public initialized: boolean = false;
    public needsUpdate: boolean = false;

    public uuid: string;

    /**
     * Constructs a new VideoTexture instance.
     * @param videoElement - The HTML video element to be used as the texture source.
     */
    constructor(
        private videoElement: HTMLVideoElement
    ) {
        this.uuid = crypto.randomUUID();
    }

    /**
     * Updates the video texture. This method is asynchronous.
     * @returns A promise that resolves when the update is complete.
     */
    update(): GPUExternalTexture {
        const res = this.gpuDevice!.importExternalTexture({ source: this.videoElement! });
        res.label = 'VideoTexture ' + this.uuid;
        return res;
    }

    /**
     * Gets the GPU binding resource for the video texture.
     * @returns The GPUBindingResource associated with the video texture.
     */
    get resource(): GPUBindingResource {
        const res = this.gpuDevice!.importExternalTexture({ source: this.videoElement! });
        res.label = 'VideoTexture ' + this.uuid;
        return res;
    }

    /**
     * Initializes the video texture with a GPU device.
     * @param gpuDevice - The GPU device to bind the texture to.
     */
    initialize(gpuDevice: GPUDevice) {
        this.gpuDevice = gpuDevice;
        this.initialized = true;
    }

}

export { VideoTexture };
