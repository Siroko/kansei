import { IBindable } from "./IBindable";

/**
 * Represents a WebGPU texture that can be bound to a shader.
 * Handles the creation and management of GPUTexture objects from ImageBitmap sources.
 * @implements {IBindable}
 */
class Texture implements IBindable {
    /** Flag indicating if the texture has been initialized */
    public initialized: boolean = false;
    /** Magnification filter mode for the texture */
    public magFilter: GPUFilterMode = 'linear';
    /** Minification filter mode for the texture */
    public minFilter: GPUFilterMode = 'linear';
    /** Unique identifier for the texture */
    public uuid: string;
    /** Type identifier for the texture */
    type: string = 'texture';
    /** Flag indicating if the texture needs to be updated */
    public needsUpdate: boolean = false;
    /** The underlying WebGPU texture object */
    private texture?: GPUTexture;

    /**
     * Creates a new Texture instance.
     * @param {ImageBitmap} imageBitmap - The source image bitmap to create the texture from
     */
    constructor(
        private imageBitmap: ImageBitmap
    ) {
        this.uuid = crypto.randomUUID();
    }

    /**
     * Updates the texture. Currently a placeholder for future implementation.
     * @returns {Promise<void>}
     */
    public async update(): Promise<void> {
    }

    /**
     * Initializes the texture with the given WebGPU device.
     * @param {GPUDevice} gpuDevice - The WebGPU device to create the texture with
     */
    public initialize(gpuDevice: GPUDevice) {
        this.texture = this.webGPUTextureFromImageBitmapOrCanvas(gpuDevice, this.imageBitmap);
        this.initialized = true;
    }

    /**
     * Gets the binding resource for this texture.
     * @returns {GPUBindingResource} The texture view that can be used for binding
     */
    get resource(): GPUBindingResource {
        return this.texture!.createView();
    }

    /**
     * Creates a WebGPU texture from an ImageBitmap or Canvas source.
     * @param {GPUDevice} gpuDevice - The WebGPU device to create the texture with
     * @param {ImageBitmap | HTMLCanvasElement} source - The source image or canvas
     * @returns {GPUTexture} The created WebGPU texture
     * @private
     */
    private webGPUTextureFromImageBitmapOrCanvas(gpuDevice: GPUDevice, source: ImageBitmap | HTMLCanvasElement) {
        const textureDescriptor: GPUTextureDescriptor = {
            size: { width: source.width, height: source.height },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        };
        const texture = gpuDevice.createTexture(textureDescriptor);

        gpuDevice.queue.copyExternalImageToTexture({ source }, { texture }, textureDescriptor.size);

        return texture;
    }
}

export { Texture }
