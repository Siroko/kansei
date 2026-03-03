/**
 * GBuffer — a set of off-screen render targets used by the post-processing pipeline.
 *
 * Textures
 * --------
 * colorTexture   rgba16float  RENDER_ATTACHMENT | TEXTURE_BINDING | STORAGE_BINDING
 *                The scene is rendered into this texture at sampleCount=1 (no MSAA).
 *
 * depthTexture   depth32float RENDER_ATTACHMENT | TEXTURE_BINDING
 *                Scene depth.  Compute shaders read it via texture_depth_2d / textureLoad.
 *
 * outputTexture  rgba16float  TEXTURE_BINDING | STORAGE_BINDING
 * pingPongTexture rgba16float TEXTURE_BINDING | STORAGE_BINDING
 *                Ping-pong pair used by the effect chain.  Each effect reads from one
 *                and writes to the other.  The final result is blitted to the canvas.
 */
class GBuffer {
    public colorTexture!: GPUTexture;
    public depthTexture!: GPUTexture;
    public outputTexture!: GPUTexture;
    public pingPongTexture!: GPUTexture;
    public width: number;
    public height: number;

    constructor(private device: GPUDevice, width: number, height: number) {
        this.width = width;
        this.height = height;
        this._create();
    }

    /** Destroys all GPU textures and re-creates them at the new size. */
    public resize(width: number, height: number): void {
        this.width = width;
        this.height = height;
        this.destroy();
        this._create();
    }

    private _create(): void {
        const { width, height } = this;

        const colorUsage =
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING;

        this.colorTexture = this.device.createTexture({
            label: 'GBuffer/Color',
            size: [width, height],
            format: 'rgba16float',
            usage: colorUsage,
        });

        // depth32float supports TEXTURE_BINDING so compute shaders can read it.
        this.depthTexture = this.device.createTexture({
            label: 'GBuffer/Depth',
            size: [width, height],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        const effectUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

        this.outputTexture = this.device.createTexture({
            label: 'GBuffer/Output',
            size: [width, height],
            format: 'rgba16float',
            usage: effectUsage,
        });

        this.pingPongTexture = this.device.createTexture({
            label: 'GBuffer/PingPong',
            size: [width, height],
            format: 'rgba16float',
            usage: effectUsage,
        });
    }

    /** Destroys all GPU textures held by this GBuffer. */
    public destroy(): void {
        this.colorTexture?.destroy();
        this.depthTexture?.destroy();
        this.outputTexture?.destroy();
        this.pingPongTexture?.destroy();
    }
}

export { GBuffer };
