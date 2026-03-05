/**
 * GBuffer — a set of off-screen render targets used by the post-processing pipeline.
 *
 * Textures
 * --------
 * colorTexture      rgba16float  RENDER_ATTACHMENT | TEXTURE_BINDING | STORAGE_BINDING
 *                   Scene colour — resolve target for MSAA, or direct render target
 *                   when msaaSampleCount === 1.
 *
 * depthTexture      depth32float RENDER_ATTACHMENT | TEXTURE_BINDING
 *                   Resolved (non-MSAA) scene depth.  Compute shaders read it via
 *                   texture_depth_2d / textureLoad.
 *
 * colorMSAATexture  rgba16float  RENDER_ATTACHMENT  (only when msaaSampleCount > 1)
 *                   Multi-sample colour render target; resolved into colorTexture at
 *                   the end of each GBuffer render pass.
 *
 * depthMSAATexture  depth32float RENDER_ATTACHMENT | TEXTURE_BINDING
 *                   (only when msaaSampleCount > 1)
 *                   Multi-sample depth render target; copied into depthTexture via a
 *                   depth-copy render pass so compute shaders can read non-MSAA depth.
 *
 * outputTexture     rgba16float  TEXTURE_BINDING | STORAGE_BINDING
 * pingPongTexture   rgba16float  TEXTURE_BINDING | STORAGE_BINDING
 *                   Ping-pong pair used by the effect chain.  Each effect reads from
 *                   one and writes to the other.  The final result is blitted to the
 *                   canvas.
 */
class GBuffer {
    public colorTexture!: GPUTexture;
    public depthTexture!: GPUTexture;
    public emissiveTexture!: GPUTexture;
    public colorMSAATexture: GPUTexture | null = null;
    public depthMSAATexture: GPUTexture | null = null;
    public emissiveMSAATexture: GPUTexture | null = null;
    public outputTexture!: GPUTexture;
    public pingPongTexture!: GPUTexture;
    public width: number;
    public height: number;
    public readonly msaaSampleCount: number;

    constructor(
        private device: GPUDevice,
        width: number,
        height: number,
        msaaSampleCount: number = 1
    ) {
        this.width = width;
        this.height = height;
        this.msaaSampleCount = msaaSampleCount;
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

        // colorTexture is the resolve target (MSAA path) or direct render target (non-MSAA).
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

        this.emissiveTexture = this.device.createTexture({
            label: 'GBuffer/Emissive',
            size: [width, height],
            format: 'rgba16float',
            usage: colorUsage,
        });

        // depthTexture: non-MSAA depth for compute-shader reads.
        // Used as RENDER_ATTACHMENT by the depth-copy pass.
        this.depthTexture = this.device.createTexture({
            label: 'GBuffer/Depth',
            size: [width, height],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        if (this.msaaSampleCount > 1) {
            // MSAA colour — render target only; resolved into colorTexture each frame.
            this.colorMSAATexture = this.device.createTexture({
                label: 'GBuffer/ColorMSAA',
                size: [width, height],
                format: 'rgba16float',
                sampleCount: this.msaaSampleCount,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });

            this.emissiveMSAATexture = this.device.createTexture({
                label: 'GBuffer/EmissiveMSAA',
                size: [width, height],
                format: 'rgba16float',
                sampleCount: this.msaaSampleCount,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });

            // MSAA depth — also bound as a texture so the depth-copy shader can read it
            // via texture_depth_multisampled_2d / textureLoad.
            this.depthMSAATexture = this.device.createTexture({
                label: 'GBuffer/DepthMSAA',
                size: [width, height],
                format: 'depth32float',
                sampleCount: this.msaaSampleCount,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
        }

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
        this.emissiveTexture?.destroy();
        this.depthTexture?.destroy();
        this.colorMSAATexture?.destroy();
        this.colorMSAATexture = null;
        this.emissiveMSAATexture?.destroy();
        this.emissiveMSAATexture = null;
        this.depthMSAATexture?.destroy();
        this.depthMSAATexture = null;
        this.outputTexture?.destroy();
        this.pingPongTexture?.destroy();
    }
}

export { GBuffer };
