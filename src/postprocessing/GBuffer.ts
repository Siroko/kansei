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
 *
 * backgroundTexture rgba16float  RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_DST
 *                   A copy of colorTexture taken AFTER opaque objects are drawn but
 *                   BEFORE transmissive objects. Transmission post-processing effects
 *                   sample it to do screen-space refraction (e.g. the fluid surface).
 *
 * velocityTexture   rg16float    RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC
 *                   Screen-space motion (uv, current minus previous) of the opaque materials
 *                   with `outputsVelocity`, drawn by the renderer's velocity pass after the
 *                   GBuffer pass; `NO_VELOCITY` elsewhere (TAA and motion blur reproject those
 *                   pixels by depth). Single-sample whatever `msaaSampleCount` is: the pass
 *                   depth-tests against `depthTexture`.
 */
class GBuffer {
    /** Format of the scene depth (Rust `GBuffer::DEPTH_FORMAT`). */
    static readonly DEPTH_FORMAT: GPUTextureFormat = 'depth32float';
    /** Format of screen-space motion, written by materials with `outputsVelocity` in the velocity pass (Rust `GBuffer::VELOCITY_FORMAT`). */
    static readonly VELOCITY_FORMAT: GPUTextureFormat = 'rg16float';
    /** The colour target index of velocity in the velocity pass, after the four MRT targets: the shader writes it at @location(4) (Rust `GBuffer::VELOCITY_TARGET`). */
    static readonly VELOCITY_TARGET = 4;
    /** What the velocity texture is cleared to: no motion vector drawn here (an impossible velocity, far outside the screen; Rust `GBuffer::NO_VELOCITY`). */
    static readonly NO_VELOCITY = 1.0e4;

    public colorTexture!: GPUTexture;
    public depthTexture!: GPUTexture;
    public emissiveTexture!: GPUTexture;
    public colorMSAATexture: GPUTexture | null = null;
    public depthMSAATexture: GPUTexture | null = null;
    public emissiveMSAATexture: GPUTexture | null = null;
    public normalTexture!: GPUTexture;
    public albedoTexture!: GPUTexture;
    public normalMSAATexture: GPUTexture | null = null;
    public albedoMSAATexture: GPUTexture | null = null;
    public backgroundTexture!: GPUTexture;
    public velocityTexture!: GPUTexture;
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
        // COPY_SRC is required so the transmissive-split path can snapshot the opaque
        // result into backgroundTexture between render passes.
        const colorUsage =
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC;

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

        this.normalTexture = this.device.createTexture({
            label: 'GBuffer/Normal',
            size: [width, height],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });

        this.albedoTexture = this.device.createTexture({
            label: 'GBuffer/Albedo',
            size: [width, height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
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

            this.normalMSAATexture = this.device.createTexture({
                label: 'GBuffer/NormalMSAA',
                size: [width, height],
                format: 'rgba16float',
                sampleCount: this.msaaSampleCount,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });

            this.albedoMSAATexture = this.device.createTexture({
                label: 'GBuffer/AlbedoMSAA',
                size: [width, height],
                format: 'rgba8unorm',
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

        // Background capture texture — filled via copyTextureToTexture after the
        // opaque pass, consumed by transmission effects for screen-space refraction.
        this.backgroundTexture = this.device.createTexture({
            label: 'GBuffer/Background',
            size: [width, height],
            format: 'rgba16float',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST,
        });

        this.velocityTexture = this.device.createTexture({
            label: 'GBuffer/Velocity',
            size: [width, height],
            format: GBuffer.VELOCITY_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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
        this.emissiveTexture?.destroy();
        this.depthTexture?.destroy();
        this.colorMSAATexture?.destroy();
        this.colorMSAATexture = null;
        this.emissiveMSAATexture?.destroy();
        this.emissiveMSAATexture = null;
        this.normalTexture?.destroy();
        this.albedoTexture?.destroy();
        this.normalMSAATexture?.destroy();
        this.normalMSAATexture = null;
        this.albedoMSAATexture?.destroy();
        this.albedoMSAATexture = null;
        this.depthMSAATexture?.destroy();
        this.depthMSAATexture = null;
        this.backgroundTexture?.destroy();
        this.velocityTexture?.destroy();
        this.outputTexture?.destroy();
        this.pingPongTexture?.destroy();
    }
}

export { GBuffer };
