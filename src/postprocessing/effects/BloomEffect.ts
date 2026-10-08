import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { gpuPass } from '../../profiling/Profiler';
import { BLOOM_COMPOSITE_WGSL, BLOOM_DOWNSAMPLE_WGSL, BLOOM_UPSAMPLE_WGSL } from '../../materials/shaders/SharedWGSL';

export interface BloomOptions {
    /**
     * Luminance, in exposed units (see `BloomEffect.exposure`), above which light blooms. Zero or
     * below disables the threshold: physically based bloom, where every light scatters a little
     * and `intensity` is the scattered fraction (energy-conserving; try 0.03-0.1). Default 1.0
     */
    threshold?: number;
    /** Soft threshold transition width. Default 0.1 */
    knee?: number;
    /** With a threshold, the gain of the bloom added on top of the scene. Default 0.8 */
    intensity?: number;
    /** Spread of the upsample tent filter, in texels of the smaller level. Default 1.0 */
    radius?: number;
    /**
     * When true, bloom reads from the emissive MRT texture instead of scene color. Physically
     * based bloom (`threshold <= 0`) always reads scene color, since it mixes the blur back in
     * place of the scene. Default true.
     */
    useEmissive?: boolean;
    /** See `BloomEffect.exposure`. Default 1.0 */
    exposure?: number;
}

/**
 * UE-style Progressive Downsample/Upsample Bloom
 * ================================================
 *
 * Multi-level mip chain bloom, on scene-linear HDR (it belongs before the tonemapper):
 *  1. Downsample chain (6 levels: full → 1/2 → … → 1/64), 3x3 tent filter.
 *     Level 0 weights its taps by 1 / (1 + luma) (Karis average) against fireflies and
 *     applies the brightness threshold.
 *  2. Upsample chain (5 levels back up)
 *     Each level tent-filters the smaller mip, adds to current level content.
 *  3. Composite: add the bloom onto the scene, or with no threshold mix in the average of the
 *     levels (physically based).
 *
 * The shaders are the Rust engine's (`SharedWGSL`); like Rust, the output's alpha is 1.
 */
class BloomEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;

    // Per-dispatch param buffers — each dispatch gets its own uniform buffer
    // so queue.writeBuffer calls don't overwrite each other before execution.
    private _downsampleParamBuffers: GPUBuffer[] = [];
    private _upsampleParamBuffers: GPUBuffer[] = [];
    private _compositeParamBuffer: GPUBuffer | null = null;

    private _downsamplePipeline: GPUComputePipeline | null = null;
    private _upsamplePipeline: GPUComputePipeline | null = null;
    private _compositePipeline: GPUComputePipeline | null = null;

    // Downsample mip chain (read/write during downsample, read-only during upsample)
    private _mipChain: GPUTexture[] = [];
    // Separate upsample output textures (avoids same-texture read+write per dispatch)
    private _upsampleMips: GPUTexture[] = [];
    private _sampler: GPUSampler | null = null;

    private _downsampleBindGroups: (GPUBindGroup | null)[] = [];
    private _upsampleBindGroups: (GPUBindGroup | null)[] = [];
    private _compositeBindGroup: GPUBindGroup | null = null;

    private _currentInput: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;
    private _currentEmissive: GPUTexture | null = null;

    static readonly MIP_COUNT = 6;

    threshold: number;
    knee: number;
    intensity: number;
    radius: number;
    /** When true, bloom reads from the emissive MRT texture. When false, reads scene color. */
    useEmissive: boolean;
    /**
     * Scene multiplier the threshold and firefly filter see, so they work in the tonemapper's
     * exposed units with physical light values: set it to `ToneMapEffect.totalExposure()`.
     * 1 (the default) compares raw scene values.
     */
    exposure: number;

    constructor(options: BloomOptions = {}) {
        super();
        this.threshold    = options.threshold ?? 1.0;
        this.knee         = options.knee      ?? 0.1;
        this.intensity    = options.intensity ?? 0.8;
        this.radius       = options.radius    ?? 1.0;
        this.useEmissive  = options.useEmissive ?? true;
        this.exposure     = options.exposure  ?? 1.0;
    }

    // ========================================================================
    // PostProcessingEffect interface
    // ========================================================================

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        // Create per-dispatch param buffers (6 downsample + 5 upsample + 1 composite)
        this._downsampleParamBuffers = [];
        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            this._downsampleParamBuffers.push(device.createBuffer({
                label: `Bloom/Params/Down${i}`,
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }
        this._upsampleParamBuffers = [];
        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            this._upsampleParamBuffers.push(device.createBuffer({
                label: `Bloom/Params/Up${i}`,
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }
        this._compositeParamBuffer = device.createBuffer({
            label: 'Bloom/Params/Composite',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._sampler = device.createSampler({
            label: 'Bloom/Sampler',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        this._downsamplePipeline = this._createPipeline(device, 'Bloom/Downsample', BLOOM_DOWNSAMPLE_WGSL, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        this._upsamplePipeline = this._createPipeline(device, 'Bloom/Upsample', BLOOM_UPSAMPLE_WGSL, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        this._compositePipeline = this._createPipeline(device, 'Bloom/Composite', BLOOM_COMPOSITE_WGSL, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        this._createMipTextures(gbuffer.width, gbuffer.height);
        this._buildBindGroups(gbuffer.colorTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        _depth: GPUTexture,
        output: GPUTexture,
        _camera: Camera,
        width: number,
        height: number,
        emissive?: GPUTexture,
    ): void {
        if (!this._downsamplePipeline) return;

        const effectiveEmissive = this.useEmissive && this.threshold > 0 ? emissive : undefined;
        if (input !== this._currentInput || output !== this._currentOutput
            || effectiveEmissive !== this._currentEmissive) {
            this._buildBindGroups(input, output, effectiveEmissive);
        }

        const wg = (t: number) => Math.ceil(t / 8);

        // --- Downsample chain ---
        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            const { width: w, height: h } = this._mipChain[i];
            const src = i === 0 ? { width, height } : this._mipChain[i - 1];

            this._writeParamsTo(this._downsampleParamBuffers[i], src.width, src.height, i);

            const pass = commandEncoder.beginComputePass({ label: `Bloom/Down/${i}`, timestampWrites: gpuPass('Bloom/Downsample') });
            pass.setPipeline(this._downsamplePipeline!);
            pass.setBindGroup(0, this._downsampleBindGroups[i]!);
            pass.dispatchWorkgroups(wg(w), wg(h));
            pass.end();
        }

        // --- Upsample chain ---
        for (let i = BloomEffect.MIP_COUNT - 2; i >= 0; i--) {
            const { width: w, height: h } = this._mipChain[i];
            const smaller = this._mipChain[i + 1];

            this._writeParamsTo(this._upsampleParamBuffers[i], smaller.width, smaller.height, i);

            const pass = commandEncoder.beginComputePass({ label: `Bloom/Up/${i}`, timestampWrites: gpuPass('Bloom/Upsample') });
            pass.setPipeline(this._upsamplePipeline!);
            pass.setBindGroup(0, this._upsampleBindGroups[i]!);
            pass.dispatchWorkgroups(wg(w), wg(h));
            pass.end();
        }

        // --- Composite ---
        // the composite's level is the number of levels summed into the bloom texture
        this._writeParamsTo(this._compositeParamBuffer!, width, height, BloomEffect.MIP_COUNT);

        const pass = commandEncoder.beginComputePass({ label: 'Bloom/Composite', timestampWrites: gpuPass('Bloom/Composite') });
        pass.setPipeline(this._compositePipeline!);
        pass.setBindGroup(0, this._compositeBindGroup!);
        pass.dispatchWorkgroups(wg(width), wg(height));
        pass.end();
    }

    resize(w: number, h: number, _gbuffer: GBuffer): void {
        this._destroyMipTextures();
        this._createMipTextures(w, h);
        this._compositeBindGroup = null;
        this._currentInput = null;
        this._currentOutput = null;
    }

    destroy(): void {
        for (const buf of this._downsampleParamBuffers) buf.destroy();
        for (const buf of this._upsampleParamBuffers) buf.destroy();
        this._compositeParamBuffer?.destroy();
        this._downsampleParamBuffers = [];
        this._upsampleParamBuffers = [];
        this._compositeParamBuffer = null;
        this._destroyMipTextures();
        this._downsamplePipeline = null;
        this._upsamplePipeline = null;
        this._compositePipeline = null;
        this._compositeBindGroup = null;
        this._sampler = null;
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    private _writeParamsTo(buffer: GPUBuffer, w: number, h: number, level: number): void {
        const buf = new ArrayBuffer(32);
        const f = new Float32Array(buf);
        const u = new Uint32Array(buf);
        f[0] = this.threshold;
        f[1] = this.knee;
        f[2] = this.intensity;
        f[3] = this.radius;
        f[4] = w;
        f[5] = h;
        u[6] = level;
        f[7] = this.exposure;
        this._device!.queue.writeBuffer(buffer, 0, buf);
    }

    private _createPipeline(
        device: GPUDevice,
        label: string,
        shaderCode: string,
        entries: GPUBindGroupLayoutEntry[],
    ): GPUComputePipeline {
        const module = device.createShaderModule({ label: `${label}/Module`, code: shaderCode });
        const bgl = device.createBindGroupLayout({ label: `${label}/BGL`, entries });
        return device.createComputePipeline({
            label: `${label}/Pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });
    }

    private _createMipTextures(width: number, height: number): void {
        const device = this._device!;
        const texUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

        this._mipChain = [];
        this._upsampleMips = [];

        let w = width;
        let h = height;
        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            w = Math.max(1, w >> 1);
            h = Math.max(1, h >> 1);

            this._mipChain.push(device.createTexture({
                label: `Bloom/Mip${i}`,
                size: [w, h],
                format: 'rgba16float',
                usage: texUsage,
            }));

            this._upsampleMips.push(device.createTexture({
                label: `Bloom/UpMip${i}`,
                size: [w, h],
                format: 'rgba16float',
                usage: texUsage,
            }));
        }
    }

    private _destroyMipTextures(): void {
        for (const tex of this._mipChain) tex.destroy();
        for (const tex of this._upsampleMips) tex.destroy();
        this._mipChain = [];
        this._upsampleMips = [];
        this._downsampleBindGroups = [];
        this._upsampleBindGroups = [];
    }

    private _buildBindGroups(input: GPUTexture, output: GPUTexture, emissive?: GPUTexture): void {
        const device = this._device!;

        // Downsample: source → mipChain[i]
        // Level 0 reads from the emissive texture (if available) so bloom
        // is driven purely by emissive contribution, not full scene color.
        this._downsampleBindGroups = [];
        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            const src = i === 0 ? (emissive ?? input) : this._mipChain[i - 1];
            this._downsampleBindGroups.push(device.createBindGroup({
                label: `Bloom/Down/BG${i}`,
                layout: this._downsamplePipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: src.createView() },
                    { binding: 1, resource: this._mipChain[i].createView() },
                    { binding: 2, resource: { buffer: this._downsampleParamBuffers[i] } },
                ],
            }));
        }

        // Upsample: smallerSrc + mipChain[i] → upsampleMips[i]
        // smallerSrc for first step = mipChain[last], then upsampleMips[i+1]
        this._upsampleBindGroups = [];
        for (let i = BloomEffect.MIP_COUNT - 2; i >= 0; i--) {
            const smallerSrc = (i === BloomEffect.MIP_COUNT - 2)
                ? this._mipChain[BloomEffect.MIP_COUNT - 1]
                : this._upsampleMips[i + 1];

            this._upsampleBindGroups[i] = device.createBindGroup({
                label: `Bloom/Up/BG${i}`,
                layout: this._upsamplePipeline!.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: smallerSrc.createView() },
                    { binding: 1, resource: this._sampler! },
                    { binding: 2, resource: this._mipChain[i].createView() },
                    { binding: 3, resource: this._upsampleMips[i].createView() },
                    { binding: 4, resource: { buffer: this._upsampleParamBuffers[i] } },
                ],
            });
        }

        // Composite: input (scene color) + upsampleMips[0] (bloom) → output
        this._compositeBindGroup = device.createBindGroup({
            label: 'Bloom/Composite/BG',
            layout: this._compositePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: this._upsampleMips[0].createView() },
                { binding: 2, resource: this._sampler! },
                { binding: 3, resource: output.createView() },
                { binding: 4, resource: { buffer: this._compositeParamBuffer! } },
            ],
        });

        this._currentInput = input;
        this._currentOutput = output;
        this._currentEmissive = emissive ?? null;
    }
}

export { BloomEffect };
