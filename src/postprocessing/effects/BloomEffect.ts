import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface BloomOptions {
    /** Luminance cutoff for bloom. Default 1.0 */
    threshold?: number;
    /** Soft threshold transition width. Default 0.1 */
    knee?: number;
    /** Final bloom strength. Default 0.8 */
    intensity?: number;
    /** Blur spread multiplier (controls upsample blend). Default 1.0 */
    radius?: number;
    /** When true, bloom reads from the emissive MRT texture instead of scene color. Default true. */
    useEmissive?: boolean;
}

/**
 * UE-style Progressive Downsample/Upsample Bloom
 * ================================================
 *
 * Multi-level mip chain bloom:
 *  1. Downsample chain (6 levels: full → 1/2 → … → 1/64)
 *     Level 0 applies brightness threshold. All levels use 13-tap box filter.
 *  2. Upsample chain (5 levels back up)
 *     Each level tent-filters the smaller mip, adds to current level content.
 *  3. Composite: blend upsampled bloom onto original scene.
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
    private _currentUseEmissive: boolean = true;

    static readonly MIP_COUNT = 6;

    threshold: number;
    knee: number;
    intensity: number;
    radius: number;
    /** When true, bloom reads from the emissive MRT texture. When false, reads scene color. */
    useEmissive: boolean;

    constructor(options: BloomOptions = {}) {
        super();
        this.threshold    = options.threshold ?? 1.0;
        this.knee         = options.knee      ?? 0.1;
        this.intensity    = options.intensity ?? 0.8;
        this.radius       = options.radius    ?? 1.0;
        this.useEmissive  = options.useEmissive ?? true;
    }

    // ========================================================================
    // WGSL Shaders
    // ========================================================================

    private static readonly _DOWNSAMPLE_SHADER = /* wgsl */`
        struct BloomParams {
            threshold  : f32,
            knee       : f32,
            intensity  : f32,
            radius     : f32,
            srcWidth   : f32,
            srcHeight  : f32,
            level      : u32,
            _pad       : u32,
        }

        @group(0) @binding(0) var srcTex  : texture_2d<f32>;
        @group(0) @binding(1) var dstTex  : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(2) var<uniform> params : BloomParams;

        fn luminance(c: vec3f) -> f32 {
            return dot(c, vec3f(0.2126, 0.7152, 0.0722));
        }

        fn softThreshold(color: vec3f, t: f32, k: f32) -> vec3f {
            let lum = luminance(color);
            let soft = lum - t + k;
            let soft2 = clamp(soft, 0.0, 2.0 * k);
            let contribution = soft2 * soft2 / (4.0 * k + 0.0001);
            let w = max(contribution, lum - t) / max(lum, 0.0001);
            return color * max(w, 0.0);
        }

        fn safeLoad(coord: vec2i, dims: vec2i) -> vec3f {
            let c = clamp(coord, vec2i(0), dims - vec2i(1));
            return textureLoad(srcTex, c, 0).rgb;
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let dstW = u32(params.srcWidth);
            let dstH = u32(params.srcHeight);
            if (gid.x >= dstW || gid.y >= dstH) { return; }

            let srcDims = vec2i(i32(dstW) * 2, i32(dstH) * 2);
            let base = vec2i(gid.xy) * 2;

            // 13-tap box filter (Jimenez 2014, "Next Generation Post Processing in CoD")
            let a = safeLoad(base + vec2i(-1, -1), srcDims);
            let b = safeLoad(base + vec2i( 0, -1), srcDims);
            let c = safeLoad(base + vec2i( 1, -1), srcDims);
            let d = safeLoad(base + vec2i(-1,  0), srcDims);
            let e = safeLoad(base + vec2i( 0,  0), srcDims);
            let f = safeLoad(base + vec2i( 1,  0), srcDims);
            let g = safeLoad(base + vec2i(-1,  1), srcDims);
            let h = safeLoad(base + vec2i( 0,  1), srcDims);
            let ii = safeLoad(base + vec2i( 1,  1), srcDims);

            // Weighted combination
            var color = e * 0.25;
            color += (b + d + f + h) * 0.125;
            color += (a + c + g + ii) * 0.0625;

            // Level 0: apply brightness threshold
            if (params.level == 0u) {
                color = softThreshold(color, params.threshold, params.knee);
            }

            textureStore(dstTex, gid.xy, vec4f(color, 1.0));
        }
    `;

    private static readonly _UPSAMPLE_SHADER = /* wgsl */`
        struct BloomParams {
            threshold  : f32,
            knee       : f32,
            intensity  : f32,
            radius     : f32,
            srcWidth   : f32,
            srcHeight  : f32,
            level      : u32,
            _pad       : u32,
        }

        @group(0) @binding(0) var smallerMip : texture_2d<f32>;
        @group(0) @binding(1) var mipSampler : sampler;
        @group(0) @binding(2) var currentMip : texture_2d<f32>;
        @group(0) @binding(3) var dstTex     : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(4) var<uniform> params : BloomParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let dstW = u32(params.srcWidth);
            let dstH = u32(params.srcHeight);
            if (gid.x >= dstW || gid.y >= dstH) { return; }

            let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(dstW), f32(dstH));
            let texelSize = 1.0 / vec2f(f32(dstW), f32(dstH));

            // 9-tap tent filter on the smaller mip (bilinear via sampler)
            var upsampled = vec3f(0.0);
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f(-1.0, -1.0) * texelSize, 0.0).rgb;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f( 0.0, -1.0) * texelSize, 0.0).rgb * 2.0;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f( 1.0, -1.0) * texelSize, 0.0).rgb;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f(-1.0,  0.0) * texelSize, 0.0).rgb * 2.0;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv,                                  0.0).rgb * 4.0;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f( 1.0,  0.0) * texelSize, 0.0).rgb * 2.0;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f(-1.0,  1.0) * texelSize, 0.0).rgb;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f( 0.0,  1.0) * texelSize, 0.0).rgb * 2.0;
            upsampled += textureSampleLevel(smallerMip, mipSampler, uv + vec2f( 1.0,  1.0) * texelSize, 0.0).rgb;
            upsampled /= 16.0;

            // Additive blend with current level's downsample content
            let current = textureLoad(currentMip, gid.xy, 0).rgb;
            let result = current + upsampled * params.radius;

            textureStore(dstTex, gid.xy, vec4f(result, 1.0));
        }
    `;

    private static readonly _COMPOSITE_SHADER = /* wgsl */`
        struct BloomParams {
            threshold  : f32,
            knee       : f32,
            intensity  : f32,
            radius     : f32,
            srcWidth   : f32,
            srcHeight  : f32,
            level      : u32,
            _pad       : u32,
        }

        @group(0) @binding(0) var inputTex    : texture_2d<f32>;
        @group(0) @binding(1) var bloomTex    : texture_2d<f32>;
        @group(0) @binding(2) var bloomSampler : sampler;
        @group(0) @binding(3) var outputTex   : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(4) var<uniform> params : BloomParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let w = u32(params.srcWidth);
            let h = u32(params.srcHeight);
            if (gid.x >= w || gid.y >= h) { return; }

            let original = textureLoad(inputTex, gid.xy, 0);
            let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(w), f32(h));
            let bloom = textureSampleLevel(bloomTex, bloomSampler, uv, 0.0).rgb;
            let result = original.rgb + bloom * params.intensity;
            textureStore(outputTex, gid.xy, vec4f(result, original.a));
        }
    `;

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

        this._downsamplePipeline = this._createPipeline(device, 'Bloom/Downsample', BloomEffect._DOWNSAMPLE_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        this._upsamplePipeline = this._createPipeline(device, 'Bloom/Upsample', BloomEffect._UPSAMPLE_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        this._compositePipeline = this._createPipeline(device, 'Bloom/Composite', BloomEffect._COMPOSITE_SHADER, [
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

        const effectiveEmissive = this.useEmissive ? emissive : undefined;
        if (input !== this._currentInput || output !== this._currentOutput
            || effectiveEmissive !== this._currentEmissive
            || this.useEmissive !== this._currentUseEmissive) {
            this._buildBindGroups(input, output, effectiveEmissive);
            this._currentUseEmissive = this.useEmissive;
        }

        const wg = (t: number) => Math.ceil(t / 8);

        // --- Downsample chain ---
        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            const w = Math.max(1, Math.ceil(width / (1 << (i + 1))));
            const h = Math.max(1, Math.ceil(height / (1 << (i + 1))));

            this._writeParamsTo(this._downsampleParamBuffers[i], w, h, i);

            const pass = commandEncoder.beginComputePass({ label: `Bloom/Down/${i}` });
            pass.setPipeline(this._downsamplePipeline!);
            pass.setBindGroup(0, this._downsampleBindGroups[i]!);
            pass.dispatchWorkgroups(wg(w), wg(h));
            pass.end();
        }

        // --- Upsample chain ---
        for (let i = BloomEffect.MIP_COUNT - 2; i >= 0; i--) {
            const w = Math.max(1, Math.ceil(width / (1 << (i + 1))));
            const h = Math.max(1, Math.ceil(height / (1 << (i + 1))));

            this._writeParamsTo(this._upsampleParamBuffers[i], w, h, i);

            const pass = commandEncoder.beginComputePass({ label: `Bloom/Up/${i}` });
            pass.setPipeline(this._upsamplePipeline!);
            pass.setBindGroup(0, this._upsampleBindGroups[i]!);
            pass.dispatchWorkgroups(wg(w), wg(h));
            pass.end();
        }

        // --- Composite ---
        this._writeParamsTo(this._compositeParamBuffer!, width, height, 0);

        const pass = commandEncoder.beginComputePass({ label: 'Bloom/Composite' });
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
        u[7] = 0;
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

        for (let i = 0; i < BloomEffect.MIP_COUNT; i++) {
            const w = Math.max(1, Math.ceil(width / (1 << (i + 1))));
            const h = Math.max(1, Math.ceil(height / (1 << (i + 1))));

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
