import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface DepthOfFieldOptions {
    /** World-space distance to the in-focus plane. Default 5.0 */
    focusDistance?: number;
    /** Half-extent of the in-focus range (world units). Default 2.0 */
    focusRange?: number;
    /** Maximum blur radius in pixels (maps to CoC = +-1). Default 14 */
    maxBlur?: number;
}

/**
 * Depth of Field -- Cinematic Multi-Pass Pipeline
 * ================================================
 *
 * 5-pass (6 dispatches) pipeline that blurs at half resolution:
 *
 *  Pass 1 -- CoC Computation (full-res, 1 dispatch)
 *      Computes signed Circle of Confusion from depth buffer.
 *
 *  Pass 2 -- Near-Field CoC Dilation (full-res, 2 dispatches)
 *      Separable horizontal + vertical max-filter that expands near-field CoC
 *      outward so in-focus/far pixels at depth edges gather from the foreground.
 *
 *  Pass 3 -- Downsample + Near/Far Separation (full->half, 1 dispatch)
 *      Downsamples to half-res and separates into pre-multiplied near-field
 *      and far-field layers.
 *
 *  Pass 4 -- Vogel Disk Blur (half-res, 1 dispatch)
 *      64-tap golden-ratio disk blur on both near and far layers.
 *
 *  Pass 5 -- Composite (full-res, 1 dispatch)
 *      Bilinear upscale of half-res results and alpha-composite onto sharp image.
 */
class DepthOfFieldEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _paramsBuffer: GPUBuffer | null = null;

    // Pipelines (6 dispatches across 5 passes)
    private _cocPipeline: GPUComputePipeline | null = null;
    private _dilateHPipeline: GPUComputePipeline | null = null;
    private _dilateVPipeline: GPUComputePipeline | null = null;
    private _downsamplePipeline: GPUComputePipeline | null = null;
    private _blurPipeline: GPUComputePipeline | null = null;
    private _compositePipeline: GPUComputePipeline | null = null;

    // Bind groups
    private _cocBindGroup: GPUBindGroup | null = null;
    private _dilateHBindGroup: GPUBindGroup | null = null;
    private _dilateVBindGroup: GPUBindGroup | null = null;
    private _downsampleBindGroup: GPUBindGroup | null = null;
    private _blurBindGroup: GPUBindGroup | null = null;
    private _compositeBindGroup: GPUBindGroup | null = null;

    // Internal textures
    private _cocTex: GPUTexture | null = null;
    private _cocDilTempTex: GPUTexture | null = null;
    private _nearHalfTex: GPUTexture | null = null;
    private _farHalfTex: GPUTexture | null = null;
    private _nearBlurTex: GPUTexture | null = null;
    private _farBlurTex: GPUTexture | null = null;

    // External texture tracking for bind group invalidation
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    private readonly _focusDistance: number;
    private readonly _focusRange: number;
    private readonly _maxBlur: number;

    constructor(options: DepthOfFieldOptions = {}) {
        super();
        this._focusDistance = options.focusDistance ?? 5.0;
        this._focusRange    = options.focusRange    ?? 2.0;
        this._maxBlur       = options.maxBlur       ?? 14;
    }

    // ========================================================================
    // WGSL Shaders
    // ========================================================================

    /** Shared DoFParams struct and linearDepth / computeCoC helpers. */
    private static readonly _COMMON = /* wgsl */`
        struct DoFParams {
            focusDistance : f32,
            focusRange   : f32,
            maxBlur      : f32,
            near         : f32,
            far          : f32,
            screenWidth  : f32,
            screenHeight : f32,
            _pad         : f32,
        }

        fn linearDepth(d: f32, n: f32, f: f32) -> f32 {
            return (n * f) / (f - d * (f - n));
        }

        fn computeCoC(d: f32, p: DoFParams) -> f32 {
            let ld = linearDepth(d, p.near, p.far);
            return clamp((ld - p.focusDistance) / p.focusRange, -1.0, 1.0) * p.maxBlur;
        }
    `;

    /** Pass 1: CoC computation (full-res). */
    private static readonly _COC_SHADER = /* wgsl */`
        ${DepthOfFieldEffect._COMMON}

        @group(0) @binding(0) var depthTex : texture_depth_2d;
        @group(0) @binding(1) var cocOut   : texture_storage_2d<r32float, write>;
        @group(0) @binding(2) var<uniform> params : DoFParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w = u32(params.screenWidth);
            let h = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let depth = textureLoad(depthTex, coord, 0);
            let coc = computeCoC(depth, params);
            textureStore(cocOut, coord, vec4f(coc, 0.0, 0.0, 0.0));
        }
    `;

    /** Pass 2a: Horizontal near-field CoC dilation (full-res). */
    private static readonly _DILATE_H_SHADER = /* wgsl */`
        ${DepthOfFieldEffect._COMMON}

        @group(0) @binding(0) var cocIn  : texture_2d<f32>;
        @group(0) @binding(1) var cocOut : texture_storage_2d<r32float, write>;
        @group(0) @binding(2) var<uniform> params : DoFParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w = u32(params.screenWidth);
            let h = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let ownCoc = textureLoad(cocIn, coord, 0).r;
            var maxR = max(0.0, -ownCoc); // own near-field radius (positive value)

            let radius = i32(params.maxBlur);
            let iCoord = vec2i(coord);

            for (var dx = -radius; dx <= radius; dx++) {
                let sx = clamp(iCoord.x + dx, 0, i32(w) - 1);
                let sc = vec2u(u32(sx), coord.y);
                let sCoc = textureLoad(cocIn, sc, 0).r;
                // Only dilate near-field (negative CoC)
                if (sCoc < 0.0) {
                    let sR = -sCoc; // positive radius
                    let dist = f32(abs(dx));
                    // Near-field CoC is large enough to reach this pixel
                    if (sR >= dist) {
                        maxR = max(maxR, sR);
                    }
                }
            }

            // Store as negative (near-field convention) if dilated, else keep original
            // We store the dilated gather radius: max of own |CoC| and qualifying neighbors
            var result = ownCoc;
            if (maxR > max(0.0, -ownCoc)) {
                // A near neighbor's CoC reaches us -- use dilated radius (negative = near)
                result = -maxR;
            }
            textureStore(cocOut, coord, vec4f(result, 0.0, 0.0, 0.0));
        }
    `;

    /** Pass 2b: Vertical near-field CoC dilation (full-res). */
    private static readonly _DILATE_V_SHADER = /* wgsl */`
        ${DepthOfFieldEffect._COMMON}

        @group(0) @binding(0) var cocIn  : texture_2d<f32>;
        @group(0) @binding(1) var cocOut : texture_storage_2d<r32float, write>;
        @group(0) @binding(2) var<uniform> params : DoFParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w = u32(params.screenWidth);
            let h = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let ownCoc = textureLoad(cocIn, coord, 0).r;
            var maxR = max(0.0, -ownCoc);

            let radius = i32(params.maxBlur);
            let iCoord = vec2i(coord);

            for (var dy = -radius; dy <= radius; dy++) {
                let sy = clamp(iCoord.y + dy, 0, i32(h) - 1);
                let sc = vec2u(coord.x, u32(sy));
                let sCoc = textureLoad(cocIn, sc, 0).r;
                if (sCoc < 0.0) {
                    let sR = -sCoc;
                    let dist = f32(abs(dy));
                    if (sR >= dist) {
                        maxR = max(maxR, sR);
                    }
                }
            }

            var result = ownCoc;
            if (maxR > max(0.0, -ownCoc)) {
                result = -maxR;
            }
            textureStore(cocOut, coord, vec4f(result, 0.0, 0.0, 0.0));
        }
    `;

    /** Pass 3: Downsample + near/far separation (full->half). */
    private static readonly _DOWNSAMPLE_SHADER = /* wgsl */`
        ${DepthOfFieldEffect._COMMON}

        @group(0) @binding(0) var colorTex : texture_2d<f32>;
        @group(0) @binding(1) var depthTex : texture_depth_2d;
        @group(0) @binding(2) var nearOut  : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(3) var farOut   : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(4) var<uniform> params : DoFParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let halfCoord = gid.xy;
            let halfW = u32(ceil(params.screenWidth * 0.5));
            let halfH = u32(ceil(params.screenHeight * 0.5));
            if (halfCoord.x >= halfW || halfCoord.y >= halfH) { return; }

            let fullW = u32(params.screenWidth);
            let fullH = u32(params.screenHeight);

            // 2x2 block origin in full-res
            let base = halfCoord * 2u;

            var nearAccum = vec4f(0.0);
            var farAccum = vec4f(0.0);

            // Sample 2x2 block
            for (var dy = 0u; dy < 2u; dy++) {
                for (var dx = 0u; dx < 2u; dx++) {
                    let fc = vec2u(
                        min(base.x + dx, fullW - 1u),
                        min(base.y + dy, fullH - 1u)
                    );
                    let color = textureLoad(colorTex, fc, 0);
                    let depth = textureLoad(depthTex, fc, 0);
                    // Recompute original un-dilated CoC from depth
                    let origCoc = computeCoC(depth, params);

                    if (origCoc < 0.0) {
                        // Near-field: pre-multiplied alpha
                        let coverage = saturate(abs(origCoc) / (params.maxBlur * 0.5));
                        nearAccum += vec4f(color.rgb * coverage, coverage);
                    } else {
                        // Far-field / in-focus: store color and normalized CoC in alpha
                        let normCoc = abs(origCoc) / max(params.maxBlur, 0.001);
                        farAccum += vec4f(color.rgb, normCoc);
                    }
                }
            }

            // Average over 4 texels
            nearAccum *= 0.25;
            farAccum *= 0.25;

            textureStore(nearOut, halfCoord, nearAccum);
            textureStore(farOut, halfCoord, farAccum);
        }
    `;

    /** Pass 4: Vogel disk blur (half-res). */
    private static readonly _BLUR_SHADER = /* wgsl */`
        ${DepthOfFieldEffect._COMMON}

        @group(0) @binding(0) var nearIn     : texture_2d<f32>;
        @group(0) @binding(1) var farIn      : texture_2d<f32>;
        @group(0) @binding(2) var cocDilated  : texture_2d<f32>;
        @group(0) @binding(3) var nearOut    : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(4) var farOut     : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(5) var<uniform> params : DoFParams;

        const GOLDEN_ANGLE : f32 = 2.39996323;
        const NUM_SAMPLES : u32 = 64u;

        fn vogelDisk(i: u32, n: u32) -> vec2f {
            let r = sqrt((f32(i) + 0.5) / f32(n));
            let theta = f32(i) * GOLDEN_ANGLE;
            return vec2f(cos(theta), sin(theta)) * r;
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let halfCoord = gid.xy;
            let halfW = u32(ceil(params.screenWidth * 0.5));
            let halfH = u32(ceil(params.screenHeight * 0.5));
            if (halfCoord.x >= halfW || halfCoord.y >= halfH) { return; }

            // Lookup dilated CoC at corresponding full-res location (center of 2x2 block)
            let fullCoord = vec2u(
                min(halfCoord.x * 2u, u32(params.screenWidth) - 1u),
                min(halfCoord.y * 2u, u32(params.screenHeight) - 1u)
            );
            let dilatedCoc = textureLoad(cocDilated, fullCoord, 0).r;
            let gatherR = abs(dilatedCoc);
            let halfR = gatherR * 0.5; // full-res CoC -> half-res pixels

            // Early out: nothing to blur
            if (halfR < 0.5) {
                textureStore(nearOut, halfCoord, textureLoad(nearIn, halfCoord, 0));
                textureStore(farOut, halfCoord, textureLoad(farIn, halfCoord, 0));
                return;
            }

            var nearAccum = vec4f(0.0);
            var nearCount = 0.0;
            var farAccum = vec4f(0.0);
            var farWeight = 0.0;

            let limHalf = vec2i(i32(halfW) - 1, i32(halfH) - 1);

            for (var i = 0u; i < NUM_SAMPLES; i++) {
                let uv = vogelDisk(i, NUM_SAMPLES);
                let off = uv * halfR;
                let sc = clamp(
                    vec2i(halfCoord) + vec2i(i32(off.x), i32(off.y)),
                    vec2i(0), limHalf
                );
                let scu = vec2u(sc);

                // Near: uniform accumulation of pre-multiplied samples
                let nearSample = textureLoad(nearIn, scu, 0);
                nearAccum += nearSample;
                nearCount += 1.0;

                // Far: scatter-as-gather with coverage weighting
                let farSample = textureLoad(farIn, scu, 0);
                let sampleFarCocNorm = farSample.a; // normalized [0,1]
                let sampleFarCocHalfRes = sampleFarCocNorm * params.maxBlur * 0.5;
                let dist = length(off);
                let coverage = smoothstep(dist - 1.0, dist + 1.0, sampleFarCocHalfRes);
                farAccum += vec4f(farSample.rgb * coverage, farSample.a * coverage);
                farWeight += coverage;
            }

            // Average near (pre-multiplied, so simple average is correct)
            if (nearCount > 0.0) {
                nearAccum /= nearCount;
            }

            // Weighted average for far
            if (farWeight > 0.001) {
                farAccum = vec4f(farAccum.rgb / farWeight, farAccum.a / farWeight);
            }

            textureStore(nearOut, halfCoord, nearAccum);
            textureStore(farOut, halfCoord, farAccum);
        }
    `;

    /** Pass 5: Composite (full-res). */
    private static readonly _COMPOSITE_SHADER = /* wgsl */`
        ${DepthOfFieldEffect._COMMON}

        @group(0) @binding(0) var colorTex   : texture_2d<f32>;
        @group(0) @binding(1) var depthTex   : texture_depth_2d;
        @group(0) @binding(2) var nearBlurTex : texture_2d<f32>;
        @group(0) @binding(3) var farBlurTex  : texture_2d<f32>;
        @group(0) @binding(4) var outputTex  : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(5) var<uniform> params : DoFParams;

        // Manual bilinear interpolation for half-res textures
        fn sampleBilinear(tex: texture_2d<f32>, fullCoord: vec2u, halfW: u32, halfH: u32) -> vec4f {
            // Map full-res pixel center to half-res continuous coordinate
            // Half-res pixel 0 represents the 2x2 block centered at full-res (0.5, 0.5)
            let hc = (vec2f(f32(fullCoord.x), f32(fullCoord.y)) + 0.5) * 0.5 - 0.5;
            let fl = floor(hc);
            let fr = hc - fl;

            let ix0 = u32(max(i32(fl.x), 0));
            let iy0 = u32(max(i32(fl.y), 0));
            let ix1 = min(ix0 + 1u, halfW - 1u);
            let iy1 = min(iy0 + 1u, halfH - 1u);

            let s00 = textureLoad(tex, vec2u(ix0, iy0), 0);
            let s10 = textureLoad(tex, vec2u(ix1, iy0), 0);
            let s01 = textureLoad(tex, vec2u(ix0, iy1), 0);
            let s11 = textureLoad(tex, vec2u(ix1, iy1), 0);

            let top = mix(s00, s10, fr.x);
            let bot = mix(s01, s11, fr.x);
            return mix(top, bot, fr.y);
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w = u32(params.screenWidth);
            let h = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let sharp = textureLoad(colorTex, coord, 0);
            let depth = textureLoad(depthTex, coord, 0);
            let coc = computeCoC(depth, params);
            let absCoc = abs(coc);

            let halfW = u32(ceil(params.screenWidth * 0.5));
            let halfH = u32(ceil(params.screenHeight * 0.5));

            // Always sample near blur (needed for foreground bleed onto in-focus pixels)
            let nearBlur = sampleBilinear(nearBlurTex, coord, halfW, halfH);
            let nearAlpha = clamp(nearBlur.a * 3.0, 0.0, 1.0);

            // Early out: truly in-focus pixel with no near-field bleed
            if (absCoc < 0.5 && nearAlpha < 0.001) {
                textureStore(outputTex, coord, sharp);
                return;
            }

            let farBlur = sampleBilinear(farBlurTex, coord, halfW, halfH);

            var result = sharp;

            // Far field: blend from sharp to blurred based on CoC magnitude
            if (coc > 0.0) {
                let farMix = saturate(absCoc / 2.0);
                result = mix(sharp, vec4f(farBlur.rgb, sharp.a), farMix);
            }

            // Near field: alpha-composite blurred foreground on top of everything
            if (nearAlpha > 0.001) {
                let nearRgb = nearBlur.rgb / max(nearBlur.a, 0.001);
                result = vec4f(mix(result.rgb, nearRgb, nearAlpha), result.a);
            }

            textureStore(outputTex, coord, result);
        }
    `;

    // ========================================================================
    // PostProcessingEffect interface
    // ========================================================================

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        // Shared params buffer: 8 x f32 = 32 bytes
        this._paramsBuffer = device.createBuffer({
            label: 'DoF/Params',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // --- Pass 1: CoC ---
        this._cocPipeline = this._createPipeline(device, 'DoF/CoC', DepthOfFieldEffect._COC_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        // --- Pass 2a: Dilate H ---
        this._dilateHPipeline = this._createPipeline(device, 'DoF/DilateH', DepthOfFieldEffect._DILATE_H_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        // --- Pass 2b: Dilate V ---
        this._dilateVPipeline = this._createPipeline(device, 'DoF/DilateV', DepthOfFieldEffect._DILATE_V_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        // --- Pass 3: Downsample ---
        this._downsamplePipeline = this._createPipeline(device, 'DoF/Downsample', DepthOfFieldEffect._DOWNSAMPLE_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        // --- Pass 4: Blur ---
        this._blurPipeline = this._createPipeline(device, 'DoF/Blur', DepthOfFieldEffect._BLUR_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        // --- Pass 5: Composite ---
        this._compositePipeline = this._createPipeline(device, 'DoF/Composite', DepthOfFieldEffect._COMPOSITE_SHADER, [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]);

        // Create internal textures
        this._createInternalTextures(gbuffer.width, gbuffer.height);

        // Build initial bind groups
        this._buildBindGroups(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number
    ): void {
        if (!this._cocPipeline) return;

        // Rebuild bind groups if external textures changed
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput) {
            this._buildBindGroups(input, depth, output);
        }

        // Write shared params
        const paramsData = new Float32Array([
            this._focusDistance,
            this._focusRange,
            this._maxBlur,
            camera.near,
            camera.far,
            width,
            height,
            0.0, // padding
        ]);
        this._device!.queue.writeBuffer(this._paramsBuffer!, 0, paramsData);

        const wg = (t: number) => Math.ceil(t / 8);
        const halfW = Math.ceil(width / 2);
        const halfH = Math.ceil(height / 2);

        // Pass 1: CoC computation (full-res)
        {
            const pass = commandEncoder.beginComputePass({ label: 'DoF/CoC' });
            pass.setPipeline(this._cocPipeline!);
            pass.setBindGroup(0, this._cocBindGroup!);
            pass.dispatchWorkgroups(wg(width), wg(height));
            pass.end();
        }

        // Pass 2a: Dilate horizontal (full-res)
        {
            const pass = commandEncoder.beginComputePass({ label: 'DoF/DilateH' });
            pass.setPipeline(this._dilateHPipeline!);
            pass.setBindGroup(0, this._dilateHBindGroup!);
            pass.dispatchWorkgroups(wg(width), wg(height));
            pass.end();
        }

        // Pass 2b: Dilate vertical (full-res)
        {
            const pass = commandEncoder.beginComputePass({ label: 'DoF/DilateV' });
            pass.setPipeline(this._dilateVPipeline!);
            pass.setBindGroup(0, this._dilateVBindGroup!);
            pass.dispatchWorkgroups(wg(width), wg(height));
            pass.end();
        }

        // Pass 3: Downsample + separation (full->half)
        {
            const pass = commandEncoder.beginComputePass({ label: 'DoF/Downsample' });
            pass.setPipeline(this._downsamplePipeline!);
            pass.setBindGroup(0, this._downsampleBindGroup!);
            pass.dispatchWorkgroups(wg(halfW), wg(halfH));
            pass.end();
        }

        // Pass 4: Vogel disk blur (half-res)
        {
            const pass = commandEncoder.beginComputePass({ label: 'DoF/Blur' });
            pass.setPipeline(this._blurPipeline!);
            pass.setBindGroup(0, this._blurBindGroup!);
            pass.dispatchWorkgroups(wg(halfW), wg(halfH));
            pass.end();
        }

        // Pass 5: Composite (full-res)
        {
            const pass = commandEncoder.beginComputePass({ label: 'DoF/Composite' });
            pass.setPipeline(this._compositePipeline!);
            pass.setBindGroup(0, this._compositeBindGroup!);
            pass.dispatchWorkgroups(wg(width), wg(height));
            pass.end();
        }
    }

    resize(w: number, h: number, _gbuffer: GBuffer): void {
        this._destroyInternalTextures();
        this._createInternalTextures(w, h);
        // Null out bind groups referencing destroyed textures
        this._cocBindGroup = null;
        this._dilateHBindGroup = null;
        this._dilateVBindGroup = null;
        this._downsampleBindGroup = null;
        this._blurBindGroup = null;
        this._compositeBindGroup = null;
        // Force rebuild on next render()
        this._currentInput = null;
        this._currentDepth = null;
        this._currentOutput = null;
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._paramsBuffer = null;
        this._destroyInternalTextures();
        this._cocPipeline = null;
        this._dilateHPipeline = null;
        this._dilateVPipeline = null;
        this._downsamplePipeline = null;
        this._blurPipeline = null;
        this._compositePipeline = null;
        this._cocBindGroup = null;
        this._dilateHBindGroup = null;
        this._dilateVBindGroup = null;
        this._downsampleBindGroup = null;
        this._blurBindGroup = null;
        this._compositeBindGroup = null;
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    private _createPipeline(
        device: GPUDevice,
        label: string,
        shaderCode: string,
        entries: GPUBindGroupLayoutEntry[]
    ): GPUComputePipeline {
        const module = device.createShaderModule({ label: `${label}/Module`, code: shaderCode });
        const bgl = device.createBindGroupLayout({ label: `${label}/BGL`, entries });
        return device.createComputePipeline({
            label: `${label}/Pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });
    }

    private _createInternalTextures(width: number, height: number): void {
        const device = this._device!;
        const halfW = Math.ceil(width / 2);
        const halfH = Math.ceil(height / 2);

        const fullR16Usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const halfRGBA16Usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

        this._cocTex = device.createTexture({
            label: 'DoF/CoCTex',
            size: [width, height],
            format: 'r32float',
            usage: fullR16Usage,
        });

        this._cocDilTempTex = device.createTexture({
            label: 'DoF/CoCDilTempTex',
            size: [width, height],
            format: 'r32float',
            usage: fullR16Usage,
        });

        this._nearHalfTex = device.createTexture({
            label: 'DoF/NearHalfTex',
            size: [halfW, halfH],
            format: 'rgba16float',
            usage: halfRGBA16Usage,
        });

        this._farHalfTex = device.createTexture({
            label: 'DoF/FarHalfTex',
            size: [halfW, halfH],
            format: 'rgba16float',
            usage: halfRGBA16Usage,
        });

        this._nearBlurTex = device.createTexture({
            label: 'DoF/NearBlurTex',
            size: [halfW, halfH],
            format: 'rgba16float',
            usage: halfRGBA16Usage,
        });

        this._farBlurTex = device.createTexture({
            label: 'DoF/FarBlurTex',
            size: [halfW, halfH],
            format: 'rgba16float',
            usage: halfRGBA16Usage,
        });
    }

    private _destroyInternalTextures(): void {
        this._cocTex?.destroy();
        this._cocDilTempTex?.destroy();
        this._nearHalfTex?.destroy();
        this._farHalfTex?.destroy();
        this._nearBlurTex?.destroy();
        this._farBlurTex?.destroy();
        this._cocTex = null;
        this._cocDilTempTex = null;
        this._nearHalfTex = null;
        this._farHalfTex = null;
        this._nearBlurTex = null;
        this._farBlurTex = null;
    }

    private _buildBindGroups(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const device = this._device!;
        const params = this._paramsBuffer!;

        // Cache texture views to avoid redundant allocations
        const depthView     = depth.createView();
        const inputView     = input.createView();
        const outputView    = output.createView();
        const cocView       = this._cocTex!.createView();
        const cocDilTmpView = this._cocDilTempTex!.createView();
        const nearHalfView  = this._nearHalfTex!.createView();
        const farHalfView   = this._farHalfTex!.createView();
        const nearBlurView  = this._nearBlurTex!.createView();
        const farBlurView   = this._farBlurTex!.createView();
        const paramsBuf     = { buffer: params };

        // Pass 1: CoC
        this._cocBindGroup = device.createBindGroup({
            label: 'DoF/CoC/BG',
            layout: this._cocPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: depthView },
                { binding: 1, resource: cocView },
                { binding: 2, resource: paramsBuf },
            ],
        });

        // Pass 2a: Dilate H
        this._dilateHBindGroup = device.createBindGroup({
            label: 'DoF/DilateH/BG',
            layout: this._dilateHPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: cocView },
                { binding: 1, resource: cocDilTmpView },
                { binding: 2, resource: paramsBuf },
            ],
        });

        // Pass 2b: Dilate V
        this._dilateVBindGroup = device.createBindGroup({
            label: 'DoF/DilateV/BG',
            layout: this._dilateVPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: cocDilTmpView },
                { binding: 1, resource: cocView },
                { binding: 2, resource: paramsBuf },
            ],
        });

        // Pass 3: Downsample
        this._downsampleBindGroup = device.createBindGroup({
            label: 'DoF/Downsample/BG',
            layout: this._downsamplePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: inputView },
                { binding: 1, resource: depthView },
                { binding: 2, resource: nearHalfView },
                { binding: 3, resource: farHalfView },
                { binding: 4, resource: paramsBuf },
            ],
        });

        // Pass 4: Blur
        this._blurBindGroup = device.createBindGroup({
            label: 'DoF/Blur/BG',
            layout: this._blurPipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: nearHalfView },
                { binding: 1, resource: farHalfView },
                { binding: 2, resource: cocView },
                { binding: 3, resource: nearBlurView },
                { binding: 4, resource: farBlurView },
                { binding: 5, resource: paramsBuf },
            ],
        });

        // Pass 5: Composite
        this._compositeBindGroup = device.createBindGroup({
            label: 'DoF/Composite/BG',
            layout: this._compositePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: inputView },
                { binding: 1, resource: depthView },
                { binding: 2, resource: nearBlurView },
                { binding: 3, resource: farBlurView },
                { binding: 4, resource: outputView },
                { binding: 5, resource: paramsBuf },
            ],
        });

        this._currentInput = input;
        this._currentDepth = depth;
        this._currentOutput = output;
    }
}

export { DepthOfFieldEffect };
