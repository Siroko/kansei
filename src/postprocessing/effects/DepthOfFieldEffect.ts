import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface DepthOfFieldOptions {
    /** World-space distance to the in-focus plane. Default 5.0 */
    focusDistance?: number;
    /** Half-extent of the in-focus range (world units). Default 2.0 */
    focusRange?: number;
    /** Maximum blur radius in pixels (maps to CoC = ±1). Default 14 */
    maxBlur?: number;
}

/**
 * Depth of Field — physically-based Bokeh
 * =========================================
 *
 * Algorithm (Unreal Cinematic DoF–inspired):
 *
 *  1. CoC dilation (4-tap cross): expand gather radius at depth discontinuities
 *     so near-field objects correctly bleed onto in-focus / far-field pixels.
 *
 *  2. Scatter-as-gather (80-tap Vogel disk): instead of naively averaging
 *     neighbouring pixels, each gathered sample contributes with a weight
 *     proportional to whether its own CoC disc is large enough to reach the
 *     current pixel.  This creates physically correct hard-disc bokeh highlights
 *     rather than soft blobs.
 *
 *  3. Near / far separation: near-field (foreground) samples and far-field
 *     (background) samples accumulate into separate layers that are composited
 *     with alpha blending so foreground objects scatter onto the background.
 *
 *  4. Focus ramp: the final output is smoothly blended between the sharp source
 *     and the bokeh result over a transition zone around the focus plane, so
 *     there is never a hard cut as pixels enter or leave the depth-of-field region.
 */
class DepthOfFieldEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _paramsBuffer: GPUBuffer | null = null;
    private _bindGroup: GPUBindGroup | null = null;
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

    // ── WGSL compute shader ──────────────────────────────────────────────────

    private static readonly _SHADER = /* wgsl */`
        struct DoFParams {
            focusDistance : f32,
            focusRange    : f32,
            maxBlur       : f32,   // maximum CoC radius in pixels
            near          : f32,
            far           : f32,
            screenWidth   : f32,
            screenHeight  : f32,
            _pad          : f32,
        }

        @group(0) @binding(0) var colorTex  : texture_2d<f32>;
        @group(0) @binding(1) var depthTex  : texture_depth_2d;
        @group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(3) var<uniform>  params : DoFParams;

        fn linearDepth(d: f32) -> f32 {
            return (params.near * params.far) / (params.far - d * (params.far - params.near));
        }

        // Signed Circle of Confusion in pixels.
        //   positive  → behind the focus plane (far / background)
        //   negative  → in front of the focus plane (near / foreground)
        fn computeCoC(d: f32) -> f32 {
            let ld = linearDepth(d);
            return clamp((ld - params.focusDistance) / params.focusRange, -1.0, 1.0)
                   * params.maxBlur;
        }

        // ── Vogel golden-ratio disc ───────────────────────────────────────────
        // Distributes N points uniformly within the unit disc without clustering.
        // Far superior to fixed Poisson arrays for smooth, grain-free bokeh.
        const GOLDEN_ANGLE : f32 = 2.39996323; // ≈ 137.5° in radians

        fn vogelDisk(i: u32, n: u32) -> vec2f {
            let r     = sqrt((f32(i) + 0.5) / f32(n));
            let theta = f32(i) * GOLDEN_ANGLE;
            return vec2f(cos(theta), sin(theta)) * r;
        }

        const NUM_SAMPLES : u32 = 80u;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w     = u32(params.screenWidth);
            let h     = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let depth  = textureLoad(depthTex, coord, 0);
            let sharp  = textureLoad(colorTex, coord, 0);
            let coc    = computeCoC(depth);
            let absCoc = abs(coc);

            // ── 1. CoC dilation (4-tap cross) ────────────────────────────────
            // If a neighbouring pixel is in the near field, enlarge the gather
            // radius so the current pixel will pick up the near-field bleed.
            var maxNearR = absCoc;
            let dilStep  = max(i32(params.maxBlur / 3.0), 2);
            let lim      = vec2i(i32(w) - 1, i32(h) - 1);

            var dilCoord : vec2i;
            var dilCoc   : f32;

            dilCoord = clamp(vec2i(coord) + vec2i( dilStep, 0), vec2i(0), lim);
            dilCoc   = computeCoC(textureLoad(depthTex, dilCoord, 0));
            if (dilCoc < 0.0) { maxNearR = max(maxNearR, -dilCoc); }

            dilCoord = clamp(vec2i(coord) + vec2i(-dilStep, 0), vec2i(0), lim);
            dilCoc   = computeCoC(textureLoad(depthTex, dilCoord, 0));
            if (dilCoc < 0.0) { maxNearR = max(maxNearR, -dilCoc); }

            dilCoord = clamp(vec2i(coord) + vec2i(0,  dilStep), vec2i(0), lim);
            dilCoc   = computeCoC(textureLoad(depthTex, dilCoord, 0));
            if (dilCoc < 0.0) { maxNearR = max(maxNearR, -dilCoc); }

            dilCoord = clamp(vec2i(coord) + vec2i(0, -dilStep), vec2i(0), lim);
            dilCoc   = computeCoC(textureLoad(depthTex, dilCoord, 0));
            if (dilCoc < 0.0) { maxNearR = max(maxNearR, -dilCoc); }

            let gatherR = maxNearR;

            // Fully in-focus pixel with no near-field neighbours → pass through.
            if (gatherR < 0.5) {
                textureStore(outputTex, coord, sharp);
                return;
            }

            // ── 2. Scatter-as-gather (Vogel disk) ────────────────────────────
            // Each sample contributes with a weight equal to the probability that
            // its own bokeh disc is large enough to "scatter" light onto the
            // current pixel.  This produces physically correct hard-disc bokeh.
            var farColor  = vec4f(0.0);  var farW  = 0.0;
            var nearColor = vec4f(0.0);  var nearW = 0.0;

            for (var i = 0u; i < NUM_SAMPLES; i++) {
                let uv  = vogelDisk(i, NUM_SAMPLES);
                let off = uv * gatherR;
                let sc  = clamp(
                    vec2i(coord) + vec2i(i32(off.x), i32(off.y)),
                    vec2i(0), lim
                );

                let sColor = textureLoad(colorTex, sc, 0);
                let sCoc   = computeCoC(textureLoad(depthTex, sc, 0));
                let sR     = abs(sCoc);
                let dist   = length(off);

                // Hard-disc coverage with a 3-pixel soft anti-aliased edge.
                // coverage → 1 when the sample's CoC disc fully covers us,
                //          → 0 when the sample is too sharp to reach this pixel.
                let coverage = smoothstep(dist - 1.5, dist + 1.5, sR);

                if (sCoc >= 0.0) {
                    farColor += sColor * coverage;
                    farW     += coverage;
                } else {
                    // Near-field samples scatter forward onto everything.
                    nearColor += sColor * coverage;
                    nearW     += coverage;
                }
            }

            // ── 3. Near / far composite ───────────────────────────────────────
            let far  = select(sharp, farColor  / farW,  farW  > 0.001);
            let near = select(sharp, nearColor / nearW, nearW > 0.001);

            // Fraction of the disc covered by near-field bokeh.
            // Amplified so even a thin near-field region blends noticeably.
            let nearAlpha = clamp(nearW / f32(NUM_SAMPLES) * 2.5, 0.0, 1.0);

            var bokehResult : vec4f;
            if (coc < 0.0) {
                // Near-field pixel: fade from sharp to near bokeh over half maxBlur.
                let t = clamp(absCoc / max(params.maxBlur * 0.5, 1.0), 0.0, 1.0);
                bokehResult = mix(sharp, near, t);
            } else {
                // Far-field / in-focus: far bokeh with near-field bleed on top.
                bokehResult = mix(far, near, nearAlpha);
            }

            // ── 4. Focus ramp ─────────────────────────────────────────────────
            // Smoothly blend between sharp and bokeh so there is never a hard
            // cut at the focus-plane boundary.  The transition spans the first
            // 25 % of maxBlur (at least 2 px), giving a gentle fade-in of blur.
            let transitionPx = max(params.maxBlur * 0.25, 2.0);
            let blendFactor  = smoothstep(0.0, transitionPx, absCoc);
            textureStore(outputTex, coord, mix(sharp, bokehResult, blendFactor));
        }
    `;

    // ── PostProcessingEffect interface ───────────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._paramsBuffer = device.createBuffer({
            label: 'DoF/Params',
            size: 32, // 8 × f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const module = device.createShaderModule({ code: DepthOfFieldEffect._SHADER });
        const bgl = device.createBindGroupLayout({
            label: 'DoF/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._pipeline = device.createComputePipeline({
            label: 'DoF/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._buildBindGroup(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _buildBindGroup(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const bgl = this._pipeline!.getBindGroupLayout(0);
        this._bindGroup = this._device!.createBindGroup({
            label: 'DoF/BindGroup',
            layout: bgl,
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: output.createView() },
                { binding: 3, resource: { buffer: this._paramsBuffer! } },
            ],
        });
        this._currentInput  = input;
        this._currentDepth  = depth;
        this._currentOutput = output;
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
        if (!this._pipeline) return;

        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput) {
            this._buildBindGroup(input, depth, output);
        }

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
        const pass = commandEncoder.beginComputePass({ label: 'DoF' });
        pass.setPipeline(this._pipeline!);
        pass.setBindGroup(0, this._bindGroup!);
        pass.dispatchWorkgroups(wg(width), wg(height));
        pass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // Params are recalculated every frame; nothing size-dependent to rebuild.
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._paramsBuffer = null;
        this._pipeline     = null;
        this._bindGroup    = null;
    }
}

export { DepthOfFieldEffect };
