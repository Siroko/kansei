import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface DepthOfFieldOptions {
    /** World-space distance to the in-focus plane. Default 5.0 */
    focusDistance?: number;
    /** Half-extent of the in-focus range around focusDistance. Default 2.0 */
    focusRange?: number;
    /** Maximum blur radius in pixels. Default 12 */
    maxBlur?: number;
}

/**
 * Depth of Field (DoF) — Bokeh blur
 * ====================================
 * Computes a per-pixel Circle of Confusion (CoC) from the linearised depth
 * buffer, then gathers the colour texture with a 32-tap multi-ring Poisson
 * disc whose radius is proportional to the CoC.  Near-field (in-front-of-focus)
 * and far-field (behind-focus) are handled separately:
 *
 *  - Far field  (coc > 0): depth-aware weights prevent sharp in-focus geometry
 *                          from bleeding into the bokeh region.
 *  - Near field (coc < 0): all samples contribute equally, so near objects
 *                          naturally bleed (scatter) onto the background.
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
        this._maxBlur       = options.maxBlur       ?? 12;
    }

    // ── WGSL compute shader ──────────────────────────────────────────────────

    private static readonly _SHADER = /* wgsl */`
        struct DoFParams {
            focusDistance : f32,
            focusRange    : f32,
            maxBlur       : f32,
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

        // Convert raw depth [0,1] to a linear view-space distance.
        fn linearDepth(d: f32) -> f32 {
            return (params.near * params.far) / (params.far - d * (params.far - params.near));
        }

        // Signed Circle of Confusion in pixels.
        //   positive → behind the focus plane (far / background)
        //   negative → in front of the focus plane (near / foreground)
        fn computeCoC(d: f32) -> f32 {
            let ld = linearDepth(d);
            return clamp((ld - params.focusDistance) / params.focusRange, -1.0, 1.0) * params.maxBlur;
        }

        // 32-tap multi-ring Poisson disc (unit radius).
        // Arranged as 4 concentric rings: 6 + 8 + 10 + 8 samples.
        const POISSON_COUNT : u32 = 32u;
        const poissonDisk = array<vec2f, 32>(
            // Ring 1 — r ≈ 0.20, 6 samples
            vec2f( 0.200,  0.000), vec2f( 0.100,  0.173), vec2f(-0.100,  0.173),
            vec2f(-0.200,  0.000), vec2f(-0.100, -0.173), vec2f( 0.100, -0.173),
            // Ring 2 — r ≈ 0.40, 8 samples
            vec2f( 0.370,  0.153), vec2f( 0.153,  0.370), vec2f(-0.153,  0.370),
            vec2f(-0.370,  0.153), vec2f(-0.370, -0.153), vec2f(-0.153, -0.370),
            vec2f( 0.153, -0.370), vec2f( 0.370, -0.153),
            // Ring 3 — r ≈ 0.65, 10 samples
            vec2f( 0.650,  0.000), vec2f( 0.526,  0.382), vec2f( 0.201,  0.618),
            vec2f(-0.201,  0.618), vec2f(-0.526,  0.382), vec2f(-0.650,  0.000),
            vec2f(-0.526, -0.382), vec2f(-0.201, -0.618), vec2f( 0.201, -0.618),
            vec2f( 0.526, -0.382),
            // Ring 4 — r ≈ 0.90, 8 samples
            vec2f( 0.832,  0.345), vec2f( 0.345,  0.832), vec2f(-0.345,  0.832),
            vec2f(-0.832,  0.345), vec2f(-0.832, -0.345), vec2f(-0.345, -0.832),
            vec2f( 0.345, -0.832), vec2f( 0.832, -0.345),
        );

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w     = u32(params.screenWidth);
            let h     = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let depth      = textureLoad(depthTex, coord, 0);
            let coc        = computeCoC(depth);
            let blurRadius = abs(coc);

            // Pixels near the focus plane are passed through unchanged.
            if (blurRadius < 0.5) {
                textureStore(outputTex, coord, textureLoad(colorTex, coord, 0));
                return;
            }

            var color       = vec4f(0.0);
            var totalWeight = 0.0;

            for (var i = 0u; i < POISSON_COUNT; i++) {
                let offset = poissonDisk[i] * blurRadius;
                let sampleCoord = clamp(
                    vec2i(coord) + vec2i(i32(offset.x), i32(offset.y)),
                    vec2i(0),
                    vec2i(i32(w) - 1, i32(h) - 1)
                );

                let sampleColor = textureLoad(colorTex, sampleCoord, 0);
                let sampleCoc   = computeCoC(textureLoad(depthTex, sampleCoord, 0));

                // Depth-aware weight:
                //   Near-field pixel (coc < 0): all samples contribute — near objects
                //   scatter / bleed onto the background naturally.
                //   Far-field pixel  (coc > 0): weight by the sample's own defocus so
                //   sharp in-focus surfaces don't bleed into the bokeh region.
                //   A small minimum weight (0.05) keeps far bokeh from going black.
                var w_s: f32;
                if (coc < 0.0) {
                    w_s = 1.0;
                } else {
                    w_s = max(smoothstep(0.0, 1.0, abs(sampleCoc) / max(1.0, blurRadius)), 0.05);
                }

                color       += sampleColor * w_s;
                totalWeight += w_s;
            }

            textureStore(outputTex, coord, color / totalWeight);
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
        // Params recalculated every frame; nothing to rebuild.
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._paramsBuffer = null;
        this._pipeline     = null;
        this._bindGroup    = null;
    }
}

export { DepthOfFieldEffect };
