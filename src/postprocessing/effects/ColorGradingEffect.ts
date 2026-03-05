import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface ColorGradingOptions {
    /** Additive brightness shift (-1 to 1). Default 0.0 */
    brightness?: number;
    /** Contrast multiplier (0 to 3). Default 1.0 */
    contrast?: number;
    /** Saturation multiplier (0 to 3). Default 1.0 */
    saturation?: number;
    /** Color temperature shift (-1 cool/blue to 1 warm/orange). Default 0.0 */
    temperature?: number;
    /** Green-magenta tint shift (-1 green to 1 magenta). Default 0.0 */
    tint?: number;
    /** Highlight region multiplier (0 to 2). Default 1.0 */
    highlights?: number;
    /** Shadow region multiplier (0 to 2). Default 1.0 */
    shadows?: number;
    /** Lifts the black floor (0 to 0.5). Default 0.0 */
    blackPoint?: number;
}

/**
 * Single-Pass Color Grading
 * ==========================
 *
 * Full-res compute pass applying: black point, shadows/highlights,
 * brightness, contrast, saturation, temperature, and tint adjustments.
 */
class ColorGradingEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _paramsBuffer: GPUBuffer | null = null;
    private _bindGroup: GPUBindGroup | null = null;

    private _currentInput: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    brightness: number;
    contrast: number;
    saturation: number;
    temperature: number;
    tint: number;
    highlights: number;
    shadows: number;
    blackPoint: number;

    constructor(options: ColorGradingOptions = {}) {
        super();
        this.brightness = options.brightness ?? 0.0;
        this.contrast   = options.contrast   ?? 1.0;
        this.saturation = options.saturation ?? 1.0;
        this.temperature = options.temperature ?? 0.0;
        this.tint       = options.tint       ?? 0.0;
        this.highlights = options.highlights ?? 1.0;
        this.shadows    = options.shadows    ?? 1.0;
        this.blackPoint = options.blackPoint ?? 0.0;
    }

    // ========================================================================
    // WGSL Shader
    // ========================================================================

    private static readonly _SHADER = /* wgsl */`
        struct GradeParams {
            brightness  : f32,
            contrast    : f32,
            saturation  : f32,
            temperature : f32,
            tint        : f32,
            highlights  : f32,
            shadows     : f32,
            blackPoint  : f32,
            screenW     : f32,
            screenH     : f32,
            _pad0       : f32,
            _pad1       : f32,
        }

        @group(0) @binding(0) var inputTex  : texture_2d<f32>;
        @group(0) @binding(1) var outputTex : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(2) var<uniform> params : GradeParams;

        fn luminance(c: vec3f) -> f32 {
            return dot(c, vec3f(0.2126, 0.7152, 0.0722));
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let w = u32(params.screenW);
            let h = u32(params.screenH);
            if (gid.x >= w || gid.y >= h) { return; }

            let pixel = textureLoad(inputTex, gid.xy, 0);
            var color = pixel.rgb;

            // 1. Black point
            let bp = params.blackPoint;
            if (bp > 0.0) {
                color = max(color - vec3f(bp), vec3f(0.0)) / (1.0 - bp);
            }

            // 2. Shadows / Highlights
            let lum = luminance(color);
            let shadowHighlightMix = smoothstep(0.0, 1.0, lum);
            let shMul = mix(params.shadows, params.highlights, shadowHighlightMix);
            color *= shMul;

            // 3. Brightness
            color += vec3f(params.brightness);

            // 4. Contrast
            color = (color - vec3f(0.5)) * params.contrast + vec3f(0.5);

            // 5. Saturation
            let gray = luminance(color);
            color = mix(vec3f(gray), color, params.saturation);

            // 6. Temperature (R/B shift)
            color.r += params.temperature * 0.1;
            color.b -= params.temperature * 0.1;

            // 7. Tint (G shift)
            color.g += params.tint * 0.1;

            textureStore(outputTex, gid.xy, vec4f(color, pixel.a));
        }
    `;

    // ========================================================================
    // PostProcessingEffect interface
    // ========================================================================

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._paramsBuffer = device.createBuffer({
            label: 'ColorGrading/Params',
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const module = device.createShaderModule({ code: ColorGradingEffect._SHADER });
        const bgl = device.createBindGroupLayout({
            label: 'ColorGrading/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._pipeline = device.createComputePipeline({
            label: 'ColorGrading/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._buildBindGroup(gbuffer.colorTexture, gbuffer.outputTexture);
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
    ): void {
        if (!this._pipeline) return;

        if (input !== this._currentInput || output !== this._currentOutput) {
            this._buildBindGroup(input, output);
        }

        const paramsData = new Float32Array([
            this.brightness,
            this.contrast,
            this.saturation,
            this.temperature,
            this.tint,
            this.highlights,
            this.shadows,
            this.blackPoint,
            width,
            height,
            0.0,
            0.0,
        ]);
        this._device!.queue.writeBuffer(this._paramsBuffer!, 0, paramsData);

        const wg = (t: number) => Math.ceil(t / 8);
        const pass = commandEncoder.beginComputePass({ label: 'ColorGrading' });
        pass.setPipeline(this._pipeline!);
        pass.setBindGroup(0, this._bindGroup!);
        pass.dispatchWorkgroups(wg(width), wg(height));
        pass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // No internal textures — params recalculated every frame.
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._paramsBuffer = null;
        this._pipeline = null;
        this._bindGroup = null;
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    private _buildBindGroup(input: GPUTexture, output: GPUTexture): void {
        this._bindGroup = this._device!.createBindGroup({
            label: 'ColorGrading/BG',
            layout: this._pipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: output.createView() },
                { binding: 2, resource: { buffer: this._paramsBuffer! } },
            ],
        });
        this._currentInput = input;
        this._currentOutput = output;
    }
}

export { ColorGradingEffect };
