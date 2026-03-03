import { Camera } from '../../cameras/Camera';
import { Vector2 } from '../../math/Vector2';
import { Vector3 } from '../../math/Vector3';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface GodRaysOptions {
    /**
     * Screen-space position of the light source in [0, 1] UV space.
     * Defaults to the top-centre of the screen (0.5, 0.1).
     */
    lightScreenPos?: Vector2;
    /**
     * World-space light position.  When provided, the effect projects it to
     * screen space each frame using the camera matrices.  Takes priority over
     * lightScreenPos.
     */
    lightWorldPos?: Vector3;
    /** Radial-sample step scale (controls how far each step marches). Default 0.97 */
    density?: number;
    /** Per-sample light energy weight. Default 0.01 */
    weight?: number;
    /** Per-step decay factor (0..1). Values close to 1 give longer rays. Default 0.96 */
    decay?: number;
    /** Final exposure multiplier applied to the accumulated rays. Default 0.65 */
    exposure?: number;
    /** Number of radial march steps. Default 100 */
    numSamples?: number;
}

/**
 * Volumetric God Rays (Crepuscular Rays)
 * ========================================
 * Implements screen-space volumetric light scattering.  From each pixel a ray
 * is marched toward the light source in screen space; at each step the sample
 * is accumulated with a decay factor.  Pixels in front of the light source
 * (sky / background, depth ≈ 1) contribute light energy, while occluded pixels
 * do not, creating the characteristic shaft-of-light appearance.
 *
 * The result is added on top of the input colour for a pure-additive glow.
 */
class GodRaysEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _paramsBuffer: GPUBuffer | null = null;
    private _bindGroup: GPUBindGroup | null = null;
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    private readonly _lightScreenPos: Vector2;
    private readonly _lightWorldPos: Vector3 | null;
    private readonly _density: number;
    private readonly _weight: number;
    private readonly _decay: number;
    private readonly _exposure: number;
    private readonly _numSamples: number;

    constructor(options: GodRaysOptions = {}) {
        super();
        this._lightScreenPos = options.lightScreenPos ?? new Vector2(0.5, 0.1);
        this._lightWorldPos  = options.lightWorldPos  ?? null;
        this._density        = options.density        ?? 0.97;
        this._weight         = options.weight         ?? 0.01;
        this._decay          = options.decay          ?? 0.96;
        this._exposure       = options.exposure       ?? 0.65;
        this._numSamples     = options.numSamples     ?? 100;
    }

    // ── WGSL compute shader ──────────────────────────────────────────────────

    private static readonly _SHADER = /* wgsl */`
        struct GodRaysParams {
            lightPos     : vec2f,
            density      : f32,
            weight       : f32,
            decay        : f32,
            exposure     : f32,
            numSamples   : u32,
            screenWidth  : f32,
            screenHeight : f32,
            _pad         : f32,
        }

        @group(0) @binding(0) var colorTex  : texture_2d<f32>;
        @group(0) @binding(1) var depthTex  : texture_depth_2d;
        @group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(3) var<uniform>  params : GodRaysParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w     = u32(params.screenWidth);
            let h     = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let uv       = vec2f(f32(coord.x) / params.screenWidth, f32(coord.y) / params.screenHeight);
            let lightPos = params.lightPos;

            // March from pixel UV toward the light source in screen space.
            let step     = (lightPos - uv) * (1.0 / f32(params.numSamples)) * params.density;
            var texCoord = uv;
            var decay    = 1.0;
            var godRays  = vec3f(0.0);

            for (var i = 0u; i < params.numSamples; i++) {
                texCoord += step;

                if (texCoord.x < 0.0 || texCoord.x > 1.0 ||
                    texCoord.y < 0.0 || texCoord.y > 1.0) {
                    break;
                }

                let sampleCoord = vec2u(
                    u32(clamp(texCoord.x * params.screenWidth,  0.0, params.screenWidth  - 1.0)),
                    u32(clamp(texCoord.y * params.screenHeight, 0.0, params.screenHeight - 1.0))
                );
                let sampleColor = textureLoad(colorTex, sampleCoord, 0).rgb;
                let sampleDepth = textureLoad(depthTex, sampleCoord, 0);

                // Only background/sky pixels (no occluding geometry) contribute rays.
                let isBackground = sampleDepth >= 0.9999;
                let brightness   = dot(sampleColor, vec3f(0.299, 0.587, 0.114));
                let contrib      = select(0.0, brightness, isBackground);

                godRays  += sampleColor * contrib * decay * params.weight;
                decay    *= params.decay;
            }

            let original   = textureLoad(colorTex, coord, 0);
            let finalColor = original.rgb + godRays * params.exposure;
            textureStore(outputTex, coord, vec4f(finalColor, original.a));
        }
    `;

    // ── PostProcessingEffect interface ───────────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._paramsBuffer = device.createBuffer({
            label: 'GodRays/Params',
            size: 48, // 10 × f32 + 2 padding → 48 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const module = device.createShaderModule({ code: GodRaysEffect._SHADER });
        const bgl = device.createBindGroupLayout({
            label: 'GodRays/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._pipeline = device.createComputePipeline({
            label: 'GodRays/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._buildBindGroup(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _buildBindGroup(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const bgl = this._pipeline!.getBindGroupLayout(0);
        this._bindGroup = this._device!.createBindGroup({
            label: 'GodRays/BindGroup',
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

        // Optionally project the world-space light position to screen space.
        let lightX = this._lightScreenPos.x;
        let lightY = this._lightScreenPos.y;

        if (this._lightWorldPos) {
            const lp = this._lightWorldPos;
            // Manual mat4 * vec4 — matrices are column-major (internalMat4[col*4+row]).
            const vm = camera.viewMatrix.internalMat4;
            const vx = vm[0]*lp.x + vm[4]*lp.y + vm[8]*lp.z  + vm[12];
            const vy = vm[1]*lp.x + vm[5]*lp.y + vm[9]*lp.z  + vm[13];
            const vz = vm[2]*lp.x + vm[6]*lp.y + vm[10]*lp.z + vm[14];
            const vw = vm[3]*lp.x + vm[7]*lp.y + vm[11]*lp.z + vm[15];

            const pm = camera.projectionMatrix.internalMat4;
            const cx = pm[0]*vx + pm[4]*vy + pm[8]*vz  + pm[12]*vw;
            const cy = pm[1]*vx + pm[5]*vy + pm[9]*vz  + pm[13]*vw;
            const cw = pm[3]*vx + pm[7]*vy + pm[11]*vz + pm[15]*vw;

            if (cw !== 0) {
                lightX = (cx / cw) * 0.5 + 0.5;
                lightY = 1.0 - ((cy / cw) * 0.5 + 0.5);
            }
        }

        // GodRaysParams layout:
        // offset  0: lightPos     (vec2f)
        // offset  8: density      (f32)
        // offset 12: weight       (f32)
        // offset 16: decay        (f32)
        // offset 20: exposure     (f32)
        // offset 24: numSamples   (u32)
        // offset 28: screenWidth  (f32)
        // offset 32: screenHeight (f32)
        // offset 36: _pad         (f32)
        const paramsData = new ArrayBuffer(48);
        const f = new Float32Array(paramsData);
        const u = new Uint32Array(paramsData);
        f[0] = lightX;
        f[1] = lightY;
        f[2] = this._density;
        f[3] = this._weight;
        f[4] = this._decay;
        f[5] = this._exposure;
        u[6] = this._numSamples;
        f[7] = width;
        f[8] = height;
        f[9] = 0.0;

        this._device!.queue.writeBuffer(this._paramsBuffer!, 0, paramsData);

        const wg = (t: number) => Math.ceil(t / 8);
        const pass = commandEncoder.beginComputePass({ label: 'GodRays' });
        pass.setPipeline(this._pipeline!);
        pass.setBindGroup(0, this._bindGroup!);
        pass.dispatchWorkgroups(wg(width), wg(height));
        pass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // Params are fully recalculated every frame.
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._paramsBuffer = null;
        this._pipeline     = null;
        this._bindGroup    = null;
    }
}

export { GodRaysEffect };
