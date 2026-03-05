import { Camera } from '../../cameras/Camera';
import { FroxelGrid } from '../../froxels/FroxelGrid';
import { ShadowMap } from '../../shadows/ShadowMap';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { mat4 } from 'gl-matrix';

export interface VolumetricFogOptions {
    froxelGrid: FroxelGrid;
    shadowMap: ShadowMap;
    lightDirection?: [number, number, number];
    lightColor?: [number, number, number];
    baseDensity?: number;
    heightFalloff?: number;
    extinctionCoeff?: number;
    anisotropy?: number;
    windDirection?: [number, number, number];
}

class VolumetricFogEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _froxelGrid: FroxelGrid;
    private _shadowMap: ShadowMap;

    lightDir: [number, number, number];
    lightColor: [number, number, number];
    baseDensity: number;
    heightFalloff: number;
    extinctionCoeff: number;
    anisotropy: number;
    windDir: [number, number, number];

    // Injection pass
    private _injectPipeline: GPUComputePipeline | null = null;
    private _injectBG: GPUBindGroup | null = null;
    private _fogParamsBuffer: GPUBuffer | null = null;

    // Composite pass
    private _compositePipeline: GPUComputePipeline | null = null;
    private _compositeBG: GPUBindGroup | null = null;
    private _compositeParamsBuffer: GPUBuffer | null = null;
    private _accumSampler: GPUSampler | null = null;
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    private _startTime = performance.now();
    private _invVP = mat4.create();

    constructor(options: VolumetricFogOptions) {
        super();
        this._froxelGrid    = options.froxelGrid;
        this._shadowMap     = options.shadowMap;
        this.lightDir       = options.lightDirection ?? [0.5, -0.8, 0.3];
        this.lightColor     = options.lightColor ?? [1.0, 0.95, 0.85];
        this.baseDensity    = options.baseDensity ?? 0.02;
        this.heightFalloff  = options.heightFalloff ?? 0.1;
        this.extinctionCoeff = options.extinctionCoeff ?? 1.0;
        this.anisotropy     = options.anisotropy ?? 0.6;
        this.windDir        = options.windDirection ?? [0, 0, 0];
    }

    // ── Injection shader ──────────────────────────────────────────────────

    private static _INJECT_SHADER = FroxelGrid.WGSL_HELPERS + /* wgsl */`

        struct FogParams {
            invViewProj     : mat4x4f,
            lightViewProj   : mat4x4f,
            cameraPos       : vec3f,
            baseDensity     : f32,
            lightDir        : vec3f,
            heightFalloff   : f32,
            lightColor      : vec3f,
            extinctionCoeff : f32,
            windOffset      : vec3f,
            anisotropy      : f32,
            gridNear        : f32,
            gridFar         : f32,
            time            : f32,
            gridW           : u32,
            gridH           : u32,
            gridD           : u32,
            cameraNear      : f32,
            cameraFar       : f32,
        }

        @group(0) @binding(0) var scatterExtTex  : texture_storage_3d<rgba16float, write>;
        @group(0) @binding(1) var shadowDepthTex  : texture_depth_2d;
        @group(0) @binding(2) var<uniform> params : FogParams;

        fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
            let g2 = g * g;
            return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
        }

        @compute @workgroup_size(4, 4, 4)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            if (gid.x >= params.gridW || gid.y >= params.gridH || gid.z >= params.gridD) { return; }

            // Compute world position: use grid near/far for slice depth,
            // but camera near/far for NDC Z (must match the projection in invViewProj).
            // gl-matrix v3 perspectiveNO maps Z to [-1,1], so we use that convention.
            let linearD = sliceDepth(f32(gid.z) + 0.5, params.gridNear, params.gridFar, f32(params.gridD));

            // Skip slices closer than the camera near plane — they can't be
            // unprojected correctly and would produce garbage world positions.
            if (linearD < params.cameraNear) {
                textureStore(scatterExtTex, gid, vec4f(0.0));
                return;
            }

            let ndcZ = ((params.cameraFar + params.cameraNear) * linearD - 2.0 * params.cameraFar * params.cameraNear)
                     / ((params.cameraFar - params.cameraNear) * linearD);
            let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(f32(params.gridW), f32(params.gridH));
            let ndcX = uv.x * 2.0 - 1.0;
            let ndcY = (1.0 - uv.y) * 2.0 - 1.0;
            let world = params.invViewProj * vec4f(ndcX, ndcY, ndcZ, 1.0);
            let worldPos = world.xyz / world.w;

            // Wind-displaced sample position
            let samplePos = worldPos + params.windOffset;

            // Height-exponential fog density
            let density = params.baseDensity * exp(-params.heightFalloff * max(samplePos.y, 0.0));
            let extinction = density * params.extinctionCoeff;

            // Shadow map lookup — 2x2 PCF for softer shadow edges
            let lightClip = params.lightViewProj * vec4f(worldPos, 1.0);
            let lightNDC  = lightClip.xyz / lightClip.w;
            let shadowUV  = vec2f(lightNDC.x * 0.5 + 0.5, 1.0 - (lightNDC.y * 0.5 + 0.5));

            var visibility = 1.0;
            if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0) {
                let shadowDim = textureDimensions(shadowDepthTex, 0);
                let texelSize = 1.0 / vec2f(shadowDim);
                let coordF = shadowUV * vec2f(shadowDim) - 0.5;
                let base = vec2i(floor(coordF));
                var pcf = 0.0;
                for (var dy = 0; dy <= 1; dy++) {
                    for (var dx = 0; dx <= 1; dx++) {
                        let sc = clamp(base + vec2i(dx, dy), vec2i(0), vec2i(shadowDim) - 1);
                        let sd = textureLoad(shadowDepthTex, sc, 0);
                        pcf += select(0.0, 1.0, lightNDC.z <= sd + 0.005);
                    }
                }
                visibility = pcf * 0.25;
            }

            // Phase function (Henyey-Greenstein)
            let viewDir  = normalize(worldPos - params.cameraPos);
            let cosTheta = dot(viewDir, -normalize(params.lightDir));
            let phase    = henyeyGreenstein(cosTheta, params.anisotropy);

            // In-scattered light
            let scatter = density * params.lightColor * visibility * phase;

            textureStore(scatterExtTex, gid, vec4f(scatter, extinction));
        }
    `;

    // ── Composite shader ──────────────────────────────────────────────────

    private static _COMPOSITE_SHADER = FroxelGrid.WGSL_HELPERS + /* wgsl */`

        struct CompositeParams {
            cameraNear   : f32,
            cameraFar    : f32,
            gridNear     : f32,
            gridFar      : f32,
            gridD        : f32,
            screenWidth  : f32,
            screenHeight : f32,
            _pad         : f32,
        }

        @group(0) @binding(0) var inputTex     : texture_2d<f32>;
        @group(0) @binding(1) var depthTex     : texture_depth_2d;
        @group(0) @binding(2) var outputTex    : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(3) var accumTex     : texture_3d<f32>;
        @group(0) @binding(4) var accumSampler : sampler;
        @group(0) @binding(5) var<uniform> cp  : CompositeParams;

        // Screen-space dither to break up froxel slice banding
        fn screenHash(p: vec2f) -> f32 {
            var p3 = fract(vec3f(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            if (f32(coord.x) >= cp.screenWidth || f32(coord.y) >= cp.screenHeight) { return; }

            let sceneColor = textureLoad(inputTex, coord, 0);
            let depth      = textureLoad(depthTex, coord, 0);

            // Reverse-perspective: NDC depth -> linear depth
            // gl-matrix v3 perspective maps Z to [-1,1]; depth buffer stores clamped [0,1]
            let linearDepth = 2.0 * cp.cameraNear * cp.cameraFar
                            / ((cp.cameraFar + cp.cameraNear) - depth * (cp.cameraFar - cp.cameraNear));

            // Fractional slice in the froxel grid (uses grid near/far for exponential distribution)
            let sliceFloat = depthToSlice(linearDepth, cp.gridNear, cp.gridFar, cp.gridD);

            // Per-pixel dither: jitter slice by ±0.5 to smooth banding
            let jitter = screenHash(vec2f(coord)) - 0.5;
            let jitteredSlice = clamp((sliceFloat + jitter) / cp.gridD, 0.0, 1.0);

            // UVW for 3D texture sampling (trilinear)
            let uv = vec2f(f32(coord.x) / cp.screenWidth, f32(coord.y) / cp.screenHeight);
            let gridUV = vec3f(uv.x, uv.y, jitteredSlice);

            let fogData = textureSampleLevel(accumTex, accumSampler, gridUV, 0.0);
            let accLight      = fogData.rgb;
            let transmittance = fogData.a;

            // Composite: scene * transmittance + accumulated in-scattered light
            let finalColor = sceneColor.rgb * transmittance + accLight;
            textureStore(outputTex, coord, vec4f(finalColor, sceneColor.a));
        }
    `;

    // ── PostProcessingEffect interface ───────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._fogParamsBuffer = device.createBuffer({
            label: 'VolumetricFog/FogParams',
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._compositeParamsBuffer = device.createBuffer({
            label: 'VolumetricFog/CompositeParams',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._accumSampler = device.createSampler({
            label: 'VolumetricFog/AccumSampler',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // ── Injection pipeline ──
        const injectModule = device.createShaderModule({
            label: 'VolumetricFog/InjectShader',
            code: VolumetricFogEffect._INJECT_SHADER,
        });

        const injectBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/Inject BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._injectPipeline = device.createComputePipeline({
            label: 'VolumetricFog/InjectPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [injectBGL] }),
            compute: { module: injectModule, entryPoint: 'main' },
        });

        this._injectBG = device.createBindGroup({
            label: 'VolumetricFog/Inject BG',
            layout: injectBGL,
            entries: [
                { binding: 0, resource: this._froxelGrid.scatterExtinctionTex.createView() },
                { binding: 1, resource: this._shadowMap.depthTexture.createView() },
                { binding: 2, resource: { buffer: this._fogParamsBuffer } },
            ],
        });

        // ── Composite pipeline ──
        const compositeModule = device.createShaderModule({
            label: 'VolumetricFog/CompositeShader',
            code: VolumetricFogEffect._COMPOSITE_SHADER,
        });

        const compositeBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/Composite BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._compositePipeline = device.createComputePipeline({
            label: 'VolumetricFog/CompositePipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [compositeBGL] }),
            compute: { module: compositeModule, entryPoint: 'main' },
        });

        this._buildCompositeBG(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _buildCompositeBG(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const bgl = this._compositePipeline!.getBindGroupLayout(0);
        this._compositeBG = this._device!.createBindGroup({
            label: 'VolumetricFog/Composite BG',
            layout: bgl,
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: output.createView() },
                { binding: 3, resource: this._froxelGrid.accumTex.createView() },
                { binding: 4, resource: this._accumSampler! },
                { binding: 5, resource: { buffer: this._compositeParamsBuffer! } },
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
        if (!this._injectPipeline || !this._compositePipeline) return;

        const device = this._device!;
        const grid = this._froxelGrid;
        const time = (performance.now() - this._startTime) / 1000;

        // ── Update fog params ──
        const vp = mat4.create();
        mat4.multiply(vp, camera.projectionMatrix.internalMat4, camera.viewMatrix.internalMat4);

        // Use raw gl-matrix VP inverse — the injection shader computes NDC Z
        // in [-1,1] to match gl-matrix v3's perspectiveNO convention.
        mat4.invert(this._invVP, vp);

        const iv = camera.inverseViewMatrix.internalMat4;

        const fogParams = new Float32Array(64); // 256 bytes
        fogParams.set(this._invVP as unknown as Float32Array, 0);               // invViewProj
        fogParams.set(this._shadowMap.lightViewProjMatrix, 16);                  // lightViewProj
        fogParams[32] = iv[12]; fogParams[33] = iv[13]; fogParams[34] = iv[14]; // cameraPos
        fogParams[35] = this.baseDensity;
        fogParams[36] = this.lightDir[0]; fogParams[37] = this.lightDir[1]; fogParams[38] = this.lightDir[2];
        fogParams[39] = this.heightFalloff;
        fogParams[40] = this.lightColor[0]; fogParams[41] = this.lightColor[1]; fogParams[42] = this.lightColor[2];
        fogParams[43] = this.extinctionCoeff;
        fogParams[44] = this.windDir[0] * time; fogParams[45] = this.windDir[1] * time; fogParams[46] = this.windDir[2] * time;
        fogParams[47] = this.anisotropy;
        fogParams[48] = grid.near;                                          // gridNear
        fogParams[49] = grid.far;                                           // gridFar
        fogParams[50] = time;
        new Uint32Array(fogParams.buffer, 204, 1)[0] = grid.gridW;
        new Uint32Array(fogParams.buffer, 208, 1)[0] = grid.gridH;
        new Uint32Array(fogParams.buffer, 212, 1)[0] = grid.gridD;
        fogParams[54] = camera.near;                                        // cameraNear
        fogParams[55] = camera.far;                                         // cameraFar

        device.queue.writeBuffer(this._fogParamsBuffer!, 0, fogParams.buffer as ArrayBuffer);

        // ── Update composite params ──
        const compositeParams = new Float32Array(8);
        compositeParams[0] = camera.near;                                   // cameraNear
        compositeParams[1] = camera.far;                                    // cameraFar
        compositeParams[2] = grid.near;                                     // gridNear
        compositeParams[3] = grid.far;                                      // gridFar
        compositeParams[4] = grid.gridD;                                    // gridD
        compositeParams[5] = width;
        compositeParams[6] = height;
        device.queue.writeBuffer(this._compositeParamsBuffer!, 0, compositeParams.buffer as ArrayBuffer);

        // Rebuild composite bind group on texture change (ping-pong)
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput) {
            this._buildCompositeBG(input, depth, output);
        }

        // ── Pass 1: Fog injection ──
        const injectPass = commandEncoder.beginComputePass({ label: 'VolumetricFog/Inject' });
        injectPass.setPipeline(this._injectPipeline!);
        injectPass.setBindGroup(0, this._injectBG!);
        injectPass.dispatchWorkgroups(
            Math.ceil(grid.gridW / 4),
            Math.ceil(grid.gridH / 4),
            Math.ceil(grid.gridD / 4)
        );
        injectPass.end();

        // ── Pass 2: Temporal reprojection blend ──
        grid.temporalBlend(
            commandEncoder,
            this._invVP as unknown as Float32Array,
            vp as unknown as Float32Array,
            camera.near,
            camera.far
        );

        // ── Pass 3: Front-to-back accumulation ──
        grid.accumulate(commandEncoder);

        // ── Pass 4: Composite fog onto scene ──
        const compositePass = commandEncoder.beginComputePass({ label: 'VolumetricFog/Composite' });
        compositePass.setPipeline(this._compositePipeline!);
        compositePass.setBindGroup(0, this._compositeBG!);
        compositePass.dispatchWorkgroups(
            Math.ceil(width / 8),
            Math.ceil(height / 8)
        );
        compositePass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // Composite params updated every frame. Bind group rebuilt on texture change in render().
    }

    destroy(): void {
        this._fogParamsBuffer?.destroy();
        this._compositeParamsBuffer?.destroy();
        this._fogParamsBuffer = null;
        this._compositeParamsBuffer = null;
        this._injectPipeline = null;
        this._compositePipeline = null;
        this._injectBG = null;
        this._compositeBG = null;
    }
}

export { VolumetricFogEffect };
