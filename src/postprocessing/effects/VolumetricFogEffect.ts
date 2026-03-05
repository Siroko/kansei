import { Camera } from '../../cameras/Camera';
import { DirectionalLight } from '../../lights/DirectionalLight';
import { PointLight } from '../../lights/PointLight';
import { FroxelGrid } from '../../froxels/FroxelGrid';
import { ShadowMap } from '../../shadows/ShadowMap';
import { CubeMapShadowMap } from '../../shadows/CubeMapShadowMap';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { mat4 } from 'gl-matrix';

export interface VolumetricFogOptions {
    froxelGrid: FroxelGrid;
    shadowMap: ShadowMap;
    cubeMapShadowMap?: CubeMapShadowMap;
    baseDensity?: number;
    heightFalloff?: number;
    extinctionCoeff?: number;
    anisotropy?: number;
    windDirection?: [number, number, number];
}

// Byte sizes per storage-buffer element (must match WGSL struct layout)
const DIR_LIGHT_STRIDE  = 96;   // vec3f+f32 + vec3f+f32 + mat4x4f
const POINT_LIGHT_STRIDE = 32;  // vec3f+f32 + vec3f+u32

class VolumetricFogEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _froxelGrid: FroxelGrid;
    private _shadowMap: ShadowMap;
    private _cubeMapShadowMap: CubeMapShadowMap | null;

    baseDensity: number;
    heightFalloff: number;
    extinctionCoeff: number;
    anisotropy: number;
    windDir: [number, number, number];

    // Injection pass
    private _injectPipeline: GPUComputePipeline | null = null;
    private _injectBGL: GPUBindGroupLayout | null = null;
    private _injectBG: GPUBindGroup | null = null;
    private _fogParamsBuffer: GPUBuffer | null = null;

    // Light storage buffers
    private _dirLightsBuffer: GPUBuffer | null = null;
    private _pointLightsBuffer: GPUBuffer | null = null;
    private _dirLightsCapacity = 0;
    private _pointLightsCapacity = 0;
    private _numDirLights = 0;
    private _numPointLights = 0;
    private _injectBGDirty = true;

    // Point shadow atlas
    private _pointShadowAtlasView: GPUTextureView | null = null;
    private _dummyPointShadowTex: GPUTexture | null = null;

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
        this._froxelGrid     = options.froxelGrid;
        this._shadowMap      = options.shadowMap;
        this._cubeMapShadowMap = options.cubeMapShadowMap ?? null;
        this.baseDensity     = options.baseDensity ?? 0.02;
        this.heightFalloff   = options.heightFalloff ?? 0.1;
        this.extinctionCoeff = options.extinctionCoeff ?? 1.0;
        this.anisotropy      = options.anisotropy ?? 0.6;
        this.windDir         = options.windDirection ?? [0, 0, 0];
    }

    // ── Injection shader (multi-light) ─────────────────────────────────────

    private static _INJECT_SHADER = FroxelGrid.WGSL_HELPERS + /* wgsl */`

        struct FogParams {
            invViewProj     : mat4x4f,
            cameraPos       : vec3f,
            baseDensity     : f32,
            windOffset      : vec3f,
            heightFalloff   : f32,
            gridNear        : f32,
            gridFar         : f32,
            time            : f32,
            gridW           : u32,
            gridH           : u32,
            gridD           : u32,
            cameraNear      : f32,
            cameraFar       : f32,
            extinctionCoeff : f32,
            anisotropy      : f32,
            numDirLights    : u32,
            numPointLights  : u32,
        }

        struct DirLightData {
            direction     : vec3f,
            _pad0         : f32,
            color         : vec3f,
            _pad1         : f32,
            lightViewProj : mat4x4f,
        }

        struct PointLightData {
            position  : vec3f,
            radius    : f32,
            color     : vec3f,
            atlasBase : u32,
        }

        @group(0) @binding(0) var scatterExtTex    : texture_storage_3d<rgba16float, write>;
        @group(0) @binding(1) var shadowDepthTex   : texture_depth_2d;
        @group(0) @binding(2) var<uniform> params  : FogParams;
        @group(0) @binding(3) var<storage, read> dirLights : array<DirLightData>;
        @group(0) @binding(4) var<storage, read> ptLights  : array<PointLightData>;
        @group(0) @binding(5) var pointShadowAtlas : texture_2d_array<f32>;

        fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
            let g2 = g * g;
            return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
        }

        fn smoothFalloff(dist: f32, radius: f32) -> f32 {
            let r = clamp(dist / radius, 0.0, 1.0);
            let f = 1.0 - r * r;
            return f * f;
        }

        fn dirShadowLookup(worldPos: vec3f, lvp: mat4x4f) -> f32 {
            let lightClip = lvp * vec4f(worldPos, 1.0);
            let lightNDC  = lightClip.xyz / lightClip.w;
            let shadowUV  = vec2f(lightNDC.x * 0.5 + 0.5, 1.0 - (lightNDC.y * 0.5 + 0.5));

            if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0) {
                return 1.0;
            }

            let shadowDim = textureDimensions(shadowDepthTex, 0);
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
            return pcf * 0.25;
        }

        fn samplePointShadow(worldPos: vec3f, lightPos: vec3f, radius: f32, atlasBase: u32) -> f32 {
            let dir = worldPos - lightPos;
            let dist = length(dir);
            let ad = abs(dir);

            var faceIdx: u32;
            var faceUV: vec2f;
            var ma: f32;

            // Face UV mapping matches lookAt conventions with WebGPU Y-down framebuffer
            if (ad.x >= ad.y && ad.x >= ad.z) {
                ma = ad.x;
                if (dir.x > 0.0) {
                    faceIdx = 0u;
                    faceUV = vec2f(-dir.z, dir.y);
                } else {
                    faceIdx = 1u;
                    faceUV = vec2f(dir.z, dir.y);
                }
            } else if (ad.y >= ad.x && ad.y >= ad.z) {
                ma = ad.y;
                if (dir.y > 0.0) {
                    faceIdx = 2u;
                    faceUV = vec2f(dir.x, -dir.z);
                } else {
                    faceIdx = 3u;
                    faceUV = vec2f(dir.x, dir.z);
                }
            } else {
                ma = ad.z;
                if (dir.z > 0.0) {
                    faceIdx = 4u;
                    faceUV = vec2f(dir.x, dir.y);
                } else {
                    faceIdx = 5u;
                    faceUV = vec2f(-dir.x, dir.y);
                }
            }

            let uv = faceUV / (2.0 * ma) + 0.5;
            let layer = i32(atlasBase + faceIdx);
            let texDim = textureDimensions(pointShadowAtlas, 0);
            let tc = clamp(vec2i(uv * vec2f(texDim)), vec2i(0), vec2i(texDim) - 1);
            let storedDist = textureLoad(pointShadowAtlas, tc, layer, 0).r;
            let bias = 0.05 + dist * 0.002;
            return select(0.0, 1.0, dist <= storedDist + bias);
        }

        @compute @workgroup_size(4, 4, 4)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            if (gid.x >= params.gridW || gid.y >= params.gridH || gid.z >= params.gridD) { return; }

            let linearD = sliceDepth(f32(gid.z) + 0.5, params.gridNear, params.gridFar, f32(params.gridD));

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

            let samplePos = worldPos + params.windOffset;
            let density = params.baseDensity * exp(-params.heightFalloff * max(samplePos.y, 0.0));
            let extinction = density * params.extinctionCoeff;

            let viewDir = normalize(worldPos - params.cameraPos);
            var totalScatter = vec3f(0.0);

            // Directional lights
            for (var di = 0u; di < params.numDirLights; di++) {
                let dl = dirLights[di];
                let visibility = dirShadowLookup(worldPos, dl.lightViewProj);
                let phase = henyeyGreenstein(dot(viewDir, -normalize(dl.direction)), params.anisotropy);
                totalScatter += density * dl.color * visibility * phase;
            }

            // Point lights
            for (var pi = 0u; pi < params.numPointLights; pi++) {
                let pl = ptLights[pi];
                let dist = length(pl.position - worldPos);
                if (dist > pl.radius) { continue; }

                let attenuation = smoothFalloff(dist, pl.radius);
                let visibility = samplePointShadow(worldPos, pl.position, pl.radius, pl.atlasBase);
                let phase = henyeyGreenstein(dot(viewDir, normalize(pl.position - worldPos)), params.anisotropy);
                totalScatter += density * pl.color * attenuation * visibility * phase;
            }

            textureStore(scatterExtTex, gid, vec4f(totalScatter, extinction));
        }
    `;

    // ── Composite shader (unchanged) ──────────────────────────────────────

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

            let linearDepth = 2.0 * cp.cameraNear * cp.cameraFar
                            / ((cp.cameraFar + cp.cameraNear) - depth * (cp.cameraFar - cp.cameraNear));

            let sliceFloat = depthToSlice(linearDepth, cp.gridNear, cp.gridFar, cp.gridD);

            let jitter = screenHash(vec2f(coord)) - 0.5;
            let jitteredSlice = clamp((sliceFloat + jitter) / cp.gridD, 0.0, 1.0);

            let uv = vec2f(f32(coord.x) / cp.screenWidth, f32(coord.y) / cp.screenHeight);
            let gridUV = vec3f(uv.x, uv.y, jitteredSlice);

            let fogData = textureSampleLevel(accumTex, accumSampler, gridUV, 0.0);
            let accLight      = fogData.rgb;
            let transmittance = fogData.a;

            let finalColor = sceneColor.rgb * transmittance + accLight;
            textureStore(outputTex, coord, vec4f(finalColor, sceneColor.a));
        }
    `;

    // ── PostProcessingEffect interface ────────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._fogParamsBuffer = device.createBuffer({
            label: 'VolumetricFog/FogParams',
            size: 160,
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

        // Initial light storage buffers (minimum 1 element each)
        this._ensureLightBuffers(1, 1);

        // Point shadow atlas view
        if (this._cubeMapShadowMap) {
            this._pointShadowAtlasView = this._cubeMapShadowMap.distanceTexture.createView({
                dimension: '2d-array',
            });
        } else {
            this._dummyPointShadowTex = device.createTexture({
                label: 'VolumetricFog/DummyPointShadow',
                size: [1, 1, 6],
                format: 'r32float',
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            this._pointShadowAtlasView = this._dummyPointShadowTex.createView({
                dimension: '2d-array',
            });
        }

        // ── Injection pipeline ──
        const injectModule = device.createShaderModule({
            label: 'VolumetricFog/InjectShader',
            code: VolumetricFogEffect._INJECT_SHADER,
        });

        this._injectBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/Inject BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
            ],
        });

        this._injectPipeline = device.createComputePipeline({
            label: 'VolumetricFog/InjectPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._injectBGL] }),
            compute: { module: injectModule, entryPoint: 'main' },
        });

        this._rebuildInjectBG();

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

    private _ensureLightBuffers(dirCount: number, pointCount: number): void {
        const device = this._device!;
        const dirSize  = Math.max(dirCount, 1) * DIR_LIGHT_STRIDE;
        const pointSize = Math.max(pointCount, 1) * POINT_LIGHT_STRIDE;

        if (dirSize > this._dirLightsCapacity) {
            this._dirLightsBuffer?.destroy();
            this._dirLightsBuffer = device.createBuffer({
                label: 'VolumetricFog/DirLights',
                size: dirSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._dirLightsCapacity = dirSize;
            this._injectBGDirty = true;
        }

        if (pointSize > this._pointLightsCapacity) {
            this._pointLightsBuffer?.destroy();
            this._pointLightsBuffer = device.createBuffer({
                label: 'VolumetricFog/PointLights',
                size: pointSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._pointLightsCapacity = pointSize;
            this._injectBGDirty = true;
        }
    }

    private _rebuildInjectBG(): void {
        if (!this._injectBGL || !this._device) return;
        this._injectBG = this._device.createBindGroup({
            label: 'VolumetricFog/Inject BG',
            layout: this._injectBGL,
            entries: [
                { binding: 0, resource: this._froxelGrid.scatterExtinctionTex.createView() },
                { binding: 1, resource: this._shadowMap.depthTexture.createView() },
                { binding: 2, resource: { buffer: this._fogParamsBuffer! } },
                { binding: 3, resource: { buffer: this._dirLightsBuffer! } },
                { binding: 4, resource: { buffer: this._pointLightsBuffer! } },
                { binding: 5, resource: this._pointShadowAtlasView! },
            ],
        });
        this._injectBGDirty = false;
    }

    /**
     * Pack light data into storage buffers. Call each frame before render().
     */
    updateLights(dirLights: readonly DirectionalLight[], pointLights: readonly PointLight[]): void {
        if (!this._device) return;

        // Filter volumetric directional lights
        const volDirLights: DirectionalLight[] = [];
        for (let i = 0; i < dirLights.length; i++) {
            if (dirLights[i].volumetric) volDirLights.push(dirLights[i]);
        }
        this._numDirLights = volDirLights.length;
        // All point lights are packed (matching CubeMapShadowMap order); non-volumetric get zero color
        this._numPointLights = pointLights.length;

        this._ensureLightBuffers(this._numDirLights, this._numPointLights);
        if (this._injectBGDirty) this._rebuildInjectBG();

        // Pack DirLightData (96 bytes = 24 floats each)
        if (this._numDirLights > 0) {
            const data = new Float32Array(this._numDirLights * 24);
            for (let i = 0; i < this._numDirLights; i++) {
                const light = volDirLights[i];
                const ec = light.effectiveColor;
                const off = i * 24;
                data[off]     = light.direction[0];
                data[off + 1] = light.direction[1];
                data[off + 2] = light.direction[2];
                // off+3 = pad
                data[off + 4] = ec[0];
                data[off + 5] = ec[1];
                data[off + 6] = ec[2];
                // off+7 = pad
                data.set(this._shadowMap.lightViewProjMatrix, off + 8);
            }
            this._device.queue.writeBuffer(this._dirLightsBuffer!, 0, data.buffer as ArrayBuffer);
        }

        // Pack PointLightData (32 bytes = 8 floats each)
        if (this._numPointLights > 0) {
            const data = new Float32Array(this._numPointLights * 8);
            const uintView = new Uint32Array(data.buffer);
            for (let i = 0; i < this._numPointLights; i++) {
                const light = pointLights[i];
                light.updateModelMatrix();
                const wm = light.worldMatrix.internalMat4;
                const off = i * 8;
                data[off]     = wm[12];  // world X
                data[off + 1] = wm[13];  // world Y
                data[off + 2] = wm[14];  // world Z
                data[off + 3] = light.radius;
                if (light.volumetric) {
                    const ec = light.effectiveColor;
                    data[off + 4] = ec[0];
                    data[off + 5] = ec[1];
                    data[off + 6] = ec[2];
                }
                // non-volumetric: color stays (0,0,0) — scatter contribution is zero
                uintView[off + 7] = i * 6; // atlasBase
            }
            this._device.queue.writeBuffer(this._pointLightsBuffer!, 0, data.buffer as ArrayBuffer);
        }
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
        mat4.invert(this._invVP, vp);

        const iv = camera.inverseViewMatrix.internalMat4;

        const fogParams = new Float32Array(40); // 160 bytes
        fogParams.set(this._invVP as unknown as Float32Array, 0);                  // invViewProj
        fogParams[16] = iv[12]; fogParams[17] = iv[13]; fogParams[18] = iv[14];    // cameraPos
        fogParams[19] = this.baseDensity;
        fogParams[20] = this.windDir[0] * time;
        fogParams[21] = this.windDir[1] * time;
        fogParams[22] = this.windDir[2] * time;
        fogParams[23] = this.heightFalloff;
        fogParams[24] = grid.near;                                                 // gridNear
        fogParams[25] = grid.far;                                                  // gridFar
        fogParams[26] = time;
        new Uint32Array(fogParams.buffer, 108, 1)[0] = grid.gridW;                 // gridW
        new Uint32Array(fogParams.buffer, 112, 1)[0] = grid.gridH;                 // gridH
        new Uint32Array(fogParams.buffer, 116, 1)[0] = grid.gridD;                 // gridD
        fogParams[30] = camera.near;                                               // cameraNear
        fogParams[31] = camera.far;                                                // cameraFar
        fogParams[32] = this.extinctionCoeff;
        fogParams[33] = this.anisotropy;
        new Uint32Array(fogParams.buffer, 136, 1)[0] = this._numDirLights;
        new Uint32Array(fogParams.buffer, 140, 1)[0] = this._numPointLights;

        device.queue.writeBuffer(this._fogParamsBuffer!, 0, fogParams.buffer as ArrayBuffer);

        // ── Update composite params ──
        const compositeParams = new Float32Array(8);
        compositeParams[0] = camera.near;
        compositeParams[1] = camera.far;
        compositeParams[2] = grid.near;
        compositeParams[3] = grid.far;
        compositeParams[4] = grid.gridD;
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
        this._dirLightsBuffer?.destroy();
        this._pointLightsBuffer?.destroy();
        this._dummyPointShadowTex?.destroy();
        this._fogParamsBuffer = null;
        this._compositeParamsBuffer = null;
        this._dirLightsBuffer = null;
        this._pointLightsBuffer = null;
        this._injectPipeline = null;
        this._compositePipeline = null;
        this._injectBG = null;
        this._compositeBG = null;
    }
}

export { VolumetricFogEffect };
