import { mat4 } from 'gl-matrix';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { CLOUDS_COMPOSITE_SOURCE, CLOUDS_MARCH_SOURCE, CLOUDS_NOISE_SOURCE } from '../../atmosphere/AtmosphereWGSL';
import {
    CLOUD_MAP_SIZE, CLOUD_SHADOW_PARAMS_BYTES, CLOUD_SHADOW_SIZE, SkyAtmosphere, SkyAtmosphereBindings,
} from '../../atmosphere/SkyAtmosphere';
import { gpuPass } from '../../profiling/Profiler';

// Volumetric clouds over a SkyAtmosphere, as the Rust engine's `postprocessing/effects/clouds.rs`
// on the same WGSL (`clouds_noise.wgsl`, `clouds_march.wgsl`, `clouds_composite.wgsl`).

type Vec3 = [number, number, number];

/** Texels of the 3D shape noise per side, of the 3D detail noise, and of the 2D weather map. */
const SHAPE_SIZE = 128;
const DETAIL_SIZE = 32;
const WEATHER_SIZE = 256;
/** Bytes of the WGSL `CloudParams` (clouds_march.wgsl). */
const CLOUD_PARAMS_BYTES = 160;

/** A layer of cloud around the planet, between two altitudes. Distances are in metres. */
export interface CloudLayer {
    /** Altitude of the layer's base, above the planet's surface. */
    bottomM: number;
    /** Altitude of its top. */
    topM: number;
    /** How much of the sky it covers: 0 clear, 1 overcast. */
    coverage: number;
    /** 0 flat stratus, 0.5 stratocumulus, 1 towering cumulus (the weather map varies it). */
    cloudType: number;
    /** Extinction per metre at full density (real clouds: about 0.02 to 0.1). */
    extinction: number;
    /** Single-scattering albedo (water droplets: 0.99 and above). */
    albedo: Vec3;
    /** Wind, metres per second: the clouds drift by `wind * time`. */
    wind: Vec3;
    /** Size of one repeat of the shape noise (the billows are a quarter of it). */
    shapeSizeM: number;
    /** Size of one repeat of the detail noise, which frays the edges. */
    detailSizeM: number;
    /** Size of one repeat of the weather map, which decides where clouds form. */
    weatherSizeM: number;
}

export function defaultCloudLayer(): CloudLayer {
    return {
        bottomM: 1500,
        topM: 4000,
        coverage: 0.5,
        cloudType: 0.7,
        extinction: 0.05,
        albedo: [0.99, 0.99, 0.99],
        wind: [10, 0, 3],
        shapeSizeM: 9000,
        detailSizeM: 700,
        weatherSizeM: 40000,
    };
}

export interface VolumetricCloudsOptions {
    layer: CloudLayer;
    /** Resolution of the march relative to the image (0.5: a quarter of the pixels). */
    resolutionScale: number;
    /** Steps along each view ray through the layer, and toward the sun from each step. */
    steps: number;
    lightSteps: number;
    /**
     * Farthest the march goes into the layer, metres (the horizon's clouds beyond it fade into
     * the aerial perspective).
     */
    maxDistanceM: number;
    /** Weight of each new frame in the accumulated clouds (lower: smoother, slower to follow). */
    temporalBlend: number;
}

export function defaultVolumetricCloudsOptions(): VolumetricCloudsOptions {
    return { layer: defaultCloudLayer(), resolutionScale: 0.5, steps: 64, lightSteps: 6, maxDistanceM: 60000, temporalBlend: 0.1 };
}

/**
 * Presets of the clouds' cost (`VolumetricCloudsEffect.setQuality`), from the march's resolution
 * and steps and how much of the cloud map for the sky lighting is rewritten each frame. Rust's
 * costs per frame at 1440x602 under an overcast (M4 Pro): about 0.3 ms low, 0.8-0.9 ms medium,
 * 2 ms high. `VolumetricCloudsEffect.enabled = false` costs nothing.
 */
export enum CloudQuality {
    /** A quarter of the resolution each way, 32 steps (4 toward the sun); the cloud map 16 steps, a quarter of its rows a frame. */
    Low = 'low',
    /** Half resolution, 64 steps (6); the cloud map 24 steps, half its rows a frame. The default. */
    Medium = 'medium',
    /** Three quarters of the resolution, 96 steps (8); the cloud map 32 steps, all of it. */
    High = 'high',
}

/** (resolution scale, steps, light steps, cloud-map steps, cloud-map interleave) */
const QUALITY_SETTINGS: Record<CloudQuality, [number, number, number, number, number]> = {
    [CloudQuality.Low]: [0.25, 32, 4, 16, 4],
    [CloudQuality.Medium]: [0.5, 64, 6, 24, 2],
    [CloudQuality.High]: [0.75, 96, 8, 32, 1],
};

interface Targets {
    width: number;
    height: number;
    /** Ping-pong: each frame writes one and reads the other as its history. */
    color: [GPUTexture, GPUTexture];
    depth: GPUTexture;
}

interface Gpu {
    params: GPUBuffer;
    march: GPUComputePipeline;
    skyMap: GPUComputePipeline;
    skyMapBG: GPUBindGroup;
    // the shadow map (SkyAtmosphereBindings.cloudShadow), and its parameters staged for it and
    // copied to the bindings after it is written
    shadowMap: GPUComputePipeline;
    shadowMapBG: GPUBindGroup;
    shadowStaging: GPUBuffer;
    composite: GPUComputePipeline;
    shape: GPUTexture;
    detail: GPUTexture;
    weather: GPUTexture;
    noiseSampler: GPUSampler;
    noiseReady: boolean;
    noise: [GPUComputePipeline, GPUComputePipeline, GPUComputePipeline, GPUBindGroup];
    targets: Targets | null;
    // per ping-pong index, with what they were made for
    marchBGs: [GPUBindGroup | null, GPUBindGroup | null];
    marchFor: [GPUTexture | null, GPUTexture | null];
    compositeBGs: [GPUBindGroup | null, GPUBindGroup | null];
    compositeFor: [GPUTexture[], GPUTexture[]];
}

/**
 * Volumetric clouds over a `SkyAtmosphere` (`CloudLayer`): a layer of cloud between two
 * altitudes, shaped by a weather map, a height profile and Perlin-Worley noise, lit by the sun
 * through the atmosphere and through the cloud (with an approximation of multiple scattering)
 * and by the sky, and seen through the atmosphere in front of it. They are marched at a reduced
 * resolution with a jittered start, accumulated over frames by reprojection, and composited over
 * the sky and over any surface they are in front of.
 *
 * Put it right after the `AtmosphereEffect` (before the fogs), and set `time` every frame for
 * the wind. `layer` can change every frame (coverage for the weather of a shot). The clouds are
 * part of the sky: `AtmosphereParams.skyLuminanceFactor` scales the light they send like the
 * sky's.
 *
 * They also light the scene: each frame they march a small map of themselves all around the
 * camera (`SkyAtmosphereBindings.cloudMap`), and the next frame's sky lighting (the SH the
 * materials and fogs take their ambient light from) and environment cubemap see the sky through
 * it, so the ambient light and the reflections are occluded and tinted by the cloud layer. The clouds themselves stay lit by the clear sky
 * above them. `lightsSky = false` leaves the sky lighting clear, as does taking the effect out of
 * the chain. And they shadow it: a map of the layer's transmittance toward the sun under the
 * camera (`SkyAtmosphereBindings.cloudShadow`), which materials read with `CLOUD_SHADOW_WGSL`.
 */
export class VolumetricCloudsEffect extends PostProcessingEffect {
    /**
     * Draw the clouds (default true). Off, the post-processing volume skips them at no cost, and
     * the sky lighting and the shadows are clear from the next frame.
     */
    public enabled = true;
    public layer: CloudLayer;
    /** Whether the clouds occlude and tint the sky lighting and the environment (default true). */
    public lightsSky = true;
    /** Whether the clouds shadow the scene from the sun (`SkyAtmosphereBindings.cloudShadow`, default true). */
    public castsShadows = true;
    /**
     * The side of the shadow map (m), centred under the camera; beyond it there are no cloud
     * shadows. 256 texels, so 16 km gives 62.5 m per texel.
     */
    public shadowSizeM = 16000;
    /**
     * Steps along each ray of the cloud map for the sky lighting (its light steps are at most 3):
     * it only feeds low-frequency light, so it takes fewer than the view.
     */
    public skyMapSteps = 32;
    /** The cloud map rewrites one row in this many each frame (1: all of it), for the clouds that only drift. */
    public skyMapInterleave = 1;
    /** Seconds, drives the wind. */
    public time = 0;
    public resolutionScale: number;
    public steps: number;
    public lightSteps: number;
    public maxDistanceM: number;
    public temporalBlend: number;

    private readonly _sky: SkyAtmosphereBindings;
    private _device: GPUDevice | null = null;
    private _gpu: Gpu | null = null;
    // last frame's view-projection, while there is a history to reproject
    private readonly _prevViewProj = mat4.create();
    private _hasHistory = false;
    private _lastCameraFrame: number | null = null;
    private _frame = 0;
    private readonly _viewProj = mat4.create();
    private readonly _paramsData = new ArrayBuffer(CLOUD_PARAMS_BYTES);
    private readonly _shadowData = new Float32Array(CLOUD_SHADOW_PARAMS_BYTES / 4);

    constructor(sky: SkyAtmosphere, options: Partial<VolumetricCloudsOptions> = {}) {
        super();
        const o = { ...defaultVolumetricCloudsOptions(), ...options };
        this._sky = sky.bindings;
        this.layer = o.layer;
        this.resolutionScale = o.resolutionScale;
        this.steps = o.steps;
        this.lightSteps = o.lightSteps;
        this.maxDistanceM = o.maxDistanceM;
        this.temporalBlend = o.temporalBlend;
    }

    /** Drop the accumulated frames; call on camera cuts. */
    public resetHistory(): void {
        this._hasHistory = false;
    }

    /** Set the march's resolution and steps and the cloud map's work to a preset. */
    public setQuality(quality: CloudQuality): void {
        const [scale, steps, lightSteps, mapSteps, interleave] = QUALITY_SETTINGS[quality];
        this.resolutionScale = scale;
        this.steps = steps;
        this.lightSteps = lightSteps;
        this.skyMapSteps = mapSteps;
        this.skyMapInterleave = interleave;
    }

    isActive(): boolean {
        return this.enabled;
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        if (!this._gpu) this._initGpu(device);
        this.initialized = true;
    }

    private _initGpu(device: GPUDevice): void {
        this._device = device;
        const C = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'uniform' } });
        const texture = (binding: number, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: C, texture: { sampleType: 'float', viewDimension } });
        const unfiltered = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, texture: { sampleType: 'unfilterable-float' } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, texture: { sampleType: 'depth' } });
        const sampler = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, sampler: { type: 'filtering' } });
        const storage = (binding: number, format: GPUTextureFormat, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: C, storageTexture: { access: 'write-only', format, viewDimension } });
        const pipeline = (label: string, code: string, entryPoint: string, layout: GPUBindGroupLayout) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint },
        });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });
        const group = (label: string, layout: GPUBindGroupLayout, entries: [number, GPUBindingResource][]) =>
            device.createBindGroup({ label, layout, entries: entries.map(([binding, resource]) => ({ binding, resource })) });

        // the noise textures, generated once
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
        const shape = device.createTexture({ label: 'Clouds/Shape', size: [SHAPE_SIZE, SHAPE_SIZE, SHAPE_SIZE], dimension: '3d', format: 'rgba8unorm', usage });
        const detail = device.createTexture({ label: 'Clouds/Detail', size: [DETAIL_SIZE, DETAIL_SIZE, DETAIL_SIZE], dimension: '3d', format: 'rgba8unorm', usage });
        const weather = device.createTexture({ label: 'Clouds/Weather', size: [WEATHER_SIZE, WEATHER_SIZE], format: 'rgba8unorm', usage });
        const noiseBGL = bgl('Clouds/NoiseBGL', [storage(0, 'rgba8unorm', '3d'), storage(1, 'rgba8unorm', '3d'), storage(2, 'rgba8unorm')]);
        const noiseModule = device.createShaderModule({ label: 'Clouds/Noise', code: CLOUDS_NOISE_SOURCE });
        const noiseLayout = device.createPipelineLayout({ label: 'Clouds/Noise', bindGroupLayouts: [noiseBGL] });
        const noisePipeline = (entryPoint: string) =>
            device.createComputePipeline({ label: 'Clouds/Noise', layout: noiseLayout, compute: { module: noiseModule, entryPoint } });
        const noiseBG = group('Clouds/NoiseBG', noiseBGL, [[0, shape.createView()], [1, detail.createView()], [2, weather.createView()]]);

        const marchBGL = bgl('Clouds/MarchBGL', [
            uniform(0), uniform(1), texture(2), sampler(3), texture(4, '3d'), texture(5, '3d'), uniform(6), depth(7),
            texture(8, '3d'), texture(9, '3d'), texture(10), sampler(11), texture(12),
            storage(13, 'rgba16float'), storage(14, 'r32float'), uniform(15),
        ]);
        const skyMapBGL = bgl('Clouds/SkyMapBGL', [
            uniform(0), uniform(1), texture(2), sampler(3), uniform(6), texture(8, '3d'), texture(9, '3d'), texture(10), sampler(11),
            uniform(15), texture(16), sampler(17), storage(18, 'rgba16float'),
        ]);
        const shadowMapBGL = bgl('Clouds/ShadowMapBGL', [
            uniform(0), uniform(1), texture(8, '3d'), texture(9, '3d'), texture(10), sampler(11), uniform(15),
            storage(19, 'rgba8unorm'), uniform(20),
        ]);
        const compositeBGL = bgl('Clouds/CompositeBGL', [uniform(0), unfiltered(1), depth(2), unfiltered(3), unfiltered(4), storage(5, 'rgba16float')]);

        const params = device.createBuffer({ label: 'Clouds/Params', size: CLOUD_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const shadowStaging = device.createBuffer({
            label: 'Clouds/ShadowParamsStaging',
            size: CLOUD_SHADOW_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const noiseSampler = device.createSampler({
            label: 'Clouds/NoiseSampler',
            addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat', magFilter: 'linear', minFilter: 'linear',
        });
        const s = this._sky;
        const shapeView = shape.createView(), detailView = detail.createView(), weatherView = weather.createView();
        this._gpu = {
            params,
            march: pipeline('Clouds/March', CLOUDS_MARCH_SOURCE, 'main', marchBGL),
            skyMap: pipeline('Clouds/SkyMap', CLOUDS_MARCH_SOURCE, 'skyMap', skyMapBGL),
            skyMapBG: group('Clouds/SkyMapBG', skyMapBGL, [
                [0, { buffer: s.atmosphere }], [1, { buffer: s.frame }], [2, s.transmittance], [3, s.lutSampler],
                [6, { buffer: s.skyLighting }], [8, shapeView], [9, detailView], [10, weatherView], [11, noiseSampler],
                [15, { buffer: params }], [16, s.skyView], [17, s.skyViewSampler], [18, s.cloudMap],
            ]),
            shadowMap: pipeline('Clouds/ShadowMap', CLOUDS_MARCH_SOURCE, 'shadowMap', shadowMapBGL),
            shadowMapBG: group('Clouds/ShadowMapBG', shadowMapBGL, [
                [0, { buffer: s.atmosphere }], [1, { buffer: s.frame }], [8, shapeView], [9, detailView], [10, weatherView],
                [11, noiseSampler], [15, { buffer: params }], [19, s.cloudShadow], [20, { buffer: shadowStaging }],
            ]),
            shadowStaging,
            composite: pipeline('Clouds/Composite', CLOUDS_COMPOSITE_SOURCE, 'main', compositeBGL),
            shape, detail, weather, noiseSampler,
            noiseReady: false,
            noise: [noisePipeline('shape'), noisePipeline('detail'), noisePipeline('weather'), noiseBG],
            targets: null,
            marchBGs: [null, null],
            marchFor: [null, null],
            compositeBGs: [null, null],
            compositeFor: [[], []],
        };
    }

    private _ensureTargets(width: number, height: number): Targets {
        const gpu = this._gpu!;
        const w = Math.max(Math.ceil(width * this.resolutionScale), 1);
        const h = Math.max(Math.ceil(height * this.resolutionScale), 1);
        if (gpu.targets && gpu.targets.width === w && gpu.targets.height === h) return gpu.targets;
        if (gpu.targets) {
            for (const t of [...gpu.targets.color, gpu.targets.depth]) t.destroy();
        }
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const texture = (label: string, format: GPUTextureFormat) => this._device!.createTexture({ label, size: [w, h], format, usage });
        gpu.targets = {
            width: w,
            height: h,
            color: [texture('Clouds/ColorA', 'rgba16float'), texture('Clouds/ColorB', 'rgba16float')],
            depth: texture('Clouds/Depth', 'r32float'),
        };
        gpu.marchBGs = [null, null];
        gpu.marchFor = [null, null];
        gpu.compositeBGs = [null, null];
        gpu.compositeFor = [[], []];
        this._hasHistory = false;
        return gpu.targets;
    }

    /** The `CloudParams` uniform into `_paramsData`. */
    private _writeParams(viewProj: mat4, t: Targets): void {
        const f = new Float32Array(this._paramsData);
        const u = new Uint32Array(this._paramsData);
        const l = this.layer;
        const km = (m: number) => m * 0.001;
        const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);
        const wind = km(this.time);
        f.set(this._hasHistory ? this._prevViewProj : viewProj, 0);
        f.set([l.wind[0] * wind, l.wind[1] * wind, l.wind[2] * wind, clamp01(l.coverage)], 16);
        f[20] = km(l.bottomM);
        f[21] = km(Math.max(l.topM, l.bottomM + 1));
        f[22] = Math.max(l.extinction, 0) * 1000;
        f[23] = clamp01(l.cloudType);
        f.set(l.albedo, 24);
        f[27] = 1 / km(Math.max(l.shapeSizeM, 1));
        f[28] = 1 / km(Math.max(l.detailSizeM, 1));
        f[29] = 1 / km(Math.max(l.weatherSizeM, 1));
        f[30] = km(Math.max(this.maxDistanceM, 1));
        u[31] = this._frame;
        u[32] = t.width;
        u[33] = t.height;
        u[34] = this._hasHistory ? 1 : 0;
        u[35] = Math.max(this.steps, 1);
        u[36] = Math.max(this.lightSteps, 1);
        f[37] = Math.min(Math.max(this.temporalBlend, 0.01), 1);
        u[38] = Math.max(this.skyMapSteps, 1);
        u[39] = Math.max(this.skyMapInterleave, 1);
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
    ): void {
        if (!this._gpu) return;
        const device = this._device!;
        const queue = device.queue;
        const t = this._ensureTargets(width, height);
        // a gap in the camera's frames (the clouds were off, or a cut) invalidates the history
        const cameraFrame = camera.frame;
        const last = this._lastCameraFrame;
        if (last !== null && cameraFrame !== last && cameraFrame !== ((last + 1) >>> 0)) {
            this._hasHistory = false;
        }
        this._lastCameraFrame = cameraFrame;
        const gpu = this._gpu;
        if (!gpu.noiseReady) {
            const [shape, detail, weather, bg] = gpu.noise;
            const pass = commandEncoder.beginComputePass({ label: 'Clouds/Noise', timestampWrites: gpuPass('Clouds/Noise') });
            pass.setBindGroup(0, bg);
            pass.setPipeline(shape);
            pass.dispatchWorkgroups(SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
            pass.setPipeline(detail);
            pass.dispatchWorkgroups(DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
            pass.setPipeline(weather);
            pass.dispatchWorkgroups(WEATHER_SIZE / 8, WEATHER_SIZE / 8);
            pass.end();
            gpu.noiseReady = true;
        }
        const viewProj = camera.viewProjection(this._viewProj);
        this._writeParams(viewProj, t);
        queue.writeBuffer(gpu.params, 0, this._paramsData);
        const current = this._frame % 2;
        this._frame = (this._frame + 1) >>> 0;
        mat4.copy(this._prevViewProj, viewProj);
        this._hasHistory = true;

        const s = this._sky;
        if (gpu.marchFor[current] !== depth || !gpu.marchBGs[current]) {
            const resources: GPUBindingResource[] = [
                { buffer: s.atmosphere }, { buffer: s.frame }, s.transmittance, s.lutSampler, s.apScattering, s.apTransmittance,
                { buffer: s.skyLighting }, depth.createView(), gpu.shape.createView(), gpu.detail.createView(), gpu.weather.createView(),
                gpu.noiseSampler, t.color[1 - current].createView(), t.color[current].createView(), t.depth.createView(), { buffer: gpu.params },
            ];
            gpu.marchBGs[current] = device.createBindGroup({
                label: 'Clouds/MarchBG',
                layout: gpu.march.getBindGroupLayout(0),
                entries: resources.map((resource, binding) => ({ binding, resource })),
            });
            gpu.marchFor[current] = depth;
        }
        const bound = gpu.compositeFor[current];
        if (!gpu.compositeBGs[current] || bound[0] !== input || bound[1] !== depth || bound[2] !== output) {
            const resources: GPUBindingResource[] = [
                { buffer: s.frame }, input.createView(), depth.createView(), t.color[current].createView(), t.depth.createView(), output.createView(),
            ];
            gpu.compositeBGs[current] = device.createBindGroup({
                label: 'Clouds/CompositeBG',
                layout: gpu.composite.getBindGroupLayout(0),
                entries: resources.map((resource, binding) => ({ binding, resource })),
            });
            gpu.compositeFor[current] = [input, depth, output];
        }
        const sun = s.sunDirection;
        const shadows = this.castsShadows && sun[1] > 0.01;
        if (this.castsShadows) {
            const inv = camera.inverseViewMatrix.internalMat4;
            const size = Math.max(this.shadowSizeM, 100);
            // centred under the camera, snapped to whole texels so the shadows don't crawl
            const texel = size / CLOUD_SHADOW_SIZE;
            const d = this._shadowData;
            d[0] = Math.round(inv[12] / texel) * texel;
            d[1] = Math.round(inv[14] / texel) * texel;
            d[2] = 1 / size;
            d[3] = inv[13];
            d.set(sun, 4);
            d[7] = shadows ? 1 : 0;
            queue.writeBuffer(gpu.shadowStaging, 0, d);
        }

        const pass = commandEncoder.beginComputePass({ label: 'Clouds', timestampWrites: gpuPass('Clouds') });
        pass.setPipeline(gpu.march);
        pass.setBindGroup(0, gpu.marchBGs[current]!);
        pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
        // the clouds all around, for the sky lighting and the environment of the next frame
        if (this.lightsSky) {
            pass.setPipeline(gpu.skyMap);
            pass.setBindGroup(0, gpu.skyMapBG);
            pass.dispatchWorkgroups(Math.ceil(CLOUD_MAP_SIZE[0] / 8), Math.ceil(CLOUD_MAP_SIZE[1] / 8));
            s.cloudMapFrame = (camera.frame + 1) >>> 0;
        }
        // the shadow on the scene from the sun, for the next frame's materials
        if (shadows) {
            pass.setPipeline(gpu.shadowMap);
            pass.setBindGroup(0, gpu.shadowMapBG);
            pass.dispatchWorkgroups(Math.ceil(CLOUD_SHADOW_SIZE / 8), Math.ceil(CLOUD_SHADOW_SIZE / 8));
        }
        pass.setPipeline(gpu.composite);
        pass.setBindGroup(0, gpu.compositeBGs[current]!);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        if (this.castsShadows) {
            // the parameters of the map just written (or its shadows off, with the sun down)
            commandEncoder.copyBufferToBuffer(gpu.shadowStaging, 0, s.cloudShadowParams, 0, CLOUD_SHADOW_PARAMS_BYTES);
            s.cloudShadowFrame = (camera.frame + 1) >>> 0;
        }
    }

    resize(_width: number, _height: number, _gbuffer: GBuffer): void {
        // the GBuffer's textures are new: the bind groups follow the next render's
        if (!this._gpu) return;
        this._gpu.marchBGs = [null, null];
        this._gpu.compositeBGs = [null, null];
    }

    destroy(): void {
        const gpu = this._gpu;
        if (gpu) {
            for (const texture of [gpu.shape, gpu.detail, gpu.weather, ...(gpu.targets ? [...gpu.targets.color, gpu.targets.depth] : [])]) texture.destroy();
            gpu.params.destroy();
            gpu.shadowStaging.destroy();
        }
        this._gpu = null;
        this._device = null;
        this.initialized = false;
    }
}
