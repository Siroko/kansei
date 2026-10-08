import { mat4 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { GBuffer } from '../postprocessing/GBuffer';
import { PostProcessingEffect } from '../postprocessing/PostProcessingEffect';
import { gpuPass } from '../profiling/Profiler';
import { VoxelVolume } from '../gi/VoxelVolume';
import { VoxelClipmap, clipmapEntries, clipmapLayoutEntries } from '../gi/VoxelClipmap';
import { DirectionalLight } from '../lights/DirectionalLight';
import { PointLight } from '../lights/PointLight';
import { AreaLight } from '../lights/AreaLight';
import { CascadedShadowSource, ComputeShadows } from '../shadows/ComputeShadows';
import { ShadowMap } from '../shadows/ShadowMap';
import { CubeMapShadowMap } from '../shadows/CubeMapShadowMap';
import { SpotShadowAtlas } from '../shadows/SpotShadowAtlas';
import { RtGrid, RtGridHandle } from './RtGrid';
import { RT_DEFAULT_COVERED_WGSL, RT_GI_COMPOSITE_WGSL, RT_GI_SVGF_WGSL, rtGiTraceWgsl } from './RtWGSL';

/**
 * What the rays are traced at: one ray for every pixel (`full`), or one for each 2 x 2 block
 * (`half`, a quarter of the rays), denoised there and upsampled by depth and normal: about a
 * quarter of the cost, what real-time scenes use. Rust: `rt::RtGiResolution`.
 */
export type RtGiResolution = 'full' | 'half';
/**
 * What lights a ray's hit: the voxels' radiance there (`voxels`: the voxels as the surface cache,
 * no shadow rays, cheapest, but light leaks into contacts the voxels are too coarse for), or its
 * exact direct light (`direct`: the lights passed to `updateLights` and `setSpotLights`, shadowed
 * as `RtGiShadows` says) plus the indirect light round it from one voxel cone in a
 * cosine-distributed direction: the further bounces. Rust: `rt::RtGiHitLighting`.
 */
export type RtGiHitLighting = 'voxels' | 'direct';
/**
 * What shadows a hit's direct light: shadow rays through the grid (`rays`: a spot light's toward
 * a point of its disk, a soft shadow over frames; past the grid's box the directional shadow map),
 * or the renderer's shadow maps (`maps`). Rust: `rt::RtGiShadows`.
 */
export type RtGiShadows = 'rays' | 'maps';
/**
 * The denoiser: the raw 1 spp signal (`off`), SVGF's temporal accumulation alone (`temporal`), or
 * SVGF (`svgf`, Schied et al. 2017: temporal accumulation, a variance estimate, then
 * `atrousIterations` of the edge-avoiding a-trous wavelet). Rust: `rt::RtGiDenoise`.
 */
export type RtGiDenoise = 'off' | 'temporal' | 'svgf';
/**
 * The a-trous wavelet's kernel: 3 x 3 taps (`3x3`, 1-2-1: half the cost of the 5 x 5, and as good
 * on the scenes measured) or 5 x 5 (`5x5`, the B3 spline, SVGF's). Rust: `rt::RtGiKernel`.
 */
export type RtGiKernel = '3x3' | '5x5';
/**
 * What the rays compute: one bounce traced through the grid, the voxels past it (`hybrid`), or a
 * path tracer through the grid alone (`reference`: `referenceBounces` vertices, the direct light at
 * each, Russian roulette past the second, the same voxel cone past the grid's box). With
 * `accumulate`, the reference is the ground truth the hybrid converges toward: a debug and quality
 * tool, several times the hybrid's cost. Rust: `rt::RtGiMode`.
 */
export type RtGiMode = 'hybrid' | 'reference';
/**
 * What the screen shows: the lit image with the GI (`lit`), the light the GI adds alone
 * (`indirect`: albedo times the signal), the signal (`signal`: the incoming light's irradiance
 * over pi, no albedo), the variance the wavelet starts from relative to the signal (`variance`),
 * the frames of history the temporal pass holds of `maxHistory` (`history`), or the rays' cost
 * (`cost`: cells visited and triangles tested, blue to red over 0-400); the last three scaled by
 * `heatScale`. Rust: `rt::RtGiView`.
 */
export type RtGiView = 'lit' | 'indirect' | 'signal' | 'variance' | 'history' | 'cost';

/** The settings' names, in the order of the Rust enums' variants (what the WGSL and the accumulation key see). */
export const RT_GI_RESOLUTIONS: readonly RtGiResolution[] = ['full', 'half'];
export const RT_GI_HIT_LIGHTINGS: readonly RtGiHitLighting[] = ['voxels', 'direct'];
export const RT_GI_SHADOWS: readonly RtGiShadows[] = ['rays', 'maps'];
export const RT_GI_DENOISERS: readonly RtGiDenoise[] = ['off', 'temporal', 'svgf'];
export const RT_GI_KERNELS: readonly RtGiKernel[] = ['3x3', '5x5'];
export const RT_GI_MODES: readonly RtGiMode[] = ['hybrid', 'reference'];
export const RT_GI_VIEWS: readonly RtGiView[] = ['lit', 'indirect', 'signal', 'variance', 'history', 'cost'];

/**
 * What `RtDiffuseGiEffect` sets up. The defaults are the measured real-time ones: half resolution,
 * exact direct light at hits with ray shadows, one anisotropic voxel cone at the hit, SVGF with
 * five 3 x 3 iterations. Rust: `rt::RtDiffuseGiOptions`.
 */
export interface RtDiffuseGiOptions {
    /** Default `half`. */
    resolution?: RtGiResolution;
    /** Default `direct`. */
    hitLighting?: RtGiHitLighting;
    /** Default `rays`. */
    shadows?: RtGiShadows;
    /** Default `svgf`. */
    denoise?: RtGiDenoise;
    /** Default `3x3`. */
    kernel?: RtGiKernel;
    /** Iterations of the wavelet (steps 1, 2, 4 ...; at most 8). Default 5. */
    atrousIterations?: number;
    /** Metres a ray looks, through the grid then the voxels. Default 1e4. */
    maxDistance?: number;
    /**
     * Metres a ray walks the grid before the voxel cone takes over (and a sun's shadow ray before
     * the shadow map); 0 (the default) walks the grid's whole box. In a large outdoor box 4-8 m
     * halves the trace for some energy lost under thin occluders the voxels are too coarse for.
     */
    nearDistance?: number;
    /** The voxel cone a ray takes past the grid (tan of its half-angle; default 0.1) and its steps (48). */
    coneTan?: number;
    coneSteps?: number;
    /**
     * The cone a hit's indirect light is read through (`direct` hit lighting; default 0.577, 16
     * steps); 0 steps lights hits by their direct light alone (one bounce).
     */
    hitConeTan?: number;
    hitConeSteps?: number;
    /** Scale of the sky past the voxels (`setSkyLighting`). Default 1. */
    skyScale?: number;
    /** Scale of the light the GI adds. Default 1. */
    intensity?: number;
    /** Share of the material's own sky ambient the GI replaces (with `setSkyLighting`), as `VoxelGIOptions.ambient` does. Default 1. */
    ambient?: number;
    /** SVGF's temporal blend floors: the weight of a new frame in the colour and the luminance moments once the history is long. Default 0.2 each. */
    temporalAlpha?: number;
    momentsAlpha?: number;
    /**
     * SVGF's edge stops: luminance (in the variance's standard deviations; default 4), normal (an
     * exponent of their cosine; 128), depth (in pixels' footprints off the centre's tangent plane; 1).
     */
    phiColor?: number;
    phiNormal?: number;
    phiDepth?: number;
    /** Frames the temporal history holds at most. Default 32. */
    maxHistory?: number;
    /** Alpha-test the grid's alpha-tested triangles (`RtSurface.alphaLayer`); off, they are solid. Default true. */
    alphaTest?: boolean;
    /**
     * WGSL defining `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, which may sample
     * `kansei_rt_alpha_texture` with `kansei_rt_alpha_sampler` (`setAlphaTexture`). Default: the
     * texture's alpha is at least a half (a white texture until one is set: every hit).
     */
    coveredWgsl?: string | null;
    /** `reference` mode's path vertices. Default 4. */
    referenceBounces?: number;
}

/** The trace's counters of a recent frame (`collectStats`). Rust: `rt::RtGiStats`. */
export interface RtGiStats {
    /** Surface texels traced, and those whose ray hit a triangle of the grid. */
    rays: number;
    hits: number;
    /** Cells visited plus triangles tested, by all the rays (shadow rays included). */
    cost: number;
    /** The most one texel's rays cost. */
    maxCost: number;
    shadowRays: number;
}

/** Bytes of the WGSL `RtGiParams` (rt_gi_common.wgsl; Rust `RtGiParamsGpu`). */
export const RT_GI_PARAMS_BYTES = 320;
/** Bytes of the WGSL `AtrousParams` (rt_gi_svgf.wgsl; Rust `AtrousParamsGpu`). */
export const RT_GI_ATROUS_PARAMS_BYTES = 16;

// rt_gi_common.wgsl's RT_GI_*
const FLAG_ALPHA = 1;
const FLAG_STATS = 2;
const FLAG_HISTORY = 4;
const FLAG_GRID = 8;
const FLAG_HIT_DIRECT = 16;
const FLAG_SHADOW_RAY = 32;
const FLAG_REFERENCE = 64;
const FLAG_ACCUMULATE = 128;
const FLAG_HAS_SKY = 256;
const MAX_ATROUS = 8;

/**
 * The targets at the trace resolution: 8 bytes a texel for each texture (16 for the moments), 16
 * for each wavelet buffer.
 */
interface Targets {
    width: number;
    height: number;
    traceWidth: number;
    traceHeight: number;
    /** this frame's raw signal */
    trace: GPUTexture;
    /** the temporal pass's output (rgb, the variance's square root) */
    integrated: GPUTexture;
    /** the wavelet's ping-pong (guide, colour and variance packed): the variance pass writes the second, the iterations alternate from there */
    ping: [GPUBuffer, GPUBuffer];
    /** the wavelet's result for the composite */
    denoised: GPUTexture;
    colorHistory: [GPUTexture, GPUTexture];
    moments: [GPUTexture, GPUTexture];
    guide: [GPUTexture, GPUTexture];
    /** the running sums (`accumulate`), made when first asked for */
    accum: GPUBuffer | null;
}

interface Gpu {
    params: GPUBuffer;
    /** one per iteration, last or not */
    atrousParams: GPUBuffer[];
    traceBGL: GPUBindGroupLayout;
    gridBGL: GPUBindGroupLayout;
    temporalBGL: GPUBindGroupLayout;
    varianceBGL: GPUBindGroupLayout;
    atrousBGL: GPUBindGroupLayout;
    compositeBGL: GPUBindGroupLayout;
    trace: GPUComputePipeline;
    temporal: GPUComputePipeline;
    variance: GPUComputePipeline;
    atrous: GPUComputePipeline;
    composite: GPUComputePipeline;
    noSky: GPUBuffer;
    /** bound for the running sums while not accumulating */
    noAccum: GPUBuffer;
    white: GPUTexture;
    alphaSampler: GPUSampler;
    stats: GPUBuffer;
    staging: GPUBuffer;
    /** the grid's group, with the grid generation and the alpha texture it was made with */
    gridGroup: { generation: number; alpha: GPUTextureView | null; group: GPUBindGroup } | null;
    targets: Targets | null;
}

/** The stats readback: nothing in flight, copied this frame (awaiting its submit), mapping. */
const enum StatsState { Free, Copied, Mapping }

const ATROUS_LABELS = ['RtGi/Atrous1', 'RtGi/Atrous2', 'RtGi/Atrous3', 'RtGi/Atrous4', 'RtGi/Atrous5', 'RtGi/Atrous6', 'RtGi/Atrous7', 'RtGi/Atrous8'];

/**
 * Diffuse GI traced through the renderer's ray tracing grid (`Renderer.enableRtGrid`;
 * `SceneRtGrid.handle`): one ray a pixel (by default one for each 2 x 2) a frame from the GBuffer's
 * surface in a cosine-distributed direction, the hits lit by their exact direct light and one
 * voxel cone's indirect light, the voxel GI's volume or clipmap (`withVolume`, `withClipmap`) past
 * the grid's box, then the sky (`setSkyLighting`). SVGF denoises the 1 spp signal at the trace
 * resolution; the composite upsamples it by depth and normal and adds albedo times it to the lit
 * colour, taking out the material's own sky ambient as `VoxelGIEffect` does.
 *
 * It is one GI path among voxel cones (`VoxelGIEffect`) and screen-space GI
 * (`ScreenSpaceGIEffect`), on the same scene setup: closest to a path-traced reference (contacts,
 * thin walls, sky through foliage), at a cost between theirs and a path tracer's. Put it where
 * `VoxelGIEffect` would go (before reflections, the atmosphere and TAA), and each frame pass it the
 * scene's lights (`updateLights`). The voxel GI it reads still updates (`SceneVoxelGi`,
 * `SceneVoxelClipmap`); the grid must hold the renderables the rays should hit (`Renderable.rt`).
 * Needs a single-sampled GBuffer depth and normal (an MSAA volume's resolved ones are).
 *
 * `mode` and `accumulate` turn it into a reference: a path tracer through the grid, and a running
 * mean of the raw signal while the view and settings stay put. `view` shows the indirect light,
 * the signal, SVGF's variance and history, or the rays' cost.
 *
 * Rust: `rt::RtDiffuseGiEffect` (`with_volume`, `with_clipmap`).
 */
export class RtDiffuseGiEffect extends PostProcessingEffect {
    public enabled = true;
    public view: RtGiView = 'lit';
    public mode: RtGiMode = 'hybrid';
    /**
     * Show the running mean of the raw signal instead of the denoised one, restarted when the
     * camera or the settings change (call `resetHistory` after changing the scene or lights).
     */
    public accumulate = false;
    public hitLighting: RtGiHitLighting;
    public shadows: RtGiShadows;
    public denoise: RtGiDenoise;
    public kernel: RtGiKernel;
    public atrousIterations: number;
    /** Trace the grid of triangles (on by default); off, every ray is a voxel cone from the surface (for comparison). */
    public traceGrid = true;
    public alphaTest: boolean;
    public maxDistance: number;
    public nearDistance: number;
    public coneTan: number;
    public coneSteps: number;
    public hitConeTan: number;
    public hitConeSteps: number;
    public skyScale: number;
    public intensity: number;
    public ambient: number;
    public temporalAlpha: number;
    public momentsAlpha: number;
    public phiColor: number;
    public phiNormal: number;
    public phiDepth: number;
    public maxHistory: number;
    public referenceBounces: number;
    /** Scale of the debug views' colours (to suit the tone mapping after the effect). */
    public heatScale = 1;
    /** Count the rays' work (`stats`), a few atomics a ray. */
    public collectStats = false;

    private _resolution: RtGiResolution;
    private readonly coveredWgsl: string | null;
    private readonly grid: RtGridHandle;
    private readonly source: VoxelVolume | VoxelClipmap;
    private readonly lights = new ComputeShadows();
    private skyLighting: GPUBuffer | null = null;
    private alphaTexture: GPUTextureView | null = null;
    private frame = 0;
    private prevViewProj: mat4 | null = null;
    private lastCameraFrame: number | null = null;
    /** frames in the running sums, and what they were taken with */
    private accumCount = 0;
    private accumKey: number[] | null = null;
    private statsState = StatsState.Free;
    private _stats: RtGiStats | null = null;
    private gpu: Gpu | null = null;
    private device: GPUDevice | null = null;

    /**
     * GI whose hits and far field read `volume` (`SceneVoxelGi.volume`, which has the anisotropic
     * mips it needs): a room. Or a clipmap, as `withClipmap`.
     */
    constructor(source: VoxelVolume | VoxelClipmap, grid: RtGridHandle, o: RtDiffuseGiOptions = {}) {
        super();
        if (source instanceof VoxelVolume && !source.anisotropicViews) {
            throw new Error('RtDiffuseGiEffect reads a volume with anisotropic mips (SceneVoxelGi\'s has them)');
        }
        this.source = source;
        this.grid = grid;
        this._resolution = o.resolution ?? 'half';
        this.hitLighting = o.hitLighting ?? 'direct';
        this.shadows = o.shadows ?? 'rays';
        this.denoise = o.denoise ?? 'svgf';
        this.kernel = o.kernel ?? '3x3';
        this.atrousIterations = Math.min(o.atrousIterations ?? 5, MAX_ATROUS);
        this.maxDistance = o.maxDistance ?? 1e4;
        this.nearDistance = o.nearDistance ?? 0;
        this.coneTan = o.coneTan ?? 0.1;
        this.coneSteps = o.coneSteps ?? 48;
        this.hitConeTan = o.hitConeTan ?? 0.577;
        this.hitConeSteps = o.hitConeSteps ?? 16;
        this.skyScale = o.skyScale ?? 1;
        this.intensity = o.intensity ?? 1;
        this.ambient = o.ambient ?? 1;
        this.temporalAlpha = o.temporalAlpha ?? 0.2;
        this.momentsAlpha = o.momentsAlpha ?? 0.2;
        this.phiColor = o.phiColor ?? 4;
        this.phiNormal = o.phiNormal ?? 128;
        this.phiDepth = o.phiDepth ?? 1;
        this.maxHistory = o.maxHistory ?? 32;
        this.alphaTest = o.alphaTest ?? true;
        this.coveredWgsl = o.coveredWgsl ?? null;
        this.referenceBounces = o.referenceBounces ?? 4;
    }

    /** GI whose hits and far field read a voxel volume with anisotropic mips (`SceneVoxelGi.volume`): a room. Rust: `RtDiffuseGiEffect::with_volume`. */
    public static withVolume(volume: VoxelVolume, grid: RtGridHandle, options: RtDiffuseGiOptions = {}): RtDiffuseGiEffect {
        return new RtDiffuseGiEffect(volume, grid, options);
    }

    /** GI whose hits and far field read a voxel clipmap (`SceneVoxelClipmap.clipmap`): large and outdoor scenes. Rust: `RtDiffuseGiEffect::with_clipmap`. */
    public static withClipmap(clipmap: VoxelClipmap, grid: RtGridHandle, options: RtDiffuseGiOptions = {}): RtDiffuseGiEffect {
        return new RtDiffuseGiEffect(clipmap, grid, options);
    }

    /**
     * The scene's directional, point and area lights (`scene.directionalLights`, `pointLights`,
     * `areaLights`), each frame they may change: what lights the hits.
     */
    public updateLights(dirLights: readonly DirectionalLight[], pointLights: readonly PointLight[], areaLights: readonly AreaLight[] = []): void {
        // (every light, whether volumetric or not; area lights' 2D map is no case of compute_shadows.wgsl)
        this.lights.updateLights(dirLights, pointLights, areaLights, false, false);
    }

    /** The renderer's spot lights (`Renderer.spotLightsBuffer`) and their shadow atlas. */
    public setSpotLights(lights: GPUBuffer | null, atlas: SpotShadowAtlas | null): void {
        this.lights.setSpotLights(lights, atlas?.arrayView ?? null);
    }

    /** The directional shadow map: the sun's shadows past the grid's box (or everywhere with `maps` shadows). */
    public setShadowMap(shadowMap: ShadowMap | null): void {
        this.lights.setShadowMap(shadowMap);
    }

    /** Cascaded shadows in place of `setShadowMap`'s (`Renderer.cascadedShadowMap`). */
    public setCascadedShadowMap(csm: CascadedShadowSource | null): void {
        this.lights.setCascadedShadowMap(csm);
    }

    /** Point lights' cube shadows (`maps` shadows; `Renderer.cubeMapShadowMap`). */
    public setPointShadows(cube: CubeMapShadowMap | null): void {
        this.lights.setPointShadows(cube);
    }

    /** The sky past the voxels, and the ambient the composite takes out (a `SkyLighting` uniform); black without one. */
    public setSkyLighting(sky: GPUBuffer | null): void {
        this.skyLighting = sky;
    }

    /** The texture `kansei_rt_covered` reads (`kansei_rt_alpha_texture`). */
    public setAlphaTexture(view: GPUTextureView | null): void {
        this.alphaTexture = view;
    }

    public get resolution(): RtGiResolution {
        return this._resolution;
    }

    /** Trace at another resolution (the targets are made anew, the history restarts). */
    public setResolution(resolution: RtGiResolution): void {
        if (resolution === this._resolution) return;
        this._resolution = resolution;
        if (this.gpu?.targets) {
            RtDiffuseGiEffect.destroyTargets(this.gpu.targets);
            this.gpu.targets = null;
        }
        this.resetHistory();
    }

    /** Start the denoiser's history and the running mean over (after a cut, or a change of the scene or its lights the history should not blend through). */
    public resetHistory(): void {
        this.prevViewProj = null;
        this.accumKey = null;
        this.accumCount = 0;
    }

    /** Frames in the running mean (`accumulate`). */
    public get accumulated(): number {
        return this.accumulate ? this.accumCount : 0;
    }

    /** The counters of a recent frame, while `collectStats` is on (they arrive a few frames late). */
    public get stats(): RtGiStats | null {
        return this._stats;
    }

    /** Bytes of the effect's targets (120 a trace texel, 16 more while accumulating). */
    public memoryBytes(): number {
        const t = this.gpu?.targets;
        if (!t) return 0;
        // trace, integrated, denoised, 2 colour, 2 guides at 8 bytes; 2 moments, 2 ping at 16
        return t.traceWidth * t.traceHeight * (7 * 8 + 4 * 16) + (t.accum?.size ?? 0);
    }

    public isActive(): boolean {
        return this.enabled;
    }

    private downscale(): number {
        return this._resolution === 'full' ? 1 : 2;
    }

    public initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this.device = device;
        this.gpu ??= this.initGpu(device);
        this.initialized = true;
    }

    private initGpu(device: GPUDevice): Gpu {
        const visibility = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } });
        const storage = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } });
        const texture = (binding: number, filterable: boolean, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, texture: { sampleType: filterable ? 'float' : 'unfilterable-float', viewDimension } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth' } });
        const uint = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'uint' } });
        const storageTexture = (binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, storageTexture: { access: 'write-only', format, viewDimension: '2d' } });
        const sampler = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, sampler: { type: 'filtering' } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });

        const clipmap = this.source instanceof VoxelClipmap;
        const traceBGL = bgl('RtGi/Trace', [
            ...ComputeShadows.layoutEntries(),
            uniform(20), depth(21), texture(22, false), storageTexture(23, 'rgba16float'), uniform(24), storage(25),
            ...(clipmap ? clipmapLayoutEntries(visibility) : [
                uniform(60), texture(61, true, '3d'), sampler(62),
                ...[40, 41, 42, 43, 44, 45].map((b) => texture(b, true, '3d')),
            ]),
        ]);
        const gridBGL = bgl('RtGi/Grid', [...RtGrid.layoutEntries(0, visibility), texture(3, true), sampler(4), storage(5)]);
        const temporalBGL = bgl('RtGi/Temporal', [
            uniform(20), texture(21, false), depth(22), texture(23, false), texture(24, false), texture(25, false), texture(26, false), uint(27),
            storageTexture(28, 'rgba16float'), storageTexture(29, 'rgba32float'), storageTexture(30, 'rg32uint'),
        ]);
        const varianceBGL = bgl('RtGi/Variance', [uniform(20), texture(31, false), texture(32, false), uint(33), storage(34)]);
        const atrousBGL = bgl('RtGi/Atrous', [uniform(20), uniform(35), storage(36, true), storage(37), storageTexture(38, 'rgba16float'), storageTexture(39, 'rgba16float')]);
        const compositeBGL = bgl('RtGi/Composite', [
            uniform(20), texture(21, false), depth(22), texture(23, false), texture(24, false), texture(25, false), uint(26), storage(27, true),
            storageTexture(28, 'rgba16float'), uniform(29), texture(30, false), texture(31, false),
        ]);
        const module = (label: string, code: string) => device.createShaderModule({ label, code });
        const pipeline = (label: string, shader: GPUShaderModule, entryPoint: string, layouts: GPUBindGroupLayout[]) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: layouts }),
            compute: { module: shader, entryPoint },
        });
        const svgf = module('RtGi/Svgf', RT_GI_SVGF_WGSL);
        const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ label, size, usage });
        const white = device.createTexture({ label: 'RtGi/White', size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1]);
        // one per iteration i, its first feeding the history, last or not
        const atrousParams: GPUBuffer[] = [];
        for (let k = 0; k < MAX_ATROUS * 2; k++) {
            const i = k >> 1;
            const b = buffer('RtGi/AtrousParams', RT_GI_ATROUS_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            device.queue.writeBuffer(b, 0, new Uint32Array([1 << i, i === 0 ? 1 : 0, k & 1, 0]));
            atrousParams.push(b);
        }
        return {
            params: buffer('RtGi/Params', RT_GI_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
            atrousParams,
            traceBGL,
            gridBGL,
            temporalBGL,
            varianceBGL,
            atrousBGL,
            compositeBGL,
            trace: pipeline('RtGi/Trace', module('RtGi/Trace', rtGiTraceWgsl(this.coveredWgsl ?? RT_DEFAULT_COVERED_WGSL, clipmap)), 'main', [traceBGL, gridBGL]),
            temporal: pipeline('RtGi/Temporal', svgf, 'temporal', [temporalBGL]),
            variance: pipeline('RtGi/Variance', svgf, 'variance', [varianceBGL]),
            atrous: pipeline('RtGi/Atrous', svgf, 'atrous', [atrousBGL]),
            composite: pipeline('RtGi/Composite', module('RtGi/Composite', RT_GI_COMPOSITE_WGSL), 'main', [compositeBGL]),
            // (a zeroed SkyLighting: black)
            noSky: buffer('RtGi/NoSky', 256, GPUBufferUsage.UNIFORM),
            noAccum: buffer('RtGi/NoAccumulation', 16, GPUBufferUsage.STORAGE),
            white,
            alphaSampler: device.createSampler({ label: 'RtGi/Alpha', magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' }),
            stats: buffer('RtGi/Stats', 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
            staging: buffer('RtGi/StatsReadback', 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
            gridGroup: null,
            targets: null,
        };
    }

    private static makeTargets(device: GPUDevice, width: number, height: number, tw: number, th: number): Targets {
        const texture = (label: string, format: GPUTextureFormat, extra: GPUTextureUsageFlags = 0) => device.createTexture({
            label,
            size: [tw, th],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | extra,
        });
        const packed = (label: string) => device.createBuffer({ label, size: tw * th * 16, usage: GPUBufferUsage.STORAGE });
        return {
            width,
            height,
            traceWidth: tw,
            traceHeight: th,
            trace: texture('RtGi/Trace', 'rgba16float'),
            integrated: texture('RtGi/Integrated', 'rgba16float', GPUTextureUsage.COPY_SRC),
            ping: [packed('RtGi/Ping'), packed('RtGi/Pong')],
            denoised: texture('RtGi/Denoised', 'rgba16float'),
            colorHistory: [texture('RtGi/ColorHistory', 'rgba16float', GPUTextureUsage.COPY_DST), texture('RtGi/ColorHistory', 'rgba16float', GPUTextureUsage.COPY_DST)],
            // (f32: the luminance's second moment overflows f16)
            moments: [texture('RtGi/Moments', 'rgba32float'), texture('RtGi/Moments', 'rgba32float')],
            guide: [texture('RtGi/Guide', 'rg32uint'), texture('RtGi/Guide', 'rg32uint')],
            accum: null,
        };
    }

    private static destroyTargets(t: Targets): void {
        for (const x of [t.trace, t.integrated, t.denoised, ...t.colorHistory, ...t.moments, ...t.guide]) x.destroy();
        for (const b of [...t.ping, t.accum]) b?.destroy();
    }

    /** The settings the running sums depend on. */
    private accumSettings(): number[] {
        const bits = new Uint32Array(new Float32Array([this.coneTan, this.hitConeTan, this.skyScale, this.maxDistance, this.nearDistance]).buffer);
        return [
            RT_GI_MODES.indexOf(this.mode),
            RT_GI_HIT_LIGHTINGS.indexOf(this.hitLighting),
            RT_GI_SHADOWS.indexOf(this.shadows),
            (this.traceGrid ? 1 : 0) | (this.alphaTest ? 2 : 0),
            this.referenceBounces,
            this.coneSteps,
            bits[0],
            bits[1],
            this.hitConeSteps,
            bits[2],
            bits[3],
            bits[4],
        ];
    }

    /** The counters' readback: map last frame's copy (its frame has been submitted since), and take a finished map. */
    private pollStats(): void {
        const gpu = this.gpu!;
        if (this.statsState === StatsState.Copied) {
            this.statsState = StatsState.Mapping;
            gpu.staging.mapAsync(GPUMapMode.READ).then(
                () => {
                    const w = new Uint32Array(gpu.staging.getMappedRange().slice(0));
                    gpu.staging.unmap();
                    if (this.collectStats) this._stats = { rays: w[0], hits: w[1], cost: w[3], maxCost: w[4], shadowRays: w[5] };
                    this.statsState = StatsState.Free;
                },
                () => { this.statsState = StatsState.Free; },
            );
        }
        if (!this.collectStats) this._stats = null;
    }

    public render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
        _emissive?: GPUTexture,
        gbuffer?: GBuffer,
    ): void {
        const device = this.device!;
        const gpu = this.gpu!;
        if (!gbuffer) throw new Error('RtDiffuseGiEffect reads the GBuffer\'s normals, albedo and velocity (render it through a PostProcessingVolume)');
        this.pollStats();
        const downscale = this.downscale();
        const tw = Math.ceil(width / downscale);
        const th = Math.ceil(height / downscale);
        if (!gpu.targets || gpu.targets.width !== width || gpu.targets.height !== height || gpu.targets.traceWidth !== tw || gpu.targets.traceHeight !== th) {
            if (gpu.targets) RtDiffuseGiEffect.destroyTargets(gpu.targets);
            gpu.targets = RtDiffuseGiEffect.makeTargets(device, width, height, tw, th);
            this.prevViewProj = null;
            this.accumKey = null;
        }
        const t = gpu.targets;
        // a frame skipped (the effect was off, a cut): no history
        const cameraFrame = camera.frame;
        if (this.lastCameraFrame !== null && cameraFrame !== this.lastCameraFrame && cameraFrame !== ((this.lastCameraFrame + 1) >>> 0)) {
            this.prevViewProj = null;
            this.accumKey = null;
        }
        this.lastCameraFrame = cameraFrame;
        const proj = camera.projectionMatrix.internalMat4;
        const view = camera.viewMatrix.internalMat4;
        const viewProj = mat4.multiply(mat4.create(), proj, view);
        // the running sums restart when the view or the settings change
        const key = [...viewProj, ...this.accumSettings()];
        if (this.accumulate) {
            if (!this.accumKey || this.accumKey.length !== key.length || this.accumKey.some((v, i) => v !== key[i])) {
                this.accumCount = 0;
                this.accumKey = key;
            }
            t.accum ??= device.createBuffer({ label: 'RtGi/Accumulation', size: tw * th * 16, usage: GPUBufferUsage.STORAGE });
        } else {
            this.accumKey = null;
            this.accumCount = 0;
        }
        this.lights.prepare(device);
        let flags = 0;
        if (this.alphaTest) flags |= FLAG_ALPHA;
        if (this.collectStats) flags |= FLAG_STATS;
        if (this.prevViewProj) flags |= FLAG_HISTORY;
        if (this.traceGrid) flags |= FLAG_GRID;
        if (this.hitLighting === 'direct') flags |= FLAG_HIT_DIRECT;
        if (this.shadows === 'rays') flags |= FLAG_SHADOW_RAY;
        if (this.mode === 'reference') flags |= FLAG_REFERENCE;
        if (this.accumulate) flags |= FLAG_ACCUMULATE;
        if (this.skyLighting) flags |= FLAG_HAS_SKY;
        const invProj = mat4.invert(mat4.create(), proj);
        const data = new ArrayBuffer(RT_GI_PARAMS_BYTES);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32.set(invProj, 0);
        f32.set(mat4.invert(mat4.create(), view), 16);
        f32.set(this.prevViewProj ?? viewProj, 32);
        f32.set([width, height, tw, th], 48);
        u32[52] = this.frame;
        u32[53] = downscale;
        u32[54] = flags;
        u32[55] = RT_GI_VIEWS.indexOf(this.view);
        f32[56] = Math.max(this.maxDistance, 0);
        f32[57] = Math.max(this.coneTan, 1e-3);
        u32[58] = this.coneSteps;
        f32[59] = Math.max(this.skyScale, 0);
        f32[60] = Math.max(this.intensity, 0);
        f32[61] = this.ambient;
        f32[62] = Math.max(this.hitConeTan, 1e-3);
        u32[63] = this.hitConeSteps;
        u32[64] = Math.max(this.referenceBounces, 1);
        u32[65] = this.accumCount;
        f32[66] = Math.min(Math.max(this.temporalAlpha, 0.01), 1);
        f32[67] = Math.min(Math.max(this.momentsAlpha, 0.01), 1);
        f32[68] = Math.max(this.phiColor, 1e-3);
        f32[69] = this.phiNormal;
        f32[70] = Math.max(this.phiDepth, 1e-3);
        u32[71] = this.lights.dirCount;
        u32[72] = this.lights.pointCount;
        u32[73] = this.lights.hasShadowMap ? 1 : 0;
        f32[74] = 2 * invProj[5] / height;
        f32[75] = this.heatScale;
        f32[76] = Math.max(this.maxHistory, 1);
        u32[77] = this.kernel === '5x5' ? 2 : 1;
        f32[78] = Math.max(this.nearDistance, 0);
        device.queue.writeBuffer(gpu.params, 0, data);
        const cur = this.frame % 2;
        const prev = 1 - cur;

        const group = (label: string, layout: GPUBindGroupLayout, resources: [number, GPUBindingResource][], extra: GPUBindGroupEntry[] = []) =>
            device.createBindGroup({ label, layout, entries: [...extra, ...resources.map(([binding, resource]) => ({ binding, resource }))] });
        const params = { buffer: gpu.params };
        const sky = { buffer: this.skyLighting ?? gpu.noSky };
        const accum = { buffer: (this.accumulate ? t.accum : null) ?? gpu.noAccum };
        const depthView = depth.createView();
        const normalView = gbuffer.normalTexture.createView();
        const traceView = t.trace.createView();
        const integratedView = t.integrated.createView();
        const denoisedView = t.denoised.createView();
        const colorHistory = t.colorHistory.map((x) => x.createView());
        const moments = t.moments.map((x) => x.createView());
        const guide = t.guide.map((x) => x.createView());

        // the trace's group: the lights, the GBuffer, its targets, the voxel source
        const source = this.source;
        const sourceEntries: GPUBindGroupEntry[] = source instanceof VoxelClipmap ? clipmapEntries(source) : [
            { binding: 60, resource: { buffer: source.uniform } },
            { binding: 61, resource: source.view },
            { binding: 62, resource: source.sampler },
            ...source.anisotropicViews!.map((v, i) => ({ binding: 40 + i, resource: v })),
        ];
        const trace = group('RtGi/Trace', gpu.traceBGL, [
            [20, params], [21, depthView], [22, normalView], [23, traceView], [24, sky], [25, accum],
        ], [...this.lights.entries(), ...sourceEntries]);
        // the grid's group, made anew when the grid's buffers or the alpha texture change
        const grid = this.grid;
        if (!gpu.gridGroup || gpu.gridGroup.generation !== grid.generation || gpu.gridGroup.alpha !== this.alphaTexture) {
            gpu.gridGroup = {
                generation: grid.generation,
                alpha: this.alphaTexture,
                group: group('RtGi/Grid', gpu.gridBGL, [
                    [0, { buffer: grid.buffers[0] }], [1, { buffer: grid.buffers[1] }], [2, { buffer: grid.buffers[2] }],
                    [3, this.alphaTexture ?? gpu.white.createView()], [4, gpu.alphaSampler], [5, { buffer: gpu.stats }],
                ]),
            };
        }
        const temporal = group('RtGi/Temporal', gpu.temporalBGL, [
            [20, params], [21, traceView], [22, depthView], [23, normalView], [24, gbuffer.velocityTexture.createView()],
            [25, colorHistory[prev]], [26, moments[prev]], [27, guide[prev]], [28, integratedView], [29, moments[cur]], [30, guide[cur]],
        ]);
        const svgf = this.denoise === 'svgf';
        const iterations = svgf ? Math.min(Math.max(Math.floor(this.atrousIterations), 0), MAX_ATROUS) : 0;
        // the signal the composite shows
        const signal = this.view === 'cost' || this.denoise === 'off' ? traceView
            : this.denoise === 'temporal' || iterations === 0 ? integratedView
            : denoisedView;
        const composite = group('RtGi/Composite', gpu.compositeBGL, [
            [20, params], [21, input.createView()], [22, depthView], [23, normalView], [24, gbuffer.albedoTexture.createView()],
            [25, signal], [26, guide[cur]], [27, accum], [28, output.createView()], [29, sky], [30, moments[cur]], [31, integratedView],
        ]);

        const readStats = this.collectStats && this.statsState === StatsState.Free;
        if (readStats) commandEncoder.clearBuffer(gpu.stats);
        const gx = Math.ceil(tw / 8);
        const gy = Math.ceil(th / 8);
        const dispatch = (label: string, pipeline: GPUComputePipeline, groups: GPUBindGroup[], x: number, y: number) => {
            const pass = commandEncoder.beginComputePass({ label, timestampWrites: gpuPass(label) });
            pass.setPipeline(pipeline);
            groups.forEach((g, i) => pass.setBindGroup(i, g));
            pass.dispatchWorkgroups(x, y);
            pass.end();
        };
        dispatch('RtGi/Trace', gpu.trace, [trace, gpu.gridGroup.group], gx, gy);
        if (readStats) {
            commandEncoder.copyBufferToBuffer(gpu.stats, 0, gpu.staging, 0, 32);
            this.statsState = StatsState.Copied;
        }
        dispatch('RtGi/Temporal', gpu.temporal, [temporal], gx, gy);
        if (svgf) {
            // the variance pass writes the second ping buffer; iteration i reads the other
            const variance = group('RtGi/Variance', gpu.varianceBGL, [
                [20, params], [31, integratedView], [32, moments[cur]], [33, guide[cur]], [34, { buffer: t.ping[1] }],
            ]);
            dispatch('RtGi/Variance', gpu.variance, [variance], gx, gy);
            for (let i = 0; i < iterations; i++) {
                const atrous = group('RtGi/Atrous', gpu.atrousBGL, [
                    [20, params], [35, { buffer: gpu.atrousParams[i * 2 + (i + 1 === iterations ? 1 : 0)] }],
                    [36, { buffer: t.ping[(i + 1) % 2] }], [37, { buffer: t.ping[i % 2] }], [38, colorHistory[cur]], [39, denoisedView],
                ]);
                dispatch(ATROUS_LABELS[i], gpu.atrous, [atrous], gx, gy);
            }
        }
        if (iterations === 0) {
            // the history is the temporal pass's output
            commandEncoder.copyTextureToTexture({ texture: t.integrated }, { texture: t.colorHistory[cur] }, [tw, th]);
        }
        dispatch('RtGi/Composite', gpu.composite, [composite], Math.ceil(width / 8), Math.ceil(height / 8));
        this.frame = (this.frame + 1) >>> 0;
        this.prevViewProj = viewProj;
        if (this.accumulate) this.accumCount++;
    }

    public resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    public destroy(): void {
        const gpu = this.gpu;
        if (!gpu) return;
        for (const b of [gpu.params, ...gpu.atrousParams, gpu.noSky, gpu.noAccum, gpu.stats, gpu.staging]) b.destroy();
        gpu.white.destroy();
        if (gpu.targets) RtDiffuseGiEffect.destroyTargets(gpu.targets);
        this.lights.destroy();
        this.gpu = null;
        this.initialized = false;
    }
}
