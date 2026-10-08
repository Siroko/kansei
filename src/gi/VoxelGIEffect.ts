import { mat4 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { GBuffer } from '../postprocessing/GBuffer';
import { PostProcessingEffect } from '../postprocessing/PostProcessingEffect';
import { ScreenSpaceGIEffect, ScreenSpaceGIOptions } from '../postprocessing/effects/ScreenSpaceGIEffect';
import { gpuPass } from '../profiling/Profiler';
import { CLIPMAP_PROBE_TRACE_WGSL, CLIPMAP_TRACE_WGSL, SCREEN_COMPOSITE_WGSL, SCREEN_TEMPORAL_WGSL, SCREEN_TRACE_WGSL } from './GiWGSL';
import type { ClipmapProbes } from './ClipmapProbes';
import { VoxelClipmap, clipmapEntries, clipmapLayoutEntries } from './VoxelClipmap';
import { gradientSkyLighting } from './ParticleConeShading';
import type { JumpFloodSdf } from './JumpFloodSdf';
import { PROBE_GRID_BYTES, SdfProbes } from './SdfProbes';
import { noSdfView } from './VoxelInjection';
import { VOXEL_VOLUME_BYTES, Vec3, VoxelGiQuality, VoxelVolume } from './VoxelVolume';

/** What `VoxelGIEffect` sets up. Rust: `gi::VoxelGIOptions`. */
export interface VoxelGIOptions {
    /** Resolution of the trace (`low` a quarter each way, else half) and the cones' steps. Default `medium`. */
    quality?: VoxelGiQuality;
    /** Scale of the light added (1 is physical). Default 1. */
    intensity?: number;
    /** Voxels out along the normal the cones start from, past the surface's own voxels. Default 1.5. */
    startVoxels?: number;
    /** Metres a cone looks. Default 1e4. */
    maxDistanceM?: number;
    /** Weight of each new frame in the accumulated result. Default 0.1. */
    temporalBlend?: number;
    /**
     * How much of the materials' own sky ambient the GI replaces (needs `setSkyLighting`, and
     * materials lit by `SKY_LIGHTING_WGSL`'s `skyIrradiance`). Default 1.
     */
    materialAmbient?: number;
    /** Scale of the sky past the volume. Default 1. */
    skyScale?: number;
    /**
     * Screen-space GI in front of the voxels (`gi=voxel+ssgi`): it brings the light of what it
     * sees occlude each direction within its radius, at full screen detail, and the voxels light
     * the rest of the hemisphere. Unset: the voxels alone.
     */
    nearField?: Partial<ScreenSpaceGIOptions>;
    /**
     * Strength of the distance field's ambient occlusion on the GI (`setSdf`; 0, the default:
     * none): the contact occlusion the coarse cones miss.
     */
    sdfAo?: number;
    /**
     * With a clipmap (`withClipmap`): levels finer than they are wide the cones read. Each then
     * samples the voxels along its axis sparsely, which sees the sky through gaps narrower than the
     * cone (a road between trees, a street between walls) where voxels as wide as the cone fill
     * them; the temporal filter averages the samples. 0 reads the levels as wide as the cones.
     * Default 2.
     */
    clipmapLevelBias?: number;
}

/** What the effect reads: a volume (with its anisotropic chains) or a clipmap. */
type Source =
    | { kind: 'volume', volume: VoxelVolume, anisotropic: GPUTextureView[] }
    | { kind: 'clipmap', clipmap: VoxelClipmap };

/** The trace and the voxels' view of a clipmap source, and the stand-in the composite binds for the volume. */
interface ClipmapGpu {
    trace: GPUComputePipeline;
    showVoxels: GPUComputePipeline;
    bgl: GPUBindGroupLayout;
    /** The trace from the clipmap's probes in place of cones (clipmap_probe_trace.wgsl). */
    probeTrace: GPUComputePipeline;
    probeTraceBGL: GPUBindGroupLayout;
    noVolume: GPUBuffer;
}

/** Bytes of the WGSL `VoxelGiParams` (screen_common.wgsl; Rust `VoxelGiParamsGpu`). */
export const VOXEL_GI_PARAMS_BYTES = 352;

interface Targets {
    width: number;
    height: number;
    trace: GPUTexture;
    history: [GPUTexture, GPUTexture];
}

interface Gpu {
    params: GPUBuffer;
    trace: GPUComputePipeline;
    traceBGL: GPUBindGroupLayout;
    temporal: GPUComputePipeline;
    temporalBGL: GPUBindGroupLayout;
    composite: GPUComputePipeline;
    /** The composite with the probes as the far field (`main_probes`). */
    compositeProbes: GPUComputePipeline;
    compositeBGL: GPUBindGroupLayout;
    sampler: GPUSampler;
    /** The sky past the volume until `setSkyLighting`: `skyGradient`. */
    gradientSky: GPUBuffer;
    /** Bound as the near field without one. */
    noNear: GPUTexture;
    /** Bound as the distance field without one. */
    noSdf: GPUTextureView;
    /** Bound as the probes without them: the grid, the SH, the state, the depth. */
    noProbes: GPUBuffer[];
    /** The passes of a clipmap source, when the effect reads one. */
    clipmap: ClipmapGpu | null;
    targets: Targets | null;
}

/**
 * Diffuse global illumination on screen from a voxel volume of the scene's light (the renderer's
 * `SceneVoxelGi`, `Renderer.enableVoxelGI`): per pixel, at reduced resolution, six cones over the
 * hemisphere around the surface's normal gather the light the volume holds (off screen and
 * behind the camera too, bounces included) and the sky past it; a temporal filter smooths the
 * cones' per-frame rotation; the composite adds albedo / pi times that irradiance to the scene.
 *
 * Add it first in the chain. `enabled = false` skips it at no cost. It adds light only where a
 * material writes the GBuffer's albedo (and normal), and replaces the materials' sky ambient once
 * the sky's lighting is set. With `setProbes` the far field comes from irradiance probes instead of
 * per-pixel cones; with `setSdf` the distance field adds contact AO (`sdfAo`); with the
 * `nearField` option screen-space GI goes first and the voxels light what it cannot see.
 *
 * `withClipmap` reads a voxel clipmap (`SceneVoxelClipmap.clipmap`) instead: the cones read its
 * levels, each the level as wide as the cone there that holds the point, and `setClipmapProbes`
 * takes the far field from its probes. Rust: `gi::VoxelGIEffect`.
 */
export class VoxelGIEffect extends PostProcessingEffect {
    public enabled = true;
    public quality: VoxelGiQuality;
    public intensity: number;
    public startVoxels: number;
    public maxDistanceM: number;
    public temporalBlend: number;
    public materialAmbient: number;
    public skyScale: number;
    /**
     * Debug view: output only the light the GI adds (albedo / pi times its irradiance), black
     * elsewhere, in place of the lit image.
     */
    public showIndirect = false;
    /**
     * Debug view: the volume's voxels and their light as the camera sees them (mip 0 marched per
     * pixel), in place of the lit image, to inspect the voxelization. Over `showIndirect`.
     */
    public showVoxels = false;
    /**
     * Debug view: the distance field (`setSdf`) on the horizontal plane at this height, metres,
     * over the dimmed scene. Over the other views. Null: off.
     */
    public showSdfSlice: number | null = null;
    /**
     * Debug view: the probes (`setProbes`) as balls lit by their own irradiance, dark red where one
     * is left out (inside geometry), over the scene without its GI. Over the other views.
     */
    public showProbes = false;
    public sdfAo: number;
    /** See `VoxelGIOptions.clipmapLevelBias`. */
    public clipmapLevelBias: number;
    /** The sky past the volume without `setSkyLighting`: scene radiance straight up and down. */
    public skyGradient: [Vec3, Vec3] = [[0, 0, 0], [0, 0, 0]];

    private readonly source: Source;
    private skyLighting: GPUBuffer | null = null;
    private readonly near: ScreenSpaceGIEffect | null;
    private sdf: GPUTextureView | null = null;
    /** The probes' grid, SH, state and depth buffers (`setProbes`). */
    private probes: GPUBuffer[] | null = null;
    /** A clipmap's probes' grid and buffer (`setClipmapProbes`). */
    private clipmapProbes: [GPUBuffer, GPUBuffer] | null = null;
    private prevViewProj: mat4 | null = null;
    private frame = 0;
    private gpu: Gpu | null = null;
    private device: GPUDevice | null = null;

    /**
     * Read `volume` (`SceneVoxelGi.volume`, or any volume with anisotropic mips).
     */
    constructor(volume: VoxelVolume, options?: VoxelGIOptions);
    constructor(source: Source, options?: VoxelGIOptions);
    constructor(volume: VoxelVolume | Source, options: VoxelGIOptions = {}) {
        super();
        if (volume instanceof VoxelVolume) {
            const anisotropic = volume.anisotropicViews;
            if (!anisotropic) {
                throw new Error('VoxelGIEffect reads a volume with anisotropic mips (VoxelVolume.setAnisotropicMips; SceneVoxelGi\'s has them)');
            }
            this.source = { kind: 'volume', volume, anisotropic };
        } else {
            this.source = volume;
        }
        this.quality = options.quality ?? 'medium';
        this.intensity = options.intensity ?? 1;
        this.startVoxels = options.startVoxels ?? 1.5;
        this.maxDistanceM = options.maxDistanceM ?? 1e4;
        this.temporalBlend = options.temporalBlend ?? 0.1;
        this.materialAmbient = options.materialAmbient ?? 1;
        this.skyScale = options.skyScale ?? 1;
        this.sdfAo = options.sdfAo ?? 0;
        this.clipmapLevelBias = options.clipmapLevelBias ?? 2;
        this.near = options.nearField ? new ScreenSpaceGIEffect(options.nearField) : null;
    }

    /** The screen-space GI in front of the voxels, if any (to tune it). */
    public get nearField(): ScreenSpaceGIEffect | null {
        return this.near;
    }

    /**
     * Read `clipmap` (`SceneVoxelClipmap.clipmap`) instead of a volume: the cones read its levels
     * (the distance field and the volume's probes don't apply). Rust: `VoxelGIEffect::with_clipmap`.
     */
    public static withClipmap(clipmap: VoxelClipmap, options: VoxelGIOptions = {}): VoxelGIEffect {
        return new VoxelGIEffect({ kind: 'clipmap', clipmap }, options);
    }

    /** Whether it reads a clipmap (`withClipmap`). */
    public get readsClipmap(): boolean {
        return this.source.kind === 'clipmap';
    }

    /**
     * Take the far field from a clipmap's irradiance probes (`SceneVoxelClipmap.probes`): each
     * pixel's irradiance from the probes around it, in place of the cones traced through the
     * clipmap (whose temporal filter is then skipped). Null goes back to the cones. Ignored without
     * a clipmap (`withClipmap`).
     */
    public setClipmapProbes(probes: ClipmapProbes | null): void {
        this.clipmapProbes = probes && this.readsClipmap ? [probes.gridBuffer, probes.probeBuffer] : null;
    }

    /** Whether the far field comes from a clipmap's probes (`setClipmapProbes`). */
    public get usesClipmapProbes(): boolean {
        return this.clipmapProbes !== null;
    }

    /**
     * Read the scene's distance field (`SceneVoxelGi.sdf`, over the same volume) for `sdfAo` and
     * `showSdfSlice`; null leaves them off.
     */
    public setSdf(sdf: JumpFloodSdf | null): void {
        this.sdf = this.readsClipmap ? null : sdf?.view ?? null;
    }

    /**
     * Take the far field from `probes` (`SceneVoxelGi.probes`, over the same volume): each pixel's
     * irradiance from the probes around it, in place of the cones traced per pixel (whose passes
     * are then skipped), still under the near field if there is one. Null goes back to the cones.
     */
    public setProbes(probes: SdfProbes | null): void {
        this.probes = probes && !this.readsClipmap ? [probes.gridBuffer, probes.shBuffer, probes.stateBuffer, probes.depthBuffer] : null;
    }

    /** Whether the far field comes from probes (`setProbes`). */
    public get usesProbes(): boolean {
        return this.probes !== null;
    }

    /**
     * The sky's lighting (a `SkyLighting` uniform): the light past the volume, and the materials'
     * ambient the GI replaces. Also the near field's. Null goes back to `skyGradient`.
     */
    public setSkyLighting(skyLighting: GPUBuffer | null): void {
        this.skyLighting = skyLighting;
        this.near?.setSkyLighting(skyLighting);
    }

    /** Drop the accumulated frames; call on camera cuts. */
    public resetHistory(): void {
        this.prevViewProj = null;
        this.near?.resetHistory();
    }

    public isActive(): boolean {
        return this.enabled;
    }

    private traceScale(): number {
        return this.quality === 'low' ? 0.25 : 0.5;
    }

    public initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this.device = device;
        if (!this.gpu) this.gpu = VoxelGIEffect.initGpu(device, this.readsClipmap);
        this.initialized = true;
    }

    private static initGpu(device: GPUDevice, clipmap: boolean): Gpu {
        const visibility = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } });
        const texture = (binding: number, filterable: boolean, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, texture: { sampleType: filterable ? 'float' : 'unfilterable-float', viewDimension } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth' } });
        const storage = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } });
        const sampler = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, sampler: { type: 'filtering' } });
        const storageBuffer = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });

        const traceBGL = bgl('VoxelGI/TraceBGL', [
            uniform(0), depth(1), texture(2, false), uniform(3), texture(4, true, '3d'), sampler(5), uniform(6), storage(7),
            // the anisotropic mips and the distance field (voxel_irradiance.wgsl)
            ...[40, 41, 42, 43, 44, 45, 46].map((b) => texture(b, true, '3d')),
        ]);
        const temporalBGL = bgl('VoxelGI/TemporalBGL', [uniform(0), texture(1, false), texture(2, true), depth(3), storage(4), sampler(5)]);
        const compositeBGL = bgl('VoxelGI/CompositeBGL', [
            uniform(0), texture(1, false), depth(2), texture(3, false), texture(4, false), texture(5, false), texture(6, false),
            uniform(7), storage(8), uniform(9), texture(10, true, '3d'), sampler(11), texture(12, true, '3d'),
            uniform(13), storageBuffer(14), storageBuffer(15), storageBuffer(16),
        ]);
        const pipeline = (label: string, code: string, layout: GPUBindGroupLayout, entryPoint = 'main') => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint },
        });
        const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ label, size, usage });
        return {
            params: buffer('VoxelGI/ScreenParams', VOXEL_GI_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
            trace: pipeline('VoxelGI/Trace', SCREEN_TRACE_WGSL, traceBGL),
            traceBGL,
            temporal: pipeline('VoxelGI/Temporal', SCREEN_TEMPORAL_WGSL, temporalBGL),
            temporalBGL,
            composite: pipeline('VoxelGI/Composite', SCREEN_COMPOSITE_WGSL, compositeBGL),
            compositeProbes: pipeline('VoxelGI/CompositeProbes', SCREEN_COMPOSITE_WGSL, compositeBGL, 'main_probes'),
            compositeBGL,
            sampler: device.createSampler({ label: 'VoxelGI/Linear', magFilter: 'linear', minFilter: 'linear' }),
            gradientSky: buffer('VoxelGI/GradientSky', gradientSkyLighting([0, 0, 0], [0, 0, 0]).byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
            noNear: device.createTexture({ label: 'VoxelGI/NoNearField', size: [1, 1], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING }),
            noSdf: noSdfView(device),
            noProbes: [
                buffer('VoxelGI/NoProbeGrid', PROBE_GRID_BYTES, GPUBufferUsage.UNIFORM),
                buffer('VoxelGI/NoProbeSh', 16, GPUBufferUsage.STORAGE),
                buffer('VoxelGI/NoProbeState', 16, GPUBufferUsage.STORAGE),
                buffer('VoxelGI/NoProbeDepth', 16, GPUBufferUsage.STORAGE),
            ],
            clipmap: clipmap ? (() => {
                const bgl = device.createBindGroupLayout({
                    label: 'VoxelGI/ClipmapTraceBGL',
                    entries: [uniform(0), depth(1), texture(2, false), uniform(6), storage(7), ...clipmapLayoutEntries(visibility)],
                });
                const probeTraceBGL = device.createBindGroupLayout({
                    label: 'VoxelGI/ClipmapProbeTraceBGL',
                    entries: [uniform(0), depth(1), texture(2, false), uniform(6), storage(7), uniform(60), storageBuffer(61)],
                });
                return {
                    trace: pipeline('VoxelGI/ClipmapTrace', CLIPMAP_TRACE_WGSL, bgl),
                    showVoxels: pipeline('VoxelGI/ClipmapVoxels', CLIPMAP_TRACE_WGSL, bgl, 'show_voxels'),
                    bgl,
                    probeTrace: pipeline('VoxelGI/ClipmapProbeTrace', CLIPMAP_PROBE_TRACE_WGSL, probeTraceBGL),
                    probeTraceBGL,
                    noVolume: buffer('VoxelGI/NoVolume', VOXEL_VOLUME_BYTES, GPUBufferUsage.UNIFORM),
                };
            })() : null,
            targets: null,
        };
    }

    private ensureTargets(device: GPUDevice, width: number, height: number): Targets {
        const gpu = this.gpu!;
        const scale = this.traceScale();
        const w = Math.max(Math.ceil(width * scale), 1);
        const h = Math.max(Math.ceil(height * scale), 1);
        if (gpu.targets && gpu.targets.width === w && gpu.targets.height === h) return gpu.targets;
        if (gpu.targets) for (const t of [gpu.targets.trace, ...gpu.targets.history]) t.destroy();
        const target = (label: string) => device.createTexture({
            label,
            size: [w, h],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        gpu.targets = { width: w, height: h, trace: target('VoxelGI/Trace'), history: [target('VoxelGI/HistoryA'), target('VoxelGI/HistoryB')] };
        this.prevViewProj = null;
        return gpu.targets;
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
        if (!gbuffer) throw new Error('VoxelGIEffect reads the GBuffer\'s normals and albedo (render it through a PostProcessingVolume)');
        // the near field's bounce first (its own trace and history)
        const near = this.near?.encodeTrace(device, commandEncoder, gbuffer, input, depth, camera, width, height) ?? null;
        const t = this.ensureTargets(device, width, height);

        const proj = camera.projectionMatrix.internalMat4;
        const view = camera.viewMatrix.internalMat4;
        const viewProj = mat4.multiply(mat4.create(), proj, view);
        const data = new ArrayBuffer(VOXEL_GI_PARAMS_BYTES);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32.set(mat4.invert(mat4.create(), proj), 0);
        f32.set(mat4.invert(mat4.create(), view), 16);
        f32.set(view, 32);
        f32.set(this.prevViewProj ?? viewProj, 48);
        f32.set([width, height, t.width, t.height, ...(near?.size ?? [1, 1])], 64);
        f32[70] = Math.max(this.startVoxels, 0);
        f32[71] = Math.max(this.maxDistanceM, 0);
        u32[72] = VoxelGiQuality.coneSteps(this.quality);
        u32[73] = this.frame;
        f32[74] = Math.max(this.intensity, 0);
        f32[75] = Math.min(Math.max(this.materialAmbient, 0), 1);
        f32[76] = Math.min(Math.max(this.temporalBlend, 0.01), 1);
        u32[77] = this.prevViewProj ? 1 : 0;
        u32[78] = this.skyLighting ? 1 : 0;
        const debug = this.showProbes && this.probes ? 4
            : this.showSdfSlice !== null && this.sdf ? 3
            : this.showVoxels ? 2
            : this.showIndirect ? 1 : 0;
        u32[79] = debug;
        u32[80] = near ? 1 : 0;
        f32[81] = Math.max(this.skyScale, 0);
        f32[82] = this.sdf ? Math.min(Math.max(this.sdfAo, 0), 1) : 0;
        f32[83] = this.showSdfSlice ?? 0;
        u32[84] = this.sdf ? 1 : 0;
        f32[85] = Math.max(this.clipmapLevelBias, 0);
        device.queue.writeBuffer(gpu.params, 0, data);
        if (!this.skyLighting) {
            device.queue.writeBuffer(gpu.gradientSky, 0, gradientSkyLighting(this.skyGradient[0], this.skyGradient[1]));
        }
        const current = this.frame % 2;
        this.frame = (this.frame + 1) >>> 0;
        this.prevViewProj = viewProj;

        const sky = { buffer: this.skyLighting ?? gpu.gradientSky };
        const params = { buffer: gpu.params };
        const depthView = depth.createView();
        const normalView = gbuffer.normalTexture.createView();
        const traceView = t.trace.createView();
        const history = t.history.map((h) => h.createView());
        const group = (label: string, layout: GPUBindGroupLayout, resources: [number, GPUBindingResource][]) =>
            device.createBindGroup({ label, layout, entries: resources.map(([binding, resource]) => ({ binding, resource })) });

        const source = this.source;
        // the trace through the volume (with its anisotropic chains and distance field) or the
        // clipmap, and what the composite binds for the volume
        let trace: GPUBindGroup;
        let volume: [GPUBuffer, GPUTextureView, GPUSampler];
        let tracePipeline: GPUComputePipeline;
        if (source.kind === 'volume') {
            const v = source.volume;
            trace = group('VoxelGI/TraceBG', gpu.traceBGL, [
                [0, params], [1, depthView], [2, normalView], [3, { buffer: v.uniform }], [4, v.view],
                [5, v.sampler], [6, sky], [7, traceView],
                ...source.anisotropic.map((view, i) => [40 + i, view] as [number, GPUBindingResource]),
                [46, this.sdf ?? gpu.noSdf],
            ]);
            volume = [v.uniform, v.view, v.sampler];
            tracePipeline = gpu.trace;
        } else {
            const clipmap = gpu.clipmap!;
            const clipmapGroup = (target: GPUTextureView) => device.createBindGroup({
                label: 'VoxelGI/ClipmapTraceBG',
                layout: clipmap.bgl,
                entries: [
                    ...([[0, params], [1, depthView], [2, normalView], [6, sky], [7, target]] as [number, GPUBindingResource][])
                        .map(([binding, resource]) => ({ binding, resource })),
                    ...clipmapEntries(source.clipmap),
                ],
            });
            if (this.showVoxels) {
                // the clipmap's voxels in place of the lit image
                const pass = commandEncoder.beginComputePass({ label: 'VoxelGI/ClipmapVoxels' });
                pass.setPipeline(clipmap.showVoxels);
                pass.setBindGroup(0, clipmapGroup(output.createView()));
                pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
                pass.end();
                return;
            }
            if (this.clipmapProbes) {
                // the far field from the probes, in place of the cones
                const [grid, probes] = this.clipmapProbes;
                trace = group('VoxelGI/ClipmapProbeTraceBG', clipmap.probeTraceBGL, [
                    [0, params], [1, depthView], [2, normalView], [6, sky], [7, traceView], [60, { buffer: grid }], [61, { buffer: probes }],
                ]);
                tracePipeline = clipmap.probeTrace;
            } else {
                trace = clipmapGroup(traceView);
                tracePipeline = clipmap.trace;
            }
            volume = [clipmap.noVolume, gpu.noSdf, gpu.sampler];
        }
        const temporal = group('VoxelGI/TemporalBG', gpu.temporalBGL, [
            [0, params], [1, traceView], [2, history[1 - current]], [3, depthView], [4, history[current]], [5, gpu.sampler],
        ]);
        // (the clipmap probes' light needs no temporal filter: the composite reads the trace)
        const composite = group('VoxelGI/CompositeBG', gpu.compositeBGL, [
            [0, params], [1, input.createView()], [2, depthView], [3, this.clipmapProbes ? traceView : history[current]], [4, (near?.texture ?? gpu.noNear).createView()],
            [5, gbuffer.albedoTexture.createView()], [6, normalView], [7, sky], [8, output.createView()],
            [9, { buffer: volume[0] }], [10, volume[1]], [11, volume[2]], [12, this.sdf ?? gpu.noSdf],
            ...(this.probes ?? gpu.noProbes).map((buffer, i) => [13 + i, { buffer }] as [number, GPUBindingResource]),
        ]);

        const pass = commandEncoder.beginComputePass({ label: 'VoxelGI/Screen', timestampWrites: gpuPass('VoxelGI/Screen') });
        // with probes the composite reads them per pixel: no cones to trace and accumulate (the
        // voxels and slice views show without them)
        const probes = this.probes !== null && debug !== 2 && debug !== 3;
        if (!probes) {
            pass.setPipeline(tracePipeline);
            pass.setBindGroup(0, trace);
            pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
            if (!this.clipmapProbes) {
                pass.setPipeline(gpu.temporal);
                pass.setBindGroup(0, temporal);
                pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
            }
        }
        pass.setPipeline(probes ? gpu.compositeProbes : gpu.composite);
        pass.setBindGroup(0, composite);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    public resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    public destroy(): void {
        const gpu = this.gpu;
        if (!gpu) return;
        for (const b of [gpu.params, gpu.gradientSky, ...gpu.noProbes]) b.destroy();
        gpu.clipmap?.noVolume.destroy();
        gpu.noNear.destroy();
        this.near?.destroy();
        if (gpu.targets) for (const t of [gpu.targets.trace, ...gpu.targets.history]) t.destroy();
        this.gpu = null;
        this.initialized = false;
    }
}
