import { mat4 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { GBuffer } from '../postprocessing/GBuffer';
import { PostProcessingEffect } from '../postprocessing/PostProcessingEffect';
import { gpuPass } from '../profiling/Profiler';
import { VoxelVolume } from '../gi/VoxelVolume';
import { VoxelClipmap, clipmapEntries, clipmapLayoutEntries } from '../gi/VoxelClipmap';
import { RtGrid, RtGridHandle } from './RtGrid';
import { RT_DEFAULT_COVERED_WGSL, RT_REFLECT_RESOLVE_WGSL, rtReflectTraceWgsl } from './RtWGSL';

/** What the reflections trace at: one pixel of each 2 x 2 (`half`) or 4 x 4 (`quarter`) block a frame. Rust: `rt::RtTraceResolution`. */
export type RtTraceResolution = 'half' | 'quarter';

/**
 * What the screen shows: the lit image with the reflections (`lit`), the light the reflections
 * add alone, Fresnel applied (`reflection`), what the reflection rays see without the Fresnel
 * (`mirror`), or the rays' cost, cells visited and triangles tested, blue to red over 0-400
 * (`cost`). Rust: `rt::RtReflectionsView`.
 */
export type RtReflectionsView = 'lit' | 'reflection' | 'mirror' | 'cost';
const VIEWS: RtReflectionsView[] = ['lit', 'reflection', 'mirror', 'cost'];

/** What `RtReflectionsEffect` sets up. Rust: `rt::RtReflectionsOptions`. */
export interface RtReflectionsOptions {
    /** Default `half`. */
    resolution?: RtTraceResolution;
    /** Scale of the reflection (1 is physical). Default 1. */
    intensity?: number;
    /** Metres a reflection looks, through the grid then the voxels. Default 400. */
    maxDistance?: number;
    /** tan of the half-angle of the cone rays take through the voxels past the grid (rough surfaces widen it to their lobe). Default 0.04. */
    coneTan?: number;
    /** Default 96. */
    coneSteps?: number;
    /** Weight of each new frame in the accumulated reflection. Default 0.25. */
    temporalBlend?: number;
    /** Scale of the sky past the voxels. Default 1. */
    skyScale?: number;
    /** Alpha-test the grid's alpha-tested triangles (`RtSurface.alphaLayer`); off, they are solid. Default true. */
    alphaTest?: boolean;
    /**
     * WGSL defining `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, which may sample
     * `kansei_rt_alpha_texture` with `kansei_rt_alpha_sampler` (`setAlphaTexture`). Default: the
     * texture's alpha is at least a half (a white texture until one is set: every hit).
     */
    coveredWgsl?: string | null;
}

/** The trace's counters of a recent frame (`collectStats`). Rust: `rt::RtReflectionStats`. */
export interface RtReflectionStats {
    /** Reflective pixels traced, and those whose ray hit a triangle of the grid. */
    rays: number;
    hits: number;
    /** Cells visited and triangles tested, by all the rays. */
    cells: number;
    tests: number;
    /** The most cells and triangles one ray visited. */
    maxCost: number;
}

/** Bytes of the WGSL `RtReflectParams` (rt_reflect_common.wgsl; Rust `RtReflectParamsGpu`). */
export const RT_REFLECT_PARAMS_BYTES = 256;

interface Targets {
    width: number;
    height: number;
    trace: GPUTexture;
    history: [GPUTexture, GPUTexture];
}

interface Gpu {
    params: GPUBuffer;
    traceBGL: GPUBindGroupLayout;
    gridBGL: GPUBindGroupLayout;
    resolveBGL: GPUBindGroupLayout;
    trace: GPUComputePipeline;
    resolve: GPUComputePipeline;
    noSky: GPUBuffer;
    white: GPUTexture;
    alphaSampler: GPUSampler;
    linear: GPUSampler;
    stats: GPUBuffer;
    staging: GPUBuffer;
    /** the grid's group, with the grid generation and the alpha texture it was made with */
    gridGroup: { generation: number; alpha: GPUTextureView | null; group: GPUBindGroup } | null;
    targets: Targets | null;
}

/** The stats readback: nothing in flight, copied this frame (awaiting its submit), mapping. */
const enum StatsState { Free, Copied, Mapping }

/**
 * Sharp and glossy reflections, traced through the renderer's ray tracing grid
 * (`Renderer.enableRtGrid`; `SceneRtGrid.handle`) on the surfaces whose material writes an F0
 * (`GBUFFER_OUT_WGSL`'s `kansei_gbuffer_out_specular`), the hits lit by the voxel GI's volume
 * (`SceneVoxelGi.volume`) or clipmap (`withClipmap`, `SceneVoxelClipmap.clipmap`): the light
 * leaving the surface there, the voxels as the surface cache.
 * Rays leaving the grid's box go on as a narrow voxel cone, then the sky (`setSkyLighting`).
 *
 * It traces one pixel of each 2 x 2 (or 4 x 4) block a frame, each in turn, and accumulates them
 * at full resolution: the frame's traced pixels upsampled by depth and normal, blended with the
 * history reprojected by the surface and clamped to their range. The composite is the lit colour
 * times 1 - F plus F times the reflection, F Schlick's Fresnel of the F0 (lessened on rough
 * surfaces, whose rays jitter over their lobe). Put it after the GI and before the tone mapping.
 * Needs a single-sampled GBuffer (its alphas are the F0 and roughness).
 *
 * Rust: `rt::RtReflectionsEffect` (`with_volume`, `with_clipmap`).
 */
export class RtReflectionsEffect extends PostProcessingEffect {
    public enabled = true;
    public view: RtReflectionsView = 'lit';
    public intensity: number;
    public maxDistance: number;
    public coneTan: number;
    public coneSteps: number;
    public temporalBlend: number;
    public skyScale: number;
    public alphaTest: boolean;
    /** Trace the grid of triangles (on by default); off, every reflection is the voxel cone from the surface (for comparison). */
    public traceGrid = true;
    /** Scale of the cost view's colours. */
    public heatScale = 1;
    /** Count the rays' work (`stats`), a few atomics a ray. */
    public collectStats = false;

    private _resolution: RtTraceResolution;
    private readonly coveredWgsl: string | null;
    private readonly grid: RtGridHandle;
    private readonly source: VoxelVolume | VoxelClipmap;
    private skyLighting: GPUBuffer | null = null;
    private alphaTexture: GPUTextureView | null = null;
    private frame = 0;
    private prevViewProj: mat4 | null = null;
    private lastCameraFrame: number | null = null;
    private statsState = StatsState.Free;
    private _stats: RtReflectionStats | null = null;
    private gpu: Gpu | null = null;
    private device: GPUDevice | null = null;

    /**
     * Reflections through `grid` (`SceneRtGrid.handle`), lit by `volume` (`SceneVoxelGi.volume`),
     * or by a clipmap (as `withClipmap`).
     */
    constructor(volume: VoxelVolume | VoxelClipmap, grid: RtGridHandle, options: RtReflectionsOptions = {}) {
        super();
        this.source = volume;
        this.grid = grid;
        this._resolution = options.resolution ?? 'half';
        this.intensity = options.intensity ?? 1;
        this.maxDistance = options.maxDistance ?? 400;
        this.coneTan = options.coneTan ?? 0.04;
        this.coneSteps = options.coneSteps ?? 96;
        this.temporalBlend = options.temporalBlend ?? 0.25;
        this.skyScale = options.skyScale ?? 1;
        this.alphaTest = options.alphaTest ?? true;
        this.coveredWgsl = options.coveredWgsl ?? null;
    }

    /**
     * Reflections lit by a voxel clipmap (`SceneVoxelClipmap.clipmap`) through `grid`. Rust:
     * `RtReflectionsEffect::with_clipmap`.
     */
    public static withClipmap(clipmap: VoxelClipmap, grid: RtGridHandle, options: RtReflectionsOptions = {}): RtReflectionsEffect {
        return new RtReflectionsEffect(clipmap, grid, options);
    }

    /** The sky past the voxels (a `SkyLighting` uniform); black without one. */
    public setSkyLighting(sky: GPUBuffer | null): void {
        this.skyLighting = sky;
    }

    /** The texture `kansei_rt_covered` reads (`kansei_rt_alpha_texture`). */
    public setAlphaTexture(view: GPUTextureView | null): void {
        this.alphaTexture = view;
    }

    public get resolution(): RtTraceResolution {
        return this._resolution;
    }

    /** Trace at another resolution (the targets are made anew). */
    public setResolution(resolution: RtTraceResolution): void {
        if (resolution === this._resolution) return;
        this._resolution = resolution;
        if (this.gpu?.targets) {
            for (const t of [this.gpu.targets.trace, ...this.gpu.targets.history]) t.destroy();
            this.gpu.targets = null;
        }
        this.resetHistory();
    }

    /** Start the accumulation over (after a cut, or a change the history should not blend through). */
    public resetHistory(): void {
        this.prevViewProj = null;
    }

    /** The counters of a recent frame, while `collectStats` is on (they arrive a few frames late). */
    public get stats(): RtReflectionStats | null {
        return this._stats;
    }

    public isActive(): boolean {
        return this.enabled;
    }

    private downscale(): number {
        return this._resolution === 'quarter' ? 4 : 2;
    }

    public initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this.device = device;
        this.gpu ??= this.initGpu(device);
        this.initialized = true;
    }

    private initGpu(device: GPUDevice): Gpu {
        const visibility = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } });
        const texture = (binding: number, filterable: boolean, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, texture: { sampleType: filterable ? 'float' : 'unfilterable-float', viewDimension } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth' } });
        const storage = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } });
        const sampler = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, sampler: { type: 'filtering' } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });

        const clipmap = this.source instanceof VoxelClipmap;
        const traceBGL = bgl('RtReflections/Trace', [
            uniform(0), depth(1), texture(2, false), texture(3, false), storage(4), uniform(5),
            ...(clipmap ? clipmapLayoutEntries(visibility) : [uniform(6), texture(7, true, '3d'), sampler(8)]),
        ]);
        const gridBGL = bgl('RtReflections/Grid', [
            ...RtGrid.layoutEntries(0, visibility), texture(3, true), sampler(4),
            { binding: 5, visibility, buffer: { type: 'storage' } },
        ]);
        const resolveBGL = bgl('RtReflections/Resolve', [
            uniform(0), depth(1), texture(2, false), texture(3, false), texture(4, false), texture(5, false),
            texture(6, true), storage(7), storage(8), sampler(9),
        ]);
        const pipeline = (label: string, code: string, layouts: GPUBindGroupLayout[]) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: layouts }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
        });
        const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ label, size, usage });
        const white = device.createTexture({ label: 'RtReflections/White', size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1]);
        const filtering = (label: string, addressMode: GPUAddressMode) => device.createSampler({ label, magFilter: 'linear', minFilter: 'linear', addressModeU: addressMode, addressModeV: addressMode });
        return {
            params: buffer('RtReflections/Params', RT_REFLECT_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
            traceBGL,
            gridBGL,
            resolveBGL,
            trace: pipeline('RtReflections/Trace', rtReflectTraceWgsl(this.coveredWgsl ?? RT_DEFAULT_COVERED_WGSL, clipmap), [traceBGL, gridBGL]),
            resolve: pipeline('RtReflections/Resolve', RT_REFLECT_RESOLVE_WGSL, [resolveBGL]),
            // (a zeroed SkyLighting: black)
            noSky: buffer('RtReflections/NoSky', 256, GPUBufferUsage.UNIFORM),
            white,
            alphaSampler: filtering('RtReflections/Alpha', 'repeat'),
            linear: filtering('RtReflections/Linear', 'clamp-to-edge'),
            stats: buffer('RtReflections/Stats', 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
            staging: buffer('RtReflections/StatsReadback', 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
            gridGroup: null,
            targets: null,
        };
    }

    private ensureTargets(device: GPUDevice, width: number, height: number): Targets {
        const gpu = this.gpu!;
        if (gpu.targets && gpu.targets.width === width && gpu.targets.height === height) return gpu.targets;
        if (gpu.targets) for (const t of [gpu.targets.trace, ...gpu.targets.history]) t.destroy();
        const d = this.downscale();
        const target = (label: string, w: number, h: number) => device.createTexture({
            label,
            size: [w, h],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        gpu.targets = {
            width,
            height,
            trace: target('RtReflections/Trace', Math.ceil(width / d), Math.ceil(height / d)),
            history: [target('RtReflections/HistoryA', width, height), target('RtReflections/HistoryB', width, height)],
        };
        this.prevViewProj = null;
        return gpu.targets;
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
                    if (this.collectStats) this._stats = { rays: w[0], hits: w[1], cells: w[2], tests: w[3], maxCost: w[4] };
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
        if (!gbuffer) throw new Error('RtReflectionsEffect reads the GBuffer\'s normals and albedo (render it through a PostProcessingVolume)');
        this.pollStats();
        const t = this.ensureTargets(device, width, height);
        const d = this.downscale();
        const tw = Math.ceil(width / d);
        const th = Math.ceil(height / d);

        // a frame skipped (the effect was off, a cut): no history
        const cameraFrame = camera.frame;
        if (this.lastCameraFrame !== null && cameraFrame !== this.lastCameraFrame && cameraFrame !== this.lastCameraFrame + 1) {
            this.prevViewProj = null;
        }
        this.lastCameraFrame = cameraFrame;
        const proj = camera.projectionMatrix.internalMat4;
        const view = camera.viewMatrix.internalMat4;
        const viewProj = mat4.multiply(mat4.create(), proj, view);
        let flags = 0;
        if (this.alphaTest) flags |= 1;
        if (this.collectStats) flags |= 2;
        if (this.prevViewProj) flags |= 4;
        if (this.traceGrid) flags |= 8;
        const data = new ArrayBuffer(RT_REFLECT_PARAMS_BYTES);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32.set(mat4.invert(mat4.create(), proj), 0);
        f32.set(mat4.invert(mat4.create(), view), 16);
        f32.set(this.prevViewProj ?? viewProj, 32);
        f32.set([width, height, tw, th], 48);
        u32[52] = this.frame;
        u32[53] = d;
        u32[54] = flags;
        u32[55] = VIEWS.indexOf(this.view);
        f32[56] = Math.max(this.coneTan, 1e-3);
        u32[57] = this.coneSteps;
        f32[58] = Math.max(this.maxDistance, 0);
        f32[59] = Math.max(this.intensity, 0);
        f32[60] = Math.max(this.skyScale, 0);
        f32[61] = Math.min(Math.max(this.temporalBlend, 0.01), 1);
        f32[62] = this.heatScale;
        device.queue.writeBuffer(gpu.params, 0, data);
        const current = this.frame % 2;
        this.frame = (this.frame + 1) >>> 0;
        this.prevViewProj = viewProj;

        const params = { buffer: gpu.params };
        const depthView = depth.createView();
        const normalView = gbuffer.normalTexture.createView();
        const albedoView = gbuffer.albedoTexture.createView();
        const traceView = t.trace.createView();
        const group = (label: string, layout: GPUBindGroupLayout, resources: [number, GPUBindingResource][]) =>
            device.createBindGroup({ label, layout, entries: resources.map(([binding, resource]) => ({ binding, resource })) });
        const source = this.source;
        const trace = device.createBindGroup({
            label: 'RtReflections/Trace',
            layout: gpu.traceBGL,
            entries: [
                ...([[0, params], [1, depthView], [2, normalView], [3, albedoView], [4, traceView],
                    [5, { buffer: this.skyLighting ?? gpu.noSky }]] as [number, GPUBindingResource][])
                    .map(([binding, resource]) => ({ binding, resource })),
                ...(source instanceof VoxelClipmap ? clipmapEntries(source) : [
                    { binding: 6, resource: { buffer: source.uniform } },
                    { binding: 7, resource: source.view },
                    { binding: 8, resource: source.sampler },
                ]),
            ],
        });
        // the grid's group, made anew when the grid's buffers or the alpha texture change
        const grid = this.grid;
        if (!gpu.gridGroup || gpu.gridGroup.generation !== grid.generation || gpu.gridGroup.alpha !== this.alphaTexture) {
            gpu.gridGroup = {
                generation: grid.generation,
                alpha: this.alphaTexture,
                group: group('RtReflections/Grid', gpu.gridBGL, [
                    [0, { buffer: grid.buffers[0] }], [1, { buffer: grid.buffers[1] }], [2, { buffer: grid.buffers[2] }],
                    [3, this.alphaTexture ?? gpu.white.createView()], [4, gpu.alphaSampler], [5, { buffer: gpu.stats }],
                ]),
            };
        }
        const history = t.history.map((h) => h.createView());
        const resolve = group('RtReflections/Resolve', gpu.resolveBGL, [
            [0, params], [1, depthView], [2, normalView], [3, albedoView], [4, input.createView()], [5, traceView],
            [6, history[1 - current]], [7, output.createView()], [8, history[current]], [9, gpu.linear],
        ]);

        const readStats = this.collectStats && this.statsState === StatsState.Free;
        if (readStats) commandEncoder.clearBuffer(gpu.stats);
        {
            const pass = commandEncoder.beginComputePass({ label: 'Rt/Trace', timestampWrites: gpuPass('Rt/Trace') });
            pass.setPipeline(gpu.trace);
            pass.setBindGroup(0, trace);
            pass.setBindGroup(1, gpu.gridGroup.group);
            pass.dispatchWorkgroups(Math.ceil(tw / 8), Math.ceil(th / 8));
            pass.end();
        }
        if (readStats) {
            commandEncoder.copyBufferToBuffer(gpu.stats, 0, gpu.staging, 0, 32);
            this.statsState = StatsState.Copied;
        }
        const pass = commandEncoder.beginComputePass({ label: 'Rt/Resolve', timestampWrites: gpuPass('Rt/Resolve') });
        pass.setPipeline(gpu.resolve);
        pass.setBindGroup(0, resolve);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    public resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    public destroy(): void {
        const gpu = this.gpu;
        if (!gpu) return;
        for (const b of [gpu.params, gpu.noSky, gpu.stats, gpu.staging]) b.destroy();
        gpu.white.destroy();
        if (gpu.targets) for (const t of [gpu.targets.trace, ...gpu.targets.history]) t.destroy();
        this.gpu = null;
        this.initialized = false;
    }
}
