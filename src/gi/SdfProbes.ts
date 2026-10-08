import { mat4, quat } from 'gl-matrix';
import { gpuPass } from '../profiling/Profiler';
import { PROBE_UPDATE_WGSL } from './GiWGSL';
import type { JumpFloodSdf } from './JumpFloodSdf';
import type { MeshVoxelizer } from './MeshVoxelizer';
import type { Vec3, VoxelVolume } from './VoxelVolume';

/** Rays a probe traces each update (one workgroup). Rust: `gi::PROBE_RAYS`. */
export const PROBE_RAYS = 64;
const SH_WORDS = 9;
const DEPTH_TEXELS = 64;
/** Bytes of the WGSL `ProbeGrid` (probe_common.wgsl; Rust `ProbeGridGpu`). */
export const PROBE_GRID_BYTES = 64;
/** Bytes of the WGSL `ProbeUpdate` (probe_update.wgsl; Rust `ProbeUpdateGpu`). */
export const PROBE_UPDATE_BYTES = 112;

/**
 * What `SceneVoxelGi.enableProbes` sets up, and how the probes update (`SdfProbes.options`,
 * change it between frames; `spacingVoxels` and `maxProbesPerAxis` take effect only at creation).
 * Rust: `gi::SdfProbeOptions`.
 */
export interface SdfProbeOptions {
    /** Voxels of the volume between probes. */
    spacingVoxels: number;
    /**
     * The most probes per axis (0: as many as cover the volume). The grid follows the camera when
     * it is smaller than the volume.
     */
    maxProbesPerAxis: number;
    /** Probes updated each frame, in turn (0: all of them). */
    probesPerFrame: number;
    /** Weight of the history in each update of the irradiance (0.97: about 30 frames to settle). */
    hysteresis: number;
    /** The same for the depth moments. */
    depthHysteresis: number;
    /** Sharpness of a ray's weight on the depth map's texels around its direction. */
    depthSharpness: number;
    /** Voxels a probe is moved off the surfaces near its lattice point. */
    minClearanceVoxels: number;
    /** Voxels a lookup moves off its surface along the normal before weighing the probes. */
    normalBiasVoxels: number;
    /**
     * A probe whose rays meet more back faces than this share (it sits inside geometry) is left out
     * of the lookups.
     */
    backfaceLimit: number;
    /**
     * Weigh the probes by whether they see the point (their depth moments): stops light leaking
     * through walls.
     */
    visibility: boolean;
    /** Scale of the sky the rays that leave the volume see. */
    skyScale: number;
    /** Most sphere-tracing steps per ray. */
    maxSteps: number;
}

export function defaultSdfProbeOptions(): SdfProbeOptions {
    return {
        spacingVoxels: 8,
        maxProbesPerAxis: 0,
        probesPerFrame: 0,
        hysteresis: 0.97,
        depthHysteresis: 0.97,
        depthSharpness: 32,
        minClearanceVoxels: 2,
        normalBiasVoxels: 3,
        backfaceLimit: 0.25,
        visibility: true,
        skyScale: 1,
        maxSteps: 96,
    };
}

/** Whether two sets of probe options are the same. */
export function sameSdfProbeOptions(a: SdfProbeOptions, b: SdfProbeOptions): boolean {
    return (Object.keys(a) as (keyof SdfProbeOptions)[]).every((k) => a[k] === b[k]);
}

/**
 * Irradiance probes traced in a voxel volume's distance field (the Lumen-like consumer of the
 * scene's voxel GI; `SceneVoxelGi.enableProbes`, which the renderer updates each frame after the
 * volume is lit). A grid of probes on a world lattice follows the camera (inside the volume: by
 * whole cells, each probe keeping its history while it stays in the grid). Each update, a probe:
 * 1. moves off the surfaces near its lattice point, up the distance field's gradient;
 * 2. sphere-traces 64 rays (a spherical Fibonacci set turned at random each frame) through the
 *    field, reading the lit volume where they hit (mip 0 over the anisotropic chain the ray faces:
 *    no material or shadow lookups, the injection did those) and the sky where they leave it;
 * 3. projects their light onto order-2 SH convolved with the cosine lobe (its irradiance) and
 *    their distances onto an 8x8 octahedral map of depth moments, both blended into its history;
 * 4. counts the rays that met the back of a surface: a probe inside geometry is left out.
 *
 * Read it with `PROBES_WGSL`'s `kansei_gi_irradiance(p, n)` (materials, with `bindingsWGSL` and
 * `bindGroupEntries`), or with `VoxelGIEffect.setProbes` (the far field on screen). The lookups
 * weigh the eight probes around a point as DDGI does (Majercik et al. 2019): trilinear, by
 * facing, and by Chebyshev visibility from the depth moments, so walls don't leak.
 * Rust: `gi::SdfProbes`.
 */
export class SdfProbes {
    public options: SdfProbeOptions;
    /** Probes per axis. */
    public readonly dims: Vec3;
    /** Metres between probes. */
    public readonly spacing: number;
    /** The `ProbeGrid` uniform, for readers. */
    public readonly gridBuffer: GPUBuffer;
    /** Each probe's irradiance SH: 9 `vec4f` (rgb). */
    public readonly shBuffer: GPUBuffer;
    /** Each probe's state: 2 `vec4f` (offset and back-face share; lattice cell and frames). */
    public readonly stateBuffer: GPUBuffer;
    /** Each probe's depth moments: 64 `vec2f`. */
    public readonly depthBuffer: GPUBuffer;

    /** World position of lattice cell (0, 0, 0)'s probe. */
    private readonly anchor: Vec3;
    /** Lattice cells the volume spans per axis. */
    private readonly volumeCells: Vec3;
    private base: Vec3 = [0, 0, 0];
    private gridData: Float32Array | null = null;
    private readonly params: GPUBuffer;
    private readonly pipeline: GPUComputePipeline;
    private readonly bgl: GPUBindGroupLayout;
    private group: GPUBindGroup | null = null;
    /** What `group` binds: the dynamic surfaces, the sky and the field. */
    private bound: [GPUBuffer | null, GPUBuffer | null, JumpFloodSdf | null] = [null, null, null];
    private readonly noSurfaces: GPUBuffer;
    private cursor = 0;
    private frame = 0;
    private resetNext = true;

    constructor(private readonly device: GPUDevice, volume: VoxelVolume, options: SdfProbeOptions) {
        this.options = { ...options };
        const layout = volume.layout;
        this.spacing = layout.voxelSize * Math.max(options.spacingVoxels, 1);
        const extent = layout.dims.map((d) => d * layout.voxelSize);
        this.volumeCells = extent.map((e) => Math.max(Math.floor(e / this.spacing), 1)) as Vec3;
        const cap = Math.min(options.maxProbesPerAxis === 0 ? Infinity : Math.max(options.maxProbesPerAxis, 2),
            // (one workgroup each, within a dispatch's 65535)
            40);
        this.dims = this.volumeCells.map((c) => Math.min(Math.max(c, 2), cap)) as Vec3;
        // the lattice is centred on the volume, so a grid that covers it is symmetric in it
        this.anchor = [0, 1, 2].map((i) => layout.origin[i] + 0.5 * (extent[i] - (this.volumeCells[i] - 1) * this.spacing)) as Vec3;
        const count = this.probeCount;
        const storage = (label: string, size: number) =>
            device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        const uniform = (label: string, size: number) => device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.gridBuffer = uniform('VoxelGI/ProbeGrid', PROBE_GRID_BYTES);
        this.params = uniform('VoxelGI/ProbeUpdate', PROBE_UPDATE_BYTES);
        this.shBuffer = storage('VoxelGI/ProbeSh', count * SH_WORDS * 16);
        this.stateBuffer = storage('VoxelGI/ProbeState', count * 2 * 16);
        this.depthBuffer = storage('VoxelGI/ProbeDepth', count * DEPTH_TEXELS * 8);
        this.noSurfaces = device.createBuffer({ label: 'VoxelGI/ProbeNoSurfaces', size: 16, usage: GPUBufferUsage.STORAGE });

        const visibility = GPUShaderStage.COMPUTE;
        const texture: GPUTextureBindingLayout = { sampleType: 'float', viewDimension: '3d' };
        const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility, buffer: { type: 'uniform' } },
            { binding: 1, visibility, buffer: { type: 'uniform' } },
            { binding: 2, visibility, buffer: { type: 'uniform' } },
            { binding: 3, visibility, texture },
            { binding: 4, visibility, sampler: { type: 'filtering' } },
            { binding: 5, visibility, buffer: { type: 'uniform' } },
            { binding: 6, visibility, buffer: { type: 'read-only-storage' } },
            { binding: 7, visibility, buffer: { type: 'read-only-storage' } },
            { binding: 8, visibility, buffer: { type: 'storage' } },
            { binding: 9, visibility, buffer: { type: 'storage' } },
            { binding: 10, visibility, buffer: { type: 'storage' } },
        ];
        // the anisotropic chains and the distance field (voxel_irradiance.wgsl)
        for (let binding = 40; binding < 47; binding++) entries.push({ binding, visibility, texture });
        this.bgl = device.createBindGroupLayout({ label: 'VoxelGI/ProbesBGL', entries });
        this.pipeline = device.createComputePipeline({
            label: 'VoxelGI/Probes',
            layout: device.createPipelineLayout({ label: 'VoxelGI/Probes', bindGroupLayouts: [this.bgl] }),
            compute: { module: device.createShaderModule({ label: 'VoxelGI/Probes', code: PROBE_UPDATE_WGSL }), entryPoint: 'main' },
        });
        this.base = this.baseFor(null);
    }

    public get probeCount(): number {
        return this.dims[0] * this.dims[1] * this.dims[2];
    }

    /** World position of the grid's first probe now (before its offset). */
    public get gridOrigin(): Vec3 {
        return [0, 1, 2].map((i) => this.anchor[i] + this.base[i] * this.spacing) as Vec3;
    }

    /** Bytes on the GPU. */
    public memoryBytes(): number {
        return this.shBuffer.size + this.stateBuffer.size + this.depthBuffer.size;
    }

    /**
     * Start every probe over next update (after a cut, or a change of the scene's lighting the
     * hysteresis should not blend through).
     */
    public reset(): void {
        this.resetNext = true;
    }

    /**
     * The declarations `PROBES_WGSL` reads, at `group` and the four bindings from `first`: the
     * grid, the SH, the state and the depth moments (`bindGroupEntries` binds them).
     */
    public static bindingsWGSL(group: number, first: number): string {
        return `@group(${group}) @binding(${first}) var<uniform> kansei_probe_grid : ProbeGrid;\n`
            + `@group(${group}) @binding(${first + 1}) var<storage, read> kansei_probe_sh : array<vec4f>;\n`
            + `@group(${group}) @binding(${first + 2}) var<storage, read> kansei_probe_state : array<vec4f>;\n`
            + `@group(${group}) @binding(${first + 3}) var<storage, read> kansei_probe_depth : array<vec2f>;\n`;
    }

    /** The buffers for `bindingsWGSL(_, first)`, in order. */
    public bindGroupEntries(first: number): GPUBindGroupEntry[] {
        return [this.gridBuffer, this.shBuffer, this.stateBuffer, this.depthBuffer]
            .map((buffer, i) => ({ binding: first + i, resource: { buffer } }));
    }

    /**
     * The lattice cell of the grid's first probe for a camera at `eye`: the grid centred on it,
     * kept inside the volume (centred on the volume when it is larger).
     */
    private baseFor(eye: Vec3 | null): Vec3 {
        return [0, 1, 2].map((i) => {
            const dims = this.dims[i];
            const room = this.volumeCells[i] - dims;
            if (room <= 0) return Math.trunc(room / 2);
            const center = eye ? roundHalfAway((eye[i] - this.anchor[i]) / this.spacing) : Math.trunc(this.volumeCells[i] / 2);
            return Math.min(Math.max(center - Math.trunc(dims / 2), 0), room);
        }) as Vec3;
    }

    private packGrid(): Float32Array {
        const voxel = this.spacing / Math.max(this.options.spacingVoxels, 1);
        const data = new ArrayBuffer(PROBE_GRID_BYTES);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        const i32 = new Int32Array(data);
        f32.set(this.gridOrigin, 0);
        f32[3] = this.spacing;
        u32.set(this.dims, 4);
        u32[7] = this.probeCount;
        i32.set(this.base, 8);
        f32[11] = Math.max(this.options.normalBiasVoxels, 0) * voxel;
        f32[12] = this.options.backfaceLimit;
        f32[13] = 1.5 * this.spacing * Math.sqrt(3);
        u32[14] = this.options.visibility ? 1 : 0;
        return f32;
    }

    /**
     * Record an update of the probes (after the volume's light and mips): the grid follows `eye`,
     * then the next `probesPerFrame` probes trace.
     */
    public encode(
        encoder: GPUCommandEncoder,
        volume: VoxelVolume,
        sdf: JumpFloodSdf,
        voxelizer: MeshVoxelizer,
        sky: GPUBuffer,
        eye: Vec3 | null,
    ): void {
        const device = this.device;
        this.base = this.baseFor(eye);
        const grid = this.packGrid();
        if (!this.gridData || grid.some((v, i) => !Object.is(v, this.gridData![i]))) {
            device.queue.writeBuffer(this.gridBuffer, 0, grid);
            this.gridData = grid;
        }
        const count = this.probeCount;
        const o = this.options;
        const batch = Math.min(o.probesPerFrame === 0 ? count : Math.min(o.probesPerFrame, count), 65535);
        // a random turn of the ray set each frame (a uniform quaternion, Shoemake 1992)
        const h = (k: number) => {
            let x = (Math.imul(this.frame, 0x9e3779b9) + Math.imul(k, 0x85ebca6b)) >>> 0;
            x = (x ^ (x >>> 16)) >>> 0;
            x = Math.imul(x, 0x7feb352d) >>> 0;
            x = (x ^ (x >>> 15)) >>> 0;
            x = Math.imul(x, 0x846ca68b) >>> 0;
            x = (x ^ (x >>> 16)) >>> 0;
            return x / 0xffffffff;
        };
        const [u1, u2, u3] = [h(1), h(2) * 2 * Math.PI, h(3) * 2 * Math.PI];
        const q = quat.normalize(quat.create(), quat.fromValues(
            Math.sqrt(1 - u1) * Math.sin(u2), Math.sqrt(1 - u1) * Math.cos(u2), Math.sqrt(u1) * Math.sin(u3), Math.sqrt(u1) * Math.cos(u3)));
        const dynamic = voxelizer.dynamicSurfaces;
        const data = new ArrayBuffer(PROBE_UPDATE_BYTES);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32.set(mat4.fromQuat(mat4.create(), q), 0);
        u32[16] = this.cursor;
        f32[17] = Math.min(Math.max(o.hysteresis, 0), 0.999);
        f32[18] = Math.min(Math.max(o.depthHysteresis, 0), 0.999);
        f32[19] = Math.max(o.skyScale, 0);
        u32[20] = Math.max(o.maxSteps, 1);
        f32[21] = Math.max(o.minClearanceVoxels, 0) * volume.voxelSize;
        f32[22] = Math.max(o.depthSharpness, 1);
        u32[23] = this.resetNext ? 1 : 0;
        u32[24] = dynamic ? 1 : 0;
        device.queue.writeBuffer(this.params, 0, data);
        this.resetNext = false;
        this.cursor = (this.cursor + batch) % count;
        this.frame = (this.frame + 1) >>> 0;

        if (this.bound[0] !== dynamic || this.bound[1] !== sky || this.bound[2] !== sdf) this.group = null;
        if (!this.group) {
            const anisotropic = volume.anisotropicViews;
            if (!anisotropic) throw new Error('SdfProbes read a volume with anisotropic mips');
            this.group = device.createBindGroup({
                label: 'VoxelGI/ProbesBG',
                layout: this.bgl,
                entries: [
                    { binding: 0, resource: { buffer: volume.uniform } },
                    { binding: 1, resource: { buffer: this.gridBuffer } },
                    { binding: 2, resource: { buffer: this.params } },
                    { binding: 3, resource: volume.view },
                    { binding: 4, resource: volume.sampler },
                    { binding: 5, resource: { buffer: sky } },
                    { binding: 6, resource: { buffer: voxelizer.staticSurfaces } },
                    { binding: 7, resource: { buffer: dynamic ?? this.noSurfaces } },
                    { binding: 8, resource: { buffer: this.shBuffer } },
                    { binding: 9, resource: { buffer: this.stateBuffer } },
                    { binding: 10, resource: { buffer: this.depthBuffer } },
                    ...anisotropic.map((view, i) => ({ binding: 40 + i, resource: view })),
                    { binding: 46, resource: sdf.view },
                ],
            });
            this.bound = [dynamic, sky, sdf];
        }

        const pass = encoder.beginComputePass({ label: 'VoxelGI/Probes', timestampWrites: gpuPass('VoxelGI/Probes') });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.group);
        pass.dispatchWorkgroups(batch);
        pass.end();
    }

    public destroy(): void {
        for (const b of [this.gridBuffer, this.params, this.shBuffer, this.stateBuffer, this.depthBuffer, this.noSurfaces]) b.destroy();
    }
}

/** Rust's `f32::round`: halves away from zero. */
function roundHalfAway(x: number): number {
    return Math.sign(x) * Math.round(Math.abs(x));
}
