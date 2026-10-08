import { mat4, quat } from 'gl-matrix';
import { gpuPass } from '../profiling/Profiler';
import { CLIPMAP_PROBE_UPDATE_WGSL } from './GiWGSL';
import { ClipmapLayout, MAX_CLIPMAP_LEVELS, VoxelClipmap, clipmapEntries, clipmapLayoutEntries } from './VoxelClipmap';
import type { Vec3 } from './VoxelVolume';

/**
 * What `SceneVoxelClipmap.enableProbes` sets up, and how the probes update
 * (`ClipmapProbes.options`; `levels` and `spacingVoxels` take effect at creation).
 * Rust: `gi::ClipmapProbeOptions`.
 */
export interface ClipmapProbeOptions {
    /** Levels of probes, the clipmap's finest first (at most its levels). */
    levels: number;
    /**
     * Voxels of the matching clipmap level between probes: each probe level spans that level's
     * window with `dims / spacingVoxels` probes.
     */
    spacingVoxels: number;
    /**
     * Probes traced each frame over all levels, in turn (the probes of slabs a window moves into
     * are traced at once besides).
     */
    probesPerFrame: number;
    /** Weight of the history in each update (0.9: about ten updates to settle). */
    hysteresis: number;
    /** Scale of the sky the cones that leave the clipmap see. */
    skyScale: number;
    /**
     * Probe spacings a lookup moves off its surface along the normal, so the probes it blends lie
     * in front of the surface (1: none of them below a floor).
     */
    normalBias: number;
    /** Most steps per cone. */
    maxSteps: number;
    /**
     * Levels finer than its width each cone reads (`clipConeTraceNear`): each becomes a sparsely
     * sampled ray that sees through gaps narrower than it (a road between trees), the probes'
     * history averaging the samples; 0 reads the levels as wide as the cones, which fill such gaps.
     */
    levelBias: number;
}

export function defaultClipmapProbeOptions(): ClipmapProbeOptions {
    return { levels: MAX_CLIPMAP_LEVELS, spacingVoxels: 2, probesPerFrame: 8192, hysteresis: 0.9, skyScale: 1, normalBias: 1, maxSteps: 32, levelBias: 4 };
}

/** Whether two sets of probe options are the same. */
export function sameClipmapProbeOptions(a: ClipmapProbeOptions, b: ClipmapProbeOptions): boolean {
    return (Object.keys(a) as (keyof ClipmapProbeOptions)[]).every((k) => a[k] === b[k]);
}

/** Bytes of the WGSL `ClipProbeGrid` (clipmap_probes.wgsl; Rust `ClipProbeGridGpu`). */
export const CLIP_PROBE_GRID_BYTES = 32 + 16 * MAX_CLIPMAP_LEVELS;
/** Bytes of the WGSL `ClipProbeUpdate` (clipmap_probe_update.wgsl; Rust `ClipProbeUpdateGpu`). */
export const CLIP_PROBE_UPDATE_BYTES = 144;

const MODE_ROUND = 0;
const MODE_REGION = 1;
const MODE_CLEAR = 2;
/** Dispatches a frame may record at most (a slot of parameters each). */
const MAX_DISPATCHES = 5 * MAX_CLIPMAP_LEVELS;
/** Probes a window moves in steps of. */
const SNAP = 2;

interface Dispatch {
    level: number;
    voxelLevel: number;
    mode: number;
    first: number;
    count: number;
    lo: Vec3;
    size: Vec3;
    /** Workgroups: a probe each (16 slots each when clearing). */
    workgroups: number;
}

/**
 * Irradiance probes of a voxel clipmap (`SceneVoxelClipmap.enableProbes`): the far field of the
 * scene's light at a few probes a metre instead of cones per pixel, and the sky each point sees
 * past the forest. A level of probes per level of the clipmap (or fewer), each a window round the
 * camera of a lattice `spacingVoxels` of that level's voxels apart, stored toroidally as the
 * clipmap's voxels are. Each probe traces 16 cones through the clipmap, the light they gather and
 * the sky past it projected onto order-1 spherical harmonics with the share of each direction that
 * reaches the sky, blended into its history (`hysteresis`); a probe inside a surface is left out.
 * A window moves with the camera in steps of two probes, and the probes of the slab it moved into
 * are traced at once; the others update in turn (`probesPerFrame`).
 *
 * Read them with `CLIPMAP_PROBES_WGSL`'s `kansei_clipmap_light(p, n)` (irradiance and sky
 * visibility) or `kansei_clipmap_sky_visibility(p, n)` in a material, with `bindingsWGSL` and
 * `bindGroupEntries`; with `VoxelGIEffect.setClipmapProbes`, the far field on screen; or with
 * `VolumetricFogEffect.setClipmapProbes`, the fog's ambient. Rust: `gi::ClipmapProbes`.
 */
export class ClipmapProbes {
    public options: ClipmapProbeOptions;
    /** The probes' lattice: per level, `dims` probes `voxelSize * 2^level` metres apart. */
    public readonly layout: ClipmapLayout;
    /** The `ClipProbeGrid` uniform. */
    public readonly gridBuffer: GPUBuffer;
    /** The probes: 4 `vec4f` each (clipmap_probes.wgsl), level after level. */
    public readonly probeBuffer: GPUBuffer;

    /** The clipmap level whose voxels are half the probes' spacing, per probe level 0. */
    private readonly voxelLevel0: number;
    /** Per level: its window's first lattice point, once placed. */
    private readonly origins: (Vec3 | null)[] = Array.from({ length: MAX_CLIPMAP_LEVELS }, () => null);
    private writtenGrid: Uint8Array | null = null;
    private readonly params: GPUBuffer;
    private readonly paramsStride: number;
    private readonly pipeline: GPUComputePipeline;
    private readonly bgl: GPUBindGroupLayout;
    private group: GPUBindGroup | null = null;
    /** The sky `group` binds. */
    private boundSky: GPUBuffer | null = null;
    private readonly cursors: number[];
    private frame = 0;

    constructor(private readonly device: GPUDevice, private readonly clipmap: VoxelClipmap, options: ClipmapProbeOptions) {
        this.options = { ...options };
        const voxels = clipmap.layout;
        const spacing = Math.max(Math.floor(options.spacingVoxels), 1);
        const dims = voxels.dims.map((d) => Math.max(Math.floor(d / spacing), 8)) as Vec3;
        this.layout = new ClipmapLayout(Math.min(Math.max(options.levels, 1), voxels.levels), dims, voxels.voxelSize * spacing);
        // voxels half the spacing: level 0's for 2 voxels, level 1's for 4, ...
        this.voxelLevel0 = Math.min(Math.max(Math.floor(Math.log2(spacing)) - 1, 0), voxels.levels - 1);

        const visibility = GPUShaderStage.COMPUTE;
        this.bgl = device.createBindGroupLayout({
            label: 'VoxelClipmap/ProbesBGL',
            entries: [
                { binding: 0, visibility, buffer: { type: 'uniform' } },
                { binding: 1, visibility, buffer: { type: 'storage' } },
                { binding: 2, visibility, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: CLIP_PROBE_UPDATE_BYTES } },
                { binding: 3, visibility, buffer: { type: 'uniform' } },
                ...clipmapLayoutEntries(visibility),
            ],
        });
        this.pipeline = device.createComputePipeline({
            label: 'VoxelClipmap/Probes',
            layout: device.createPipelineLayout({ label: 'VoxelClipmap/Probes', bindGroupLayouts: [this.bgl] }),
            compute: { module: device.createShaderModule({ label: 'VoxelClipmap/Probes', code: CLIPMAP_PROBE_UPDATE_WGSL }), entryPoint: 'main' },
        });
        const alignment = device.limits.minUniformBufferOffsetAlignment;
        this.paramsStride = Math.ceil(CLIP_PROBE_UPDATE_BYTES / alignment) * alignment;
        this.gridBuffer = device.createBuffer({ label: 'VoxelClipmap/ProbeGrid', size: CLIP_PROBE_GRID_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.params = device.createBuffer({ label: 'VoxelClipmap/ProbeUpdate', size: this.paramsStride * MAX_DISPATCHES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.probeBuffer = device.createBuffer({
            label: 'VoxelClipmap/Probes',
            size: this.layout.levels * this.layout.voxelCount() * 64,
            // (COPY_SRC: readable in tests)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.cursors = Array.from({ length: this.layout.levels }, () => 0);
    }

    /** Level `level`'s window's first lattice point, once placed. */
    public origin(level: number): Vec3 | null {
        return this.origins[level] ?? null;
    }

    public memoryBytes(): number {
        return this.probeBuffer.size;
    }

    /**
     * The declarations `CLIPMAP_PROBES_WGSL` reads, at `group` and the two bindings from `first`:
     * the grid and the probes (`bindGroupEntries` binds them).
     */
    public static bindingsWGSL(group: number, first: number): string {
        return `@group(${group}) @binding(${first}) var<uniform> kansei_clip_probe_grid : ClipProbeGrid;\n`
            + `@group(${group}) @binding(${first + 1}) var<storage, read> kansei_clip_probes : array<vec4f>;\n`;
    }

    /** The buffers for `bindingsWGSL(_, first)`, in order. */
    public bindGroupEntries(first: number): GPUBindGroupEntry[] {
        return [
            { binding: first, resource: { buffer: this.gridBuffer } },
            { binding: first + 1, resource: { buffer: this.probeBuffer } },
        ];
    }

    /**
     * Start every probe over (after a cut, or a change of the lighting the history should not
     * blend through): each level is placed anew round the camera next update.
     */
    public reset(): void {
        this.origins.fill(null);
    }

    private packGrid(): Uint8Array {
        const data = new ArrayBuffer(CLIP_PROBE_GRID_BYTES);
        const u32 = new Uint32Array(data);
        const i32 = new Int32Array(data);
        const f32 = new Float32Array(data);
        u32.set(this.layout.dims, 0);
        u32[3] = this.layout.levels;
        f32[4] = this.layout.voxelSize;
        f32[5] = Math.max(this.options.normalBias, 0);
        this.origins.forEach((o, k) => {
            if (!o) return;
            i32.set(o, 8 + 4 * k);
            u32[11 + 4 * k] = 1;
        });
        return new Uint8Array(data);
    }

    /**
     * Record the probes' update for an eye at `eye`, after the clipmap's light: each level's window
     * follows the eye (placed anew when it first is, or jumps past itself: its probes marked never
     * traced; the slab it moved into traced at once), then the next probes of each level in turn.
     */
    public encode(encoder: GPUCommandEncoder, sky: GPUBuffer, eye: Vec3): void {
        const device = this.device;
        const layout = this.layout;
        const dims = layout.dims;
        const perLevel = layout.voxelCount();
        const o = this.options;
        const dispatches: Dispatch[] = [];
        const perFrame = Math.min(Math.ceil(Math.max(o.probesPerFrame, 1) / layout.levels), perLevel);
        for (let level = 0; level < layout.levels; level++) {
            const voxelLevel = Math.min(this.voxelLevel0 + level, this.clipmap.layout.levels - 1);
            const at = { level, voxelLevel, first: 0, count: 0, lo: [0, 0, 0] as Vec3, size: [0, 0, 0] as Vec3 };
            const current = this.origin(level);
            const target = current ? layout.follow(level, current, eye, SNAP) : layout.centredOrigin(level, eye, SNAP);
            if (current && target.every((t, a) => Math.abs(t - current[a]) < dims[a])) {
                // the slabs the window moved into, axis by axis, traced fresh
                const from = [...current] as Vec3;
                for (let axis = 0; axis < 3; axis++) {
                    const d = target[axis] - from[axis];
                    if (d === 0) continue;
                    const moved = [...from] as Vec3;
                    moved[axis] = target[axis];
                    const lo = [...moved] as Vec3;
                    const size = [...dims] as Vec3;
                    if (d > 0) lo[axis] = from[axis] + dims[axis];
                    size[axis] = Math.abs(d);
                    const count = size[0] * size[1] * size[2];
                    dispatches.push({ ...at, mode: MODE_REGION, lo, size, count, workgroups: count });
                    from[axis] = target[axis];
                }
            } else {
                dispatches.push({ ...at, mode: MODE_CLEAR, first: 0, count: perLevel, workgroups: Math.ceil(perLevel / 16) });
            }
            this.origins[level] = target;
            // and the next ones in turn
            const first = this.cursors[level];
            dispatches.push({ ...at, mode: MODE_ROUND, first, count: perFrame, workgroups: perFrame });
            this.cursors[level] = (first + perFrame) % perLevel;
        }
        const grid = this.packGrid();
        if (!this.writtenGrid || grid.some((b, i) => b !== this.writtenGrid![i])) {
            device.queue.writeBuffer(this.gridBuffer, 0, grid);
            this.writtenGrid = grid;
        }
        dispatches.length = Math.min(dispatches.length, MAX_DISPATCHES);
        const rotation = randomRotation(this.frame);
        const data = new ArrayBuffer(this.paramsStride * dispatches.length);
        dispatches.forEach((p, k) => {
            const f32 = new Float32Array(data, k * this.paramsStride, CLIP_PROBE_UPDATE_BYTES / 4);
            const u32 = new Uint32Array(data, k * this.paramsStride, CLIP_PROBE_UPDATE_BYTES / 4);
            const i32 = new Int32Array(data, k * this.paramsStride, CLIP_PROBE_UPDATE_BYTES / 4);
            f32.set(rotation, 0);
            u32[16] = p.level;
            u32[17] = p.mode;
            u32[18] = p.first;
            u32[19] = p.count;
            i32.set(p.lo, 20);
            f32[23] = Math.min(Math.max(o.hysteresis, 0), 0.999);
            u32.set(p.size, 24);
            f32[27] = Math.max(o.skyScale, 0);
            u32[28] = Math.max(o.maxSteps, 1);
            u32[29] = p.voxelLevel;
            f32[30] = 0.6;
            f32[31] = Math.max(o.levelBias, 0);
            u32[32] = this.frame;
        });
        device.queue.writeBuffer(this.params, 0, data);
        this.frame = (this.frame + 1) >>> 0;

        if (!this.group || this.boundSky !== sky) {
            this.group = device.createBindGroup({
                label: 'VoxelClipmap/ProbesBG',
                layout: this.bgl,
                entries: [
                    { binding: 0, resource: { buffer: this.gridBuffer } },
                    { binding: 1, resource: { buffer: this.probeBuffer } },
                    { binding: 2, resource: { buffer: this.params, offset: 0, size: CLIP_PROBE_UPDATE_BYTES } },
                    { binding: 3, resource: { buffer: sky } },
                    ...clipmapEntries(this.clipmap),
                ],
            });
            this.boundSky = sky;
        }
        const pass = encoder.beginComputePass({ label: 'VoxelClipmap/Probes', timestampWrites: gpuPass('VoxelClipmap/Probes') });
        pass.setPipeline(this.pipeline);
        // a workgroup per probe (or per 16 slots cleared)
        dispatches.forEach((p, k) => {
            pass.setBindGroup(0, this.group!, [k * this.paramsStride]);
            pass.dispatchWorkgroups(Math.min(p.workgroups, 65535));
        });
        pass.end();
    }

    public destroy(): void {
        for (const b of [this.gridBuffer, this.params, this.probeBuffer]) b.destroy();
    }
}

/** A random rotation for frame `frame` (a uniform quaternion, Shoemake 1992). */
function randomRotation(frame: number): mat4 {
    const h = (k: number) => {
        let x = (Math.imul(frame, 0x9e3779b9) + Math.imul(k, 0x85ebca6b)) >>> 0;
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
    return mat4.fromQuat(mat4.create(), q);
}
