import { gpuPass } from '../profiling/Profiler';
import { RT_BUILD_WGSL, rtGatherWgsl } from './RtWGSL';
import type { InstanceTransform } from '../clusters/ClusterLod';

type Vec3 = [number, number, number];

/** Cells a grid holds at most (the scan's two levels of 1024). */
export const RT_MAX_CELLS = 1 << 20;
/** Bytes of a world triangle. */
export const RT_TRIANGLE_BYTES = 64;
/** Cells a macro cell spans each way. */
const MACRO = 4;
/**
 * Words of the counters before the wide triangles' list (one word a triangle): [0] triangles
 * claimed, [1] references needed, [2] wide triangles, the scan's block sums from 16.
 */
const COUNTER_WORDS = 1056;
/** Bytes between sources' parameters (WebGPU's dynamic uniform offset alignment, at most). */
const SOURCE_STRIDE = 256;
/** Bytes of the WGSL `KanseiRtGrid` (rt_types.wgsl; Rust `RtGridGpu`). */
export const RT_GRID_UNIFORM_BYTES = 80;
/** Bytes of the WGSL `RtSource` (rt_gather.wgsl; Rust `RtSourceGpu`). */
const RT_SOURCE_BYTES = 112;
const NO_WORD = 0xffffffff;

/** How an instance record places a mesh, as cluster LOD reads it (`clusters/ClusterLod`). */
export type { InstanceTransform };

/** A WGSL float literal (always with a point or an exponent). */
function wgslFloat(x: number): string {
    const s = String(x);
    return /[.e]/.test(s) ? s : `${s}.0`;
}

/**
 * Where a source's records place its mesh, before the source's `world` matrix: nowhere else (the
 * mesh once per record, where `world` puts it), as an instance record says (`InstanceTransform`),
 * or WGSL defining `fn kansei_rt_place(record: u32, p: vec3f) -> vec3f`, where record `record`
 * puts mesh point `p` (the renderable's space), for a material whose vertex stage does more than
 * an `InstanceTransform` says (read the record with `kansei_rt_record_f32`,
 * `kansei_rt_record_vec3` or `kansei_rt_record_vec4(record, word)`). Rust: `rt::RtPlacement`.
 */
export class RtPlacement {
    private constructor(private readonly code: string) { }

    static readonly NONE = new RtPlacement('fn kansei_rt_place(record: u32, p: vec3f) -> vec3f { return p; }\n');

    /** As an instance record says. */
    static instance(t: InstanceTransform): RtPlacement {
        if (t.kind === 'matrix') {
            const w = t.offset / 4;
            return new RtPlacement(`fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {\n    let m = mat4x4f(kansei_rt_record_vec4(record, ${w}u), kansei_rt_record_vec4(record, ${w + 4}u), kansei_rt_record_vec4(record, ${w + 8}u), kansei_rt_record_vec4(record, ${w + 12}u));\n    return (m * vec4f(p, 1.0)).xyz;\n}\n`);
        }
        // as cluster_cull.wgsl's placement: position + yaw * rotation * (scale * p)
        let body = '    var q = p;\n';
        if (t.scale != null) body += `    q = q * kansei_rt_record_f32(record, ${t.scale / 4}u);\n`;
        if (t.rotation != null) {
            body += `    let r = kansei_rt_record_vec4(record, ${t.rotation / 4}u);\n    q = q + 2.0 * cross(r.xyz, cross(r.xyz, q) + r.w * q);\n`;
        }
        if (t.yaw != null) {
            body += `    let a = kansei_rt_record_f32(record, ${t.yaw / 4}u) * ${wgslFloat(t.yawScale ?? 1)};\n    q = vec3f(cos(a) * q.x + sin(a) * q.z, q.y, -sin(a) * q.x + cos(a) * q.z);\n`;
        }
        return new RtPlacement(`fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {\n${body}    return q + kansei_rt_record_vec3(record, ${t.position / 4}u);\n}\n`);
    }

    /** WGSL defining `kansei_rt_place`. */
    static wgsl(code: string): RtPlacement {
        return new RtPlacement(code);
    }

    /** Its `kansei_rt_place`. */
    wgsl(): string {
        return this.code;
    }
}

/**
 * How a triangle's surface reads to a ray: its albedo (for lighting a hit), and whether it is
 * alpha tested (`alphaLayer`: the caller's `kansei_rt_covered(layer, uv)` decides where it is
 * there). Rust: `rt::RtSurface`.
 */
export interface RtSurface {
    albedo: Vec3;
    alphaLayer?: number | null;
    /**
     * Its triangles carry their vertex normals (in place of their uvs), which
     * `kansei_rt_shading_normal` interpolates at a hit: for the rays a curved surface bends or
     * mirrors. Its uvs are then gone: only for surfaces no `kansei_rt_covered` samples by uv.
     * Rust: `RtSurface::with_smooth_normals`.
     */
    smoothNormals?: boolean;
}

export function rtSurfaceWord(s: RtSurface): number {
    return ((s.alphaLayer == null ? 0 : (1 | ((s.alphaLayer & 255) << 8))) | (s.smoothNormals ? 4 : 0)) >>> 0;
}

export function rtAlbedoWord(s: RtSurface): number {
    const byte = (x: number) => Math.round(Math.min(Math.max(x, 0), 1) * 255);
    return (byte(s.albedo[0]) | (byte(s.albedo[1]) << 8) | (byte(s.albedo[2]) << 16) | (255 << 24)) >>> 0;
}

/** What `RtGrid` sets up. Rust: `rt::RtGridOptions`. */
export interface RtGridOptions {
    /** Cells each way: multiples of 4, at most `RT_MAX_CELLS` in all. Default [128, 64, 128]. */
    dims?: Vec3;
    /** A cell's size, metres. Default 0.5. */
    cell?: number;
    /** Where the box sits round the eye it follows (`follow`): the share of its height below the eye. Default 0.25. */
    below?: number;
    /** The box follows the eye in steps of this many cells (a multiple of 4 keeps the macro cells on the same world blocks). Default 4. */
    snapCells?: number;
    /** A box that stays put from this corner (a room) instead of following the eye. */
    fixedOrigin?: Vec3 | null;
    /**
     * World units every cell is widened by when triangles are listed, so one lying on a cell's
     * face, or a rounding error away from it, is listed on both sides (0, the default: a 1024th of a cell).
     */
    epsilon?: number;
    /**
     * A triangle whose footprint across its plane's dominant axis spans more columns of cells than
     * this goes to the big list, which every ray tests before walking the cells
     * (miaumiau.cat/?p=1457's "big triangles"). Default 1024.
     */
    bigTriangleCells?: number;
    /** Triangles the big list holds; past that they are scattered into the cells. Default 64. */
    bigTriangleCapacity?: number;
    /** Leave empty 4³ blocks of cells in one step. Default true. */
    macroSkip?: boolean;
    /** Triangles the buffers start with room for; it grows to what a build needed (read back a few frames later). Default 65536. */
    triangleCapacity?: number;
    /** References (a triangle listed in a cell) the buffers start with room for; grows likewise. Default 1 << 20. */
    referenceCapacity?: number;
}

/** `RtGridOptions` with every default filled in. */
export type ResolvedRtGridOptions = Required<RtGridOptions>;

export function resolveRtGridOptions(o: RtGridOptions = {}): ResolvedRtGridOptions {
    return {
        dims: o.dims ?? [128, 64, 128],
        cell: o.cell ?? 0.5,
        below: o.below ?? 0.25,
        snapCells: o.snapCells ?? 4,
        fixedOrigin: o.fixedOrigin ?? null,
        epsilon: o.epsilon ?? 0,
        bigTriangleCells: o.bigTriangleCells ?? 1024,
        bigTriangleCapacity: o.bigTriangleCapacity ?? 64,
        macroSkip: o.macroSkip ?? true,
        triangleCapacity: o.triangleCapacity ?? 1 << 16,
        referenceCapacity: o.referenceCapacity ?? 1 << 20,
    };
}

/** The epsilon cells are widened by. */
export function effectiveEpsilon(o: ResolvedRtGridOptions): number {
    return o.epsilon > 0 ? o.epsilon : o.cell / 1024;
}

/** What the last build read back needed, and the room the buffers have. Rust: `rt::RtGridStats`. */
export interface RtGridStats {
    /** Triangles gathered (meeting the box), whether they fitted or not. */
    triangles: number;
    /** References the cells needed. */
    references: number;
    /** Triangles too wide for the cells (those up to `bigTriangleCapacity` in the big list). */
    bigTriangles: number;
    triangleCapacity: number;
    referenceCapacity: number;
    /** Builds recorded. */
    builds: number;
}

/**
 * The buffers a pass traces an `RtGrid` through, which follow the grid when its buffers are made
 * anew (`RtGrid.handle`): an effect holds one and binds what it holds each frame.
 * Rust: `rt::RtGridHandle`.
 */
export class RtGridHandle {
    /** The buffers of `rtGridBindingsWgsl`, in order. */
    buffers: [GPUBuffer, GPUBuffer, GPUBuffer];
    /** Changed when they are made anew. */
    generation = 0;

    constructor(buffers: [GPUBuffer, GPUBuffer, GPUBuffer]) {
        this.buffers = buffers;
    }
}

/** A source of triangles for `RtGrid.gather`: a mesh (`RtMesh.createBuffer`) placed once per record. Rust: `rt::RtSource`. */
export interface RtSource {
    /** The mesh's words (`RtMesh.gpuWords`), in a STORAGE buffer. */
    mesh: GPUBuffer;
    /** Its triangles. */
    triangles: number;
    /** Instance records `stride` bytes each (a STORAGE buffer), placed by `placement`; null: the mesh once. */
    records: { buffer: GPUBuffer; stride: number } | null;
    /** The first record and how many. */
    firstRecord: number;
    recordCount: number;
    /** Where the placed mesh goes (a renderable's world matrix, column-major). */
    world: ArrayLike<number>;
    placement: RtPlacement;
    surface: RtSurface;
    /** Names the source in its triangles (`kansei_rt_source`, 12 bits). */
    id: number;
}

/**
 * How many records a gathered source has: a fixed count, or word `word` of `args` says, at most
 * `atMost` (a culled view's indirect draw).
 */
export type GatherCount = { fixed: number } | { args: GPUBuffer; word: number; atMost: number };

/**
 * A source as the gather runs it: `RtSource`'s, or one the renderer feeds from its culled views
 * and cluster cuts (`SceneRtGrid`). Rust: `GatherSource`.
 */
export interface GatherSource {
    /** An `RtMesh`'s words, or with `draws` a cluster mesh's (`ClusterMesh.gpuWords`). */
    mesh: GPUBuffer;
    triangles: number;
    /**
     * A cluster LOD cut's draw list of (record, cluster), gathered by `gather_clusters`: its
     * records are absolute (`firstRecord` is ignored) and `count` says how many entries.
     */
    draws?: GPUBuffer | null;
    records: { buffer: GPUBuffer; stride: number } | null;
    firstRecord: number;
    count: GatherCount;
    world: ArrayLike<number>;
    /** Its `kansei_rt_place` (`RtPlacement.wgsl`). */
    placement: string;
    surface: RtSurface;
    id: number;
}

/** A number per GPU buffer, for the bind group cache's keys. */
const bufferIds = new WeakMap<GPUBuffer, number>();
let nextBufferId = 1;
function bufferId(b: GPUBuffer | null | undefined): number {
    if (!b) return 0;
    let id = bufferIds.get(b);
    if (id === undefined) {
        id = nextBufferId++;
        bufferIds.set(b, id);
    }
    return id;
}

/** A capacity for a need: a quarter more, rounded up to a 64th of its power of two. */
function grow(need: number): number {
    const want = Math.max(Math.floor(need * 5 / 4), 64);
    let pow = 1;
    while (pow < want) pow *= 2;
    const step = Math.max(pow / 64, 1);
    return Math.min(Math.ceil(want / step) * step, 0xffffffff);
}

/** The readback's state: nothing in flight, mapping, or mapped (Rust `MAPPING`/`MAPPED`). */
const enum ReadbackState { Free, Mapping, Mapped }

/**
 * A uniform grid of world triangles in a box round the camera (or a fixed one), rebuilt on the
 * GPU, for rays. Each rebuild, between `begin` and `finish`, one `gather` appends the sources'
 * triangles (a thread a triangle, dropped when its box misses the grid's; a workgroup's
 * survivors take their slots with one atomic); `finish` lists them in the cells they overlap (the
 * separating-axis test, cells widened by `epsilon`): a thread a triangle, a workgroup a triangle
 * spanning more than 32 columns of cells, and room-sized ones in a short list every ray tests
 * (`bigTriangleCells`); a prefix sum gives each cell an exact list. After the frame's submit,
 * `readBack` reads what the build needed, and the buffers grow to it at a later `begin`.
 *
 * Trace it with `RT_GRID_WGSL` (`rtGridBindingsWgsl`, `bindGroupEntries`). See
 * `docs/plans/2026-10-05-rt-grid-design.md`. Rust: `rt::RtGrid`.
 */
export class RtGrid {
    public readonly options: ResolvedRtGridOptions;
    private _origin: Vec3;
    private placed = false;
    private readonly uniform: GPUBuffer;
    private written: Uint32Array | null = null;
    private triangles!: GPUBuffer;
    private cells!: GPUBuffer;
    private counters!: GPUBuffer;
    private readonly dispatch: GPUBuffer;
    private sources: GPUBuffer;
    private readonly empty: GPUBuffer;
    private triangleCapacity = 0;
    private referenceCapacity = 0;
    private readonly buildBGL: GPUBindGroupLayout;
    private readonly prepareBGL: GPUBindGroupLayout;
    private readonly gatherGridBGL: GPUBindGroupLayout;
    private readonly gatherSourceBGL: GPUBindGroupLayout;
    private readonly gatherPrepareBGL: GPUBindGroupLayout;
    private readonly gatherPrepare: GPUComputePipeline;
    private readonly gatherLayout: GPUPipelineLayout;
    /** the sources' indirect dispatches, and the bind group writing them */
    private gatherDispatch: GPUBuffer;
    private gatherDispatchGroup: GPUBindGroup | null = null;
    private readonly pipelines: Record<'prepare' | 'prepare_wide' | 'count' | 'count_wide' | 'fill' | 'fill_wide' | 'scan_blocks' | 'scan_sums' | 'scan_add', GPUComputePipeline>;
    private readonly gatherPipelines = new Map<string, GPUComputePipeline>();
    private bindGroups: { build: GPUBindGroup; prepare: GPUBindGroup; gather: GPUBindGroup } | null = null;
    private readonly sourceGroups = new Map<string, GPUBindGroup>();
    private readonly staging: GPUBuffer;
    private readbackState = ReadbackState.Free;
    private readbackWords: Uint32Array | null = null;
    private readonly _stats: RtGridStats = { triangles: 0, references: 0, bigTriangles: 0, triangleCapacity: 0, referenceCapacity: 0, builds: 0 };
    private _generation = 0;
    private readonly _handle: RtGridHandle;
    /** between `begin` and `finish`: whether this build gathered yet */
    private gathered: boolean | null = null;

    constructor(private readonly device: GPUDevice, options: RtGridOptions = {}) {
        const o = resolveRtGridOptions(options);
        if (!o.dims.every((d) => d > 0 && d % MACRO === 0)) throw new Error(`RtGrid: dims must be multiples of ${MACRO}`);
        if (o.dims[0] * o.dims[1] * o.dims[2] > RT_MAX_CELLS) throw new Error(`RtGrid: at most ${RT_MAX_CELLS} cells`);
        this.options = o;
        this._origin = o.fixedOrigin ? [...o.fixedOrigin] : [0, 0, 0];

        const visibility = GPUShaderStage.COMPUTE;
        const entry = (binding: number, buffer: GPUBufferBindingLayout): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer });
        const uniform: GPUBufferBindingLayout = { type: 'uniform' };
        const rw: GPUBufferBindingLayout = { type: 'storage' };
        const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
        const layout = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });
        this.buildBGL = layout('RtGrid/Build', [entry(0, uniform), entry(1, rw), entry(2, rw), entry(3, rw)]);
        this.prepareBGL = layout('RtGrid/Prepare', [entry(0, uniform), entry(3, rw), entry(4, rw)]);
        this.gatherGridBGL = layout('RtGrid/GatherGrid', [entry(0, uniform), entry(1, rw), entry(2, rw)]);
        this.gatherSourceBGL = layout('RtGrid/GatherSource', [entry(0, { type: 'uniform', hasDynamicOffset: true }), entry(1, ro), entry(2, ro), entry(3, ro), entry(4, ro)]);
        this.gatherPrepareBGL = layout('RtGrid/GatherPrepare', [entry(5, rw)]);

        const module = device.createShaderModule({ label: 'RtGrid/Build', code: RT_BUILD_WGSL });
        const pipeline = (entryPoint: string, bgl: GPUBindGroupLayout) => device.createComputePipeline({
            label: `RtGrid/${entryPoint}`,
            layout: device.createPipelineLayout({ label: 'RtGrid/Build', bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint },
        });
        this.pipelines = {
            prepare: pipeline('prepare', this.prepareBGL),
            prepare_wide: pipeline('prepare_wide', this.prepareBGL),
            count: pipeline('count', this.buildBGL),
            count_wide: pipeline('count_wide', this.buildBGL),
            fill: pipeline('fill', this.buildBGL),
            fill_wide: pipeline('fill_wide', this.buildBGL),
            scan_blocks: pipeline('scan_blocks', this.buildBGL),
            scan_sums: pipeline('scan_sums', this.buildBGL),
            scan_add: pipeline('scan_add', this.buildBGL),
        };
        this.gatherLayout = device.createPipelineLayout({ label: 'RtGrid/Gather', bindGroupLayouts: [this.gatherGridBGL, this.gatherSourceBGL] });
        this.gatherPrepare = device.createComputePipeline({
            label: 'RtGrid/GatherPrepare',
            layout: device.createPipelineLayout({ label: 'RtGrid/GatherPrepare', bindGroupLayouts: [this.gatherPrepareBGL, this.gatherSourceBGL] }),
            compute: { module: device.createShaderModule({ label: 'RtGrid/GatherPrepare', code: rtGatherWgsl(RtPlacement.NONE.wgsl()) }), entryPoint: 'prepare' },
        });

        const storage = (label: string, size: number, usage: GPUBufferUsageFlags = 0) => device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | usage });
        this.uniform = device.createBuffer({ label: 'RtGrid/Uniform', size: RT_GRID_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.dispatch = storage('RtGrid/Dispatch', 32, GPUBufferUsage.INDIRECT);
        this.sources = device.createBuffer({ label: 'RtGrid/Sources', size: SOURCE_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.empty = storage('RtGrid/Empty', 16);
        this.gatherDispatch = storage('RtGrid/GatherDispatch', 16, GPUBufferUsage.INDIRECT);
        this.staging = device.createBuffer({ label: 'RtGrid/Readback', size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this._handle = new RtGridHandle([this.uniform, this.empty, this.empty]);
        this.resize(Math.max(o.triangleCapacity, 64), Math.max(o.referenceCapacity, 1024));
    }

    /** The box's corner. */
    get origin(): Vec3 {
        return [...this._origin];
    }

    /** The box's size, metres. */
    get extent(): Vec3 {
        const o = this.options;
        return [o.dims[0] * o.cell, o.dims[1] * o.cell, o.dims[2] * o.cell];
    }

    /** The box: its least and greatest corners. */
    bounds(): [Vec3, Vec3] {
        const e = this.extent;
        const o = this._origin;
        return [[...o], [o[0] + e[0], o[1] + e[1], o[2] + e[2]]];
    }

    /** What the last readback said, and the room the buffers have. */
    get stats(): RtGridStats {
        return { ...this._stats, triangleCapacity: this.triangleCapacity, referenceCapacity: this.referenceCapacity };
    }

    /** Changes whenever the buffers `bindGroupEntries` binds are made anew: bind groups made with the old ones are stale. */
    get generation(): number {
        return this._generation;
    }

    /** A handle to the buffers passes trace the grid through, which follows the grid's. */
    get handle(): RtGridHandle {
        return this._handle;
    }

    /** Bytes on the GPU: the triangles, the cell words, the counters (with the wide triangles' list) and the sources' parameters. */
    memoryBytes(): number {
        return this.triangles.size + this.cells.size + this.counters.size + this.sources.size;
    }

    /**
     * Place the box round `eye` (in steps of `snapCells` cells, `below` of its height under the
     * eye), unless it is fixed. True when it moved: rebuild it.
     */
    follow(eye: ArrayLike<number>): boolean {
        const o = this.options;
        let origin: Vec3;
        if (o.fixedOrigin) {
            origin = [...o.fixedOrigin];
        } else {
            const snap = o.cell * Math.max(o.snapCells, 1);
            const size = this.extent;
            const corner = [eye[0] - size[0] * 0.5, eye[1] - size[1] * o.below, eye[2] - size[2] * 0.5];
            origin = [Math.floor(corner[0] / snap) * snap, Math.floor(corner[1] / snap) * snap, Math.floor(corner[2] / snap) * snap];
        }
        const moved = !this.placed || origin.some((x, k) => x !== this._origin[k]);
        this._origin = origin;
        this.placed = true;
        return moved;
    }

    private cellCount(): number {
        const d = this.options.dims;
        return d[0] * d[1] * d[2];
    }

    private macroDims(): Vec3 {
        const d = this.options.dims;
        return [Math.ceil(d[0] / MACRO), Math.ceil(d[1] / MACRO), Math.ceil(d[2] / MACRO)];
    }

    /** Where the macro cells, the big list and the references start in the cell words. */
    cellsLayoutWords(): [number, number, number] {
        const m = this.macroDims();
        const macroBase = this.cellCount();
        const bigBase = macroBase + m[0] * m[1] * m[2];
        const refsBase = bigBase + 1 + this.options.bigTriangleCapacity;
        return [macroBase, bigBase, refsBase];
    }

    /** Make the buffers hold `triangles` and `references` (within what a binding holds); their contents are lost. */
    private resize(triangles: number, references: number): void {
        const limits = this.device.limits;
        const limit = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
        const refsBase = this.cellsLayoutWords()[2];
        triangles = Math.min(triangles, Math.floor(limit / RT_TRIANGLE_BYTES));
        references = Math.min(references, Math.floor(limit / 4) - refsBase);
        if (triangles !== this.triangleCapacity) {
            this.triangleCapacity = triangles;
            this.triangles?.destroy();
            this.counters?.destroy();
            this.triangles = this.device.createBuffer({ label: 'RtGrid/Triangles', size: triangles * RT_TRIANGLE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            // (every triangle may be wide)
            this.counters = this.device.createBuffer({
                label: 'RtGrid/Counters',
                size: (COUNTER_WORDS + triangles) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
        }
        if (references !== this.referenceCapacity) {
            this.referenceCapacity = references;
            this.cells?.destroy();
            this.cells = this.device.createBuffer({
                label: 'RtGrid/Cells',
                size: (refsBase + references) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
        }
        this.bindGroups = null;
        this._generation++;
        this._handle.buffers = [this.uniform, this.triangles, this.cells];
        this._handle.generation = this._generation;
    }

    /**
     * Room for `triangles` this build (call between `begin` and `gather`, when the sources know how
     * many they hold at most): no frame then loses triangles to a readback still in flight.
     */
    reserveTriangles(triangles: number): void {
        if (triangles > this.triangleCapacity) {
            this.resize(grow(triangles), this.referenceCapacity);
            this.ensureBindGroups();
            if (this.gathered !== null) this.writeUniform();
        }
    }

    private ensureBindGroups(): { build: GPUBindGroup; prepare: GPUBindGroup; gather: GPUBindGroup } {
        if (this.bindGroups) return this.bindGroups;
        const group = (label: string, layout: GPUBindGroupLayout, buffers: [number, GPUBuffer][]) =>
            this.device.createBindGroup({ label, layout, entries: buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
        this.bindGroups = {
            build: group('RtGrid/Build', this.buildBGL, [[0, this.uniform], [1, this.triangles], [2, this.cells], [3, this.counters]]),
            prepare: group('RtGrid/Prepare', this.prepareBGL, [[0, this.uniform], [3, this.counters], [4, this.dispatch]]),
            gather: group('RtGrid/GatherGrid', this.gatherGridBGL, [[0, this.uniform], [1, this.triangles], [2, this.counters]]),
        };
        return this.bindGroups;
    }

    /**
     * Start a rebuild: take a finished readback (growing the buffers to what it needed), write the
     * box, and clear the triangles and the cells.
     */
    begin(encoder: GPUCommandEncoder): void {
        if (this.gathered !== null) throw new Error('RtGrid.begin twice without finish');
        this.collectReadback();
        const needTriangles = this._stats.triangles;
        // (triangles that didn't fit listed nothing: their references in proportion)
        const fitted = Math.max(Math.min(needTriangles, this.triangleCapacity), 1);
        const needRefs = Math.min(Math.floor(this._stats.references * Math.max(needTriangles, 1) / fitted), 0xffffffff);
        if (needTriangles > this.triangleCapacity || needRefs > this.referenceCapacity) {
            this.resize(
                needTriangles > this.triangleCapacity ? grow(needTriangles) : this.triangleCapacity,
                needRefs > this.referenceCapacity ? grow(needRefs) : this.referenceCapacity,
            );
        }
        this.ensureBindGroups();
        this.writeUniform();
        const bigBase = this.cellsLayoutWords()[1];
        encoder.clearBuffer(this.counters, 0, 16);
        // the cells' counts, the macro cells and the big list's count
        encoder.clearBuffer(this.cells, 0, (bigBase + 1) * 4);
        this.gathered = false;
    }

    /** Write the box and the buffers' layout, if they changed. */
    private writeUniform(): void {
        const [macroBase, bigBase, refsBase] = this.cellsLayoutWords();
        const o = this.options;
        const words = new Uint32Array(RT_GRID_UNIFORM_BYTES / 4);
        const f = new Float32Array(words.buffer);
        f.set(this._origin, 0);
        f[3] = o.cell;
        words.set(o.dims, 4);
        words[7] = o.macroSkip ? 1 : 0;
        words.set(this.macroDims(), 8);
        f[11] = effectiveEpsilon(o);
        words.set([this.cellCount(), macroBase, bigBase, refsBase, this.referenceCapacity, this.triangleCapacity, o.bigTriangleCapacity, o.bigTriangleCells], 12);
        if (this.written && this.written.every((w, k) => w === words[k])) return;
        this.device.queue.writeBuffer(this.uniform, 0, words);
        this.written = words;
    }

    /** The gather pipeline for a placement. */
    private gatherPipeline(placement: string, clusters: boolean): GPUComputePipeline {
        const key = `${clusters ? 'clusters' : 'mesh'}\n${placement}`;
        let p = this.gatherPipelines.get(key);
        if (!p) {
            p = this.device.createComputePipeline({
                label: 'RtGrid/Gather',
                layout: this.gatherLayout,
                compute: { module: this.device.createShaderModule({ label: 'RtGrid/Gather', code: rtGatherWgsl(placement) }), entryPoint: clusters ? 'gather_clusters' : 'gather' },
            });
            this.gatherPipelines.set(key, p);
        }
        return p;
    }

    /**
     * Append the triangles of `sources` that meet the box (once a build, between `begin` and
     * `finish`: their parameters go up in one write, which lands before the frame's work).
     */
    gather(encoder: GPUCommandEncoder, sources: RtSource[]): void {
        this.gatherSources(encoder, sources.map((s) => ({
            mesh: s.mesh,
            triangles: s.triangles,
            records: s.records,
            firstRecord: s.firstRecord,
            count: { fixed: s.recordCount },
            world: s.world,
            placement: s.placement.wgsl(),
            surface: s.surface,
            id: s.id,
        })));
    }

    /**
     * `gather` for the engine's sources: each source's dispatch sized on the GPU from its count
     * (`prepare`), then the gathers, dispatched indirectly.
     */
    gatherSources(encoder: GPUCommandEncoder, sources: GatherSource[]): void {
        if (this.gathered !== false) throw new Error('RtGrid.gather once a build, between begin and finish');
        this.gathered = true;
        if (sources.length === 0) return;
        const device = this.device;
        const needed = sources.length * SOURCE_STRIDE;
        if (this.sources.size < needed) {
            let size = SOURCE_STRIDE;
            while (size < needed) size *= 2;
            this.sources.destroy();
            this.sources = device.createBuffer({ label: 'RtGrid/Sources', size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this.sourceGroups.clear();
        }
        if (this.gatherDispatch.size < sources.length * 16) {
            let size = 16;
            while (size < sources.length * 16) size *= 2;
            this.gatherDispatch.destroy();
            this.gatherDispatch = device.createBuffer({ label: 'RtGrid/GatherDispatch', size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
            this.gatherDispatchGroup = null;
        }
        const bytes = new ArrayBuffer(needed);
        sources.forEach((s, k) => {
            const f = new Float32Array(bytes, k * SOURCE_STRIDE, RT_SOURCE_BYTES / 4);
            const u = new Uint32Array(bytes, k * SOURCE_STRIDE, RT_SOURCE_BYTES / 4);
            f.set(s.world, 0);
            const [records, countWord] = 'fixed' in s.count ? [s.count.fixed, NO_WORD] : [s.count.atMost, s.count.word];
            u.set([s.triangles, s.records ? s.records.stride / 4 : 0, s.firstRecord, records, countWord,
                rtSurfaceWord(s.surface), rtAlbedoWord(s.surface), ((s.id & 0xfff) << 20) >>> 0, k, s.draws ? 1 : 0, 0, 0], 16);
        });
        device.queue.writeBuffer(this.sources, 0, bytes);

        // a source's bind group: its mesh, records and count, cached by those buffers
        if (this.sourceGroups.size > 512) this.sourceGroups.clear();
        const groups = sources.map((s) => {
            const args = 'fixed' in s.count ? null : s.count.args;
            const key = `${bufferId(s.mesh)},${bufferId(s.records?.buffer)},${bufferId(args)},${bufferId(s.draws)}`;
            let group = this.sourceGroups.get(key);
            if (!group) {
                group = device.createBindGroup({
                    label: 'RtGrid/Source',
                    layout: this.gatherSourceBGL,
                    entries: [
                        { binding: 0, resource: { buffer: this.sources, offset: 0, size: RT_SOURCE_BYTES } },
                        { binding: 1, resource: { buffer: s.mesh } },
                        { binding: 2, resource: { buffer: s.records?.buffer ?? this.empty } },
                        { binding: 3, resource: { buffer: args ?? this.empty } },
                        { binding: 4, resource: { buffer: s.draws ?? this.empty } },
                    ],
                });
                this.sourceGroups.set(key, group);
            }
            return group;
        });
        const pipelines = sources.map((s) => this.gatherPipeline(s.placement, !!s.draws));
        this.gatherDispatchGroup ??= device.createBindGroup({
            label: 'RtGrid/GatherPrepare',
            layout: this.gatherPrepareBGL,
            entries: [{ binding: 5, resource: { buffer: this.gatherDispatch } }],
        });
        const { gather } = this.ensureBindGroups();
        const pass = encoder.beginComputePass({ label: 'Rt/Gather', timestampWrites: gpuPass('Rt/Gather') });
        pass.setPipeline(this.gatherPrepare);
        pass.setBindGroup(0, this.gatherDispatchGroup);
        groups.forEach((group, k) => {
            pass.setBindGroup(1, group, [k * SOURCE_STRIDE]);
            pass.dispatchWorkgroups(1);
        });
        pass.setBindGroup(0, gather);
        groups.forEach((group, k) => {
            pass.setPipeline(pipelines[k]);
            pass.setBindGroup(1, group, [k * SOURCE_STRIDE]);
            pass.dispatchWorkgroupsIndirect(this.gatherDispatch, k * 16);
        });
        pass.end();
    }

    /** Build the cells from the gathered triangles: count (and the big list), scan, fill. */
    finish(encoder: GPUCommandEncoder): void {
        if (this.gathered === null) throw new Error('RtGrid.finish without begin');
        this.gathered = null;
        this._stats.builds++;
        const { build, prepare } = this.ensureBindGroups();
        const p = this.pipelines;
        {
            const pass = encoder.beginComputePass({ label: 'Rt/GridCount', timestampWrites: gpuPass('Rt/GridCount') });
            pass.setPipeline(p.prepare);
            pass.setBindGroup(0, prepare);
            pass.dispatchWorkgroups(1);
            pass.setPipeline(p.count);
            pass.setBindGroup(0, build);
            pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
            pass.setPipeline(p.prepare_wide);
            pass.setBindGroup(0, prepare);
            pass.dispatchWorkgroups(1);
            pass.setPipeline(p.count_wide);
            pass.setBindGroup(0, build);
            pass.dispatchWorkgroupsIndirect(this.dispatch, 16);
            pass.end();
        }
        {
            const cells = this.cellCount();
            const pass = encoder.beginComputePass({ label: 'Rt/GridScan', timestampWrites: gpuPass('Rt/GridScan') });
            pass.setBindGroup(0, build);
            pass.setPipeline(p.scan_blocks);
            pass.dispatchWorkgroups(Math.ceil(cells / 1024));
            pass.setPipeline(p.scan_sums);
            pass.dispatchWorkgroups(1);
            pass.setPipeline(p.scan_add);
            pass.dispatchWorkgroups(Math.ceil(cells / 256));
            pass.end();
        }
        {
            const pass = encoder.beginComputePass({ label: 'Rt/GridFill', timestampWrites: gpuPass('Rt/GridFill') });
            pass.setPipeline(p.fill);
            pass.setBindGroup(0, build);
            pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
            pass.setPipeline(p.fill_wide);
            pass.dispatchWorkgroupsIndirect(this.dispatch, 16);
            pass.end();
        }
    }

    /**
     * After the submit of a build: read back what it needed (the gathered triangles, the
     * references, the big triangles), unless a readback is still in flight. `begin` takes it.
     */
    readBack(): void {
        if (this.readbackState !== ReadbackState.Free) return;
        const bigBase = this.cellsLayoutWords()[1];
        const encoder = this.device.createCommandEncoder({ label: 'RtGrid/Readback' });
        encoder.copyBufferToBuffer(this.counters, 0, this.staging, 0, 8);
        encoder.copyBufferToBuffer(this.cells, bigBase * 4, this.staging, 8, 4);
        this.device.queue.submit([encoder.finish()]);
        this.readbackState = ReadbackState.Mapping;
        this.staging.mapAsync(GPUMapMode.READ).then(
            () => {
                this.readbackWords = new Uint32Array(this.staging.getMappedRange().slice(0));
                this.staging.unmap();
                this.readbackState = ReadbackState.Mapped;
            },
            // a lost device or a destroyed buffer: its result never comes
            () => { this.readbackState = ReadbackState.Free; },
        );
    }

    /** Take a readback that finished into `stats` (`begin` does too). */
    pollReadback(): void {
        this.collectReadback();
    }

    private collectReadback(): void {
        if (this.readbackState !== ReadbackState.Mapped) return;
        const w = this.readbackWords!;
        this._stats.triangles = w[0];
        this._stats.references = w[1];
        this._stats.bigTriangles = w[2];
        this.readbackWords = null;
        this.readbackState = ReadbackState.Free;
    }

    /** The layout entries for `rtGridBindingsWgsl(_, first)`. */
    static layoutEntries(first: number, visibility: GPUShaderStageFlags): GPUBindGroupLayoutEntry[] {
        return [
            { binding: first, visibility, buffer: { type: 'uniform' } },
            { binding: first + 1, visibility, buffer: { type: 'read-only-storage' } },
            { binding: first + 2, visibility, buffer: { type: 'read-only-storage' } },
        ];
    }

    /** The buffers for `rtGridBindingsWgsl(_, first)`, in order. */
    bindGroupEntries(first: number): GPUBindGroupEntry[] {
        return [
            { binding: first, resource: { buffer: this.uniform } },
            { binding: first + 1, resource: { buffer: this.triangles } },
            { binding: first + 2, resource: { buffer: this.cells } },
        ];
    }

    /** The world triangles (64 bytes each, rt_types.wgsl), for tests and debugging. */
    get trianglesBuffer(): GPUBuffer {
        return this.triangles;
    }

    /** The cell words (rt_types.wgsl), for tests and debugging. */
    get cellsBuffer(): GPUBuffer {
        return this.cells;
    }

    destroy(): void {
        for (const b of [this.uniform, this.triangles, this.cells, this.counters, this.dispatch, this.sources, this.empty, this.gatherDispatch]) b.destroy();
        // (a mapping in flight fails, freeing nothing else)
        this.staging.destroy();
        this.sourceGroups.clear();
        this.bindGroups = null;
    }
}
