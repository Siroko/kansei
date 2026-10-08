import clusterCullWgsl from '../../rust/kansei-core/src/shaders/cluster_cull.wgsl?raw';
import { frustumPlanes } from '../culling/Frustum';
import { gpuPass } from '../profiling/Profiler';
import { ClusterMesh } from './ClusterMesh';
import { CLUSTER_MESH_WGSL, InstanceLayout } from './vertexStage';
import type { InstancedGeometry } from '../geometries/InstancedGeometry';
import type { DepthBias } from '../materials/Material';
import type { Renderable } from '../objects/Renderable';

/**
 * A `ClusterMesh` on the GPU, and each view's per-frame cut of it (Rust `clusters/gpu.rs`). The
 * cull shader is the Rust engine's (`shaders/cluster_cull.wgsl`), concatenated with
 * `CLUSTER_MESH_WGSL` as Rust does.
 */

/** The cull shader, shared with the Rust engine (`shaders/cluster_cull.wgsl`). */
export const CLUSTER_CULL_WGSL: string = clusterCullWgsl;

const NO_WORD = 0xffffffff;
const KIND_NONE = 0;
const KIND_PLACEMENT = 1;
const KIND_MATRIX = 2;
const FLAG_CONE = 1;
/** Entries of a draw list by default, at most (32 MB). */
export const DEFAULT_MAX_DRAWN = 1 << 22;
/** Entries a draw list starts with, until the cull says what its cut needs (128 KB). */
export const INITIAL_DRAWN = 1 << 14;
/** Entries a draw list keeps at least (8 KB). */
export const MIN_DRAWN = 1 << 10;
/** Readbacks in a row a cut must need at most a quarter of its list before the list shrinks. */
export const SHRINK_AFTER = 64;
/** Triangles an index buffer starts with, until the cull says what its cut needs (3 MB). */
export const INITIAL_TRIANGLES = 1 << 18;
/** Triangles an index buffer keeps room for at least (48 KB). */
export const MIN_TRIANGLES = 1 << 12;
/**
 * Bytes of a cluster draw: `DrawIndexedIndirect`'s five words (the index count, one instance,
 * zeros), then the visible instances, the clusters claimed and listed (drawn), the triangles
 * claimed and listed, and two pads.
 */
export const CLUSTER_DRAW_ARGS_BYTES = 48;
/** Word of the draw: the clusters claimed, drawn or not. */
export const CLAIMED_WORD = 6;
/** Word of the draw: the clusters listed (drawn). */
export const LISTED_WORD = 7;
/** Word of the draw: the triangles claimed, drawn or not. */
export const TRIANGLES_CLAIMED_WORD = 8;
/** Word of the draw: the triangles listed (drawn). */
export const TRIANGLES_WORD = 9;
/** Bytes of `ClusterCull` in cluster_cull.wgsl. */
export const CLUSTER_CULL_BYTES = 128;
/** Bytes of `ClusterView` in cluster_cull.wgsl. */
export const CLUSTER_VIEW_BYTES = 128;
/** Views a frame's cull holds at most (the camera, the shadow views: far fewer). */
export const MAX_CLUSTER_VIEWS = 64;

/**
 * Where an instance record places the mesh, as far as the cluster test needs it: spheres, errors
 * and cones follow the instance. It should say what the material's vertex stage does with the
 * record. A column-major 4x4 matrix of f32 at byte `offset`, or a position (3 x f32), then
 * optionally a uniform scale (f32), a turn about +y (f32 times `yawScale` radians, as glam's
 * `Mat3::from_rotation_y`: -1 for a bearing that turns a mesh by minus itself) and a rotation (a
 * unit quaternion, x y z w): `position + yaw * rotation * (scale * p)`. Offsets in bytes,
 * multiples of 4. Rust: `clusters::InstanceTransform`.
 */
export type InstanceTransform =
    | { kind: 'matrix'; offset: number }
    | { kind: 'placement'; position: number; scale?: number | null; yaw?: number | null; yawScale?: number; rotation?: number | null };

/**
 * Where the cull reads a renderable's instances: none (the mesh once, where the renderable is),
 * every record of a buffer, or a view's compacted records (`InstanceCulling`), as many as word
 * `countWord` of its draws says.
 */
export type InstanceSource =
    | { kind: 'none' }
    | { kind: 'all'; records: GPUBuffer; count: number }
    | { kind: 'culled'; records: GPUBuffer; firstRecord: number; capacity: number; args: GPUBuffer; countWord: number };

/** Instances `source` can hold, which sizes the draw list. */
export function sourceCapacity(source: InstanceSource): number {
    return source.kind === 'none' ? 1 : source.kind === 'all' ? source.count : source.capacity;
}

/** `ClusterCull` in cluster_cull.wgsl: a renderable's parameters for one view's cut. */
export interface ClusterCullParams {
    world: ArrayLike<number>;
    kind: number;
    positionWord: number;
    scaleWord: number;
    yawWord: number;
    rotationWord: number;
    strideWords: number;
    firstRecord: number;
    instanceCount: number;
    countWord: number;
    capacity: number;
    vertexCount: number;
    flags: number;
    yawScale: number;
    stretch: number;
    /** the cut's view (set by `ClusterGpu.bind`) */
    view: number;
    /** the triangles its index buffer holds (set by `ClusterGpu.bind`) */
    triangleCapacity: number;
}

/**
 * A renderable's parameters: its world matrix; how its records (`stride` bytes each) place the
 * mesh, and where they come from; the draw list's capacity; the draw's vertices (3 x the mesh's
 * max triangles); whether to test the cones; and how much further the material may stretch an
 * instance (`ClusterLod.stretch`: no cone test past 1).
 */
export function clusterCullParams(world: ArrayLike<number>, transform: InstanceTransform | null, stride: number, source: InstanceSource, capacity: number, vertexCount: number, coneCulling: boolean, stretch: number): ClusterCullParams {
    const word = (offset: number) => offset / 4;
    let kind = KIND_NONE, positionWord = 0, scaleWord = NO_WORD, yawWord = NO_WORD, yawScale = 1, rotationWord = NO_WORD;
    if (source.kind !== 'none' && transform) {
        if (transform.kind === 'matrix') {
            kind = KIND_MATRIX;
            positionWord = word(transform.offset);
        } else {
            kind = KIND_PLACEMENT;
            positionWord = word(transform.position);
            scaleWord = transform.scale == null ? NO_WORD : word(transform.scale);
            yawWord = transform.yaw == null ? NO_WORD : word(transform.yaw);
            yawScale = transform.yawScale ?? 1;
            rotationWord = transform.rotation == null ? NO_WORD : word(transform.rotation);
        }
    }
    const [firstRecord, instanceCount, countWord] = source.kind === 'none' ? [0, 1, NO_WORD]
        : source.kind === 'all' ? [0, source.count, NO_WORD]
            : [source.firstRecord, 0, source.countWord];
    return {
        world, kind, positionWord, scaleWord, yawWord, rotationWord,
        strideWords: stride / 4, firstRecord, instanceCount, countWord, capacity, vertexCount,
        flags: coneCulling && stretch <= 1 ? FLAG_CONE : 0,
        yawScale, stretch: Math.max(stretch, 1), view: 0, triangleCapacity: 0,
    };
}

/** `params` as `ClusterCull`'s bytes. */
export function packClusterCull(params: ClusterCullParams): ArrayBuffer {
    const bytes = new ArrayBuffer(CLUSTER_CULL_BYTES);
    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    f.set(Array.from(params.world).slice(0, 16), 0);
    u.set([params.kind, params.positionWord, params.scaleWord, params.yawWord, params.rotationWord, params.strideWords,
        params.firstRecord, params.instanceCount, params.countWord, params.capacity, params.vertexCount, params.flags], 16);
    f[28] = params.yawScale;
    f[29] = params.stretch;
    u[30] = params.view;
    u[31] = params.triangleCapacity;
    return bytes;
}

/**
 * `ClusterView` in cluster_cull.wgsl: a view every renderable is cut for. Its view-projection's
 * frustum, the eye errors are measured from, pixels per radian (per metre when `orthographic`),
 * the distance errors are clamped to, and the budget in pixels.
 */
export interface ClusterView {
    viewProj: ArrayLike<number>;
    eye: ArrayLike<number>;
    pixelsPerUnit: number;
    near: number;
    threshold: number;
    orthographic: boolean;
}

/** `view` packed into `out` (`CLUSTER_VIEW_BYTES` at `byteOffset`); `null` packs zeros. */
export function packClusterView(out: ArrayBuffer, byteOffset: number, view: ClusterView | null): void {
    const f = new Float32Array(out, byteOffset, CLUSTER_VIEW_BYTES / 4);
    const u = new Uint32Array(out, byteOffset, CLUSTER_VIEW_BYTES / 4);
    f.fill(0);
    if (!view) return;
    frustumPlanes(view.viewProj).forEach((p, k) => f.set(p, k * 4));
    f[24] = view.eye[0];
    f[25] = view.eye[1];
    f[26] = view.eye[2];
    f[27] = view.pixelsPerUnit;
    f[28] = view.near;
    f[29] = view.threshold;
    u[30] = view.orthographic ? 1 : 0;
}

/**
 * A cluster view from a view's frustum (`viewProj`), its projection and view matrices, its
 * target's height in pixels and its `near` distance (errors of nearer spheres are clamped to it),
 * `threshold` pixels of error: orthographic when the projection is (Rust `cluster_view`).
 */
export function clusterViewOf(viewProj: ArrayLike<number>, projection: ArrayLike<number>, viewInverse: ArrayLike<number>, height: number, near: number, threshold: number): ClusterView {
    // a perspective projection's w is the view depth (±1 in z's column); an orthographic one's is 1
    const orthographic = projection[11] === 0;
    // pixels per radian at the centre, or per metre: half the height times y's scale
    const pixels = height * 0.5 * Math.abs(projection[5]);
    return { viewProj, eye: [viewInverse[12], viewInverse[13], viewInverse[14]], pixelsPerUnit: pixels, near, threshold, orthographic };
}

/** A perspective projection's near distance (0.01 when it has none to read). */
export function projectionNear(projection: ArrayLike<number>): number {
    const near = Math.abs(projection[14] / projection[10]);
    return Number.isFinite(near) && near > 0 ? near : 0.01;
}

/**
 * The triangles a cut needs, from a reading of the clusters it `claimed` and the triangles it
 * claimed room for: only clusters with an entry in its draw list (`list` of them) take room, so
 * past a full list the triangles are scaled by the clusters claimed over those with an entry.
 */
export function trianglesNeeded(claimed: number, list: number, triangles: number): number {
    if (claimed <= list || list === 0) return triangles;
    return Math.min(Math.floor(triangles * claimed / list), 0xffffffff);
}

/** The smallest power of two at or above `n` (1 for 0). */
function nextPowerOfTwo(n: number): number {
    let p = 1;
    while (p < n) p *= 2;
    return p;
}

/**
 * A buffer's length (a draw list's entries, an index buffer's triangles) sized to what its cut
 * needs, from the cull's readbacks.
 */
export class Sizer {
    /** readbacks in a row that needed at most a quarter of the length */
    private low = 0;

    /**
     * The length for `current` (0: none yet) given a reading of what the cut `needed` (if one
     * arrived): `initial` at first; then half again the need, rounded up to an eighth of its
     * power of two (at least `min`), at once when that is longer; and that when it has been at
     * most a quarter of the length for `SHRINK_AFTER` readings in a row. It grows no longer than
     * `max`, but a smaller `max` alone doesn't shrink it (the draw is capped instead).
     */
    sized(current: number, initial: number, min: number, max: number, needed: number | null): number {
        if (current === 0) return Math.min(initial, max);
        let length = current;
        if (needed !== null) {
            // (rounded up to an eighth of its power of two: steps small next to the length)
            const want = Math.max(Math.floor(needed * 3 / 2), 1);
            const step = Math.max(nextPowerOfTwo(want) / 8, 1);
            const target = Math.max(Math.min(Math.ceil(want / step) * step, max), Math.min(min, max));
            if (target > length) {
                length = target;
                this.low = 0;
            } else if (target * 4 <= length) {
                this.low++;
                if (this.low >= SHRINK_AFTER) {
                    length = target;
                    this.low = 0;
                }
            } else {
                this.low = 0;
            }
        }
        return length;
    }
}

let nextGpuId = 0;

/**
 * The cull's counts read back: how many clusters and triangles each cut claimed, drawn or not,
 * copied into one buffer a frame while none is in flight and read when mapped, a few frames
 * later, so the frame never waits.
 */
class Feedback {
    private staging: GPUBuffer | null = null;
    /** the cuts (`ClusterGpu` id, view) of the copy in flight, and its state */
    private pending: { cuts: [number, number][]; state: 'copied' | 'mapping' | 'mapped' | 'failed' } | null = null;
    /** the last readback's counts (clusters, triangles), until the next frame's */
    private needed = new Map<string, [number, number]>();

    /** The clusters and triangles cut (`id`, `view`) last claimed, if a reading arrived. */
    get(id: number, view: number): [number, number] | null {
        return this.needed.get(`${id}/${view}`) ?? null;
    }

    beginFrame(): void {
        this.needed.clear();
        const pending = this.pending;
        if (!pending || pending.state === 'copied' || pending.state === 'mapping') return;
        this.pending = null;
        if (pending.state === 'failed') return;
        const words = new Uint32Array(this.staging!.getMappedRange());
        pending.cuts.forEach(([id, view], k) => this.needed.set(`${id}/${view}`, [words[2 * k], words[2 * k + 1]]));
        this.staging!.unmap();
    }

    /** Copy each cut's claimed counts at the end of `encoder`, unless a readback is in flight. */
    copy(device: GPUDevice, encoder: GPUCommandEncoder, cuts: [ClusterGpu, number][]): void {
        if (this.pending) return;
        const read = cuts.flatMap(([gpu, view]) => {
            const cut = gpu.cut(view);
            return cut ? [[cut, gpu.id, view] as const] : [];
        });
        if (read.length === 0) return;
        const size = read.length * 8;
        if (!this.staging || this.staging.size < size) {
            this.staging?.destroy();
            this.staging = device.createBuffer({ label: 'ClusterCulling/Feedback', size: Math.max(nextPowerOfTwo(size), 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        }
        read.forEach(([cut, , ], k) => {
            encoder.copyBufferToBuffer(cut.args, CLAIMED_WORD * 4, this.staging!, k * 8, 4);
            encoder.copyBufferToBuffer(cut.args, TRIANGLES_CLAIMED_WORD * 4, this.staging!, k * 8 + 4, 4);
        });
        this.pending = { cuts: read.map(([, id, view]) => [id, view]), state: 'copied' };
    }

    /** The encoder holding the copy was submitted: map it. */
    submitted(): void {
        const pending = this.pending;
        if (!pending || pending.state !== 'copied') return;
        pending.state = 'mapping';
        this.staging!.mapAsync(GPUMapMode.READ).then(
            () => { pending.state = 'mapped'; },
            () => { pending.state = 'failed'; },
        );
    }
}

/**
 * The cluster cull's pipelines and the frame's views (one per renderer): `prepare` resets each
 * cut's draw and sizes its cull's dispatch from the visible instances, `cull` runs a workgroup
 * per visible instance, `finish` draws at most the index buffer's capacity.
 */
export class ClusterCulling {
    readonly cullLayout: GPUBindGroupLayout;
    readonly prepareLayout: GPUBindGroupLayout;
    private readonly prepare: GPUComputePipeline;
    private readonly cull: GPUComputePipeline;
    private readonly finish: GPUComputePipeline;
    private readonly views: GPUBuffer;
    private readonly viewBindGroup: GPUBindGroup;
    private readonly viewBytes = new ArrayBuffer(MAX_CLUSTER_VIEWS * CLUSTER_VIEW_BYTES);
    readonly feedback = new Feedback();

    constructor(private readonly device: GPUDevice) {
        const entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
        this.cullLayout = device.createBindGroupLayout({
            label: 'ClusterCulling/Cull',
            entries: [entry(0, 'uniform'), entry(1, 'read-only-storage'), entry(2, 'read-only-storage'), entry(3, 'storage'), entry(4, 'storage'), entry(5, 'storage')],
        });
        const viewLayout = device.createBindGroupLayout({ label: 'ClusterCulling/View', entries: [entry(0, 'read-only-storage')] });
        this.prepareLayout = device.createBindGroupLayout({
            label: 'ClusterCulling/Prepare',
            entries: [entry(10, 'uniform'), entry(11, 'read-only-storage'), entry(12, 'storage'), entry(13, 'storage')],
        });
        const module = device.createShaderModule({ label: 'ClusterCulling', code: `${CLUSTER_CULL_WGSL}\n${CLUSTER_MESH_WGSL}` });
        const pipeline = (entryPoint: string, layouts: GPUBindGroupLayout[]) => device.createComputePipeline({
            label: `ClusterCulling/${entryPoint}`,
            layout: device.createPipelineLayout({ label: 'ClusterCulling', bindGroupLayouts: layouts }),
            compute: { module, entryPoint },
        });
        this.prepare = pipeline('prepare', [this.prepareLayout]);
        this.cull = pipeline('cull', [this.cullLayout, viewLayout]);
        this.finish = pipeline('finish', [this.prepareLayout]);
        this.views = device.createBuffer({ label: 'ClusterCulling/Views', size: MAX_CLUSTER_VIEWS * CLUSTER_VIEW_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.viewBindGroup = device.createBindGroup({ label: 'ClusterCulling/View', layout: viewLayout, entries: [{ binding: 0, resource: { buffer: this.views } }] });
    }

    /**
     * The frame's views, by index (`ClusterGpu.bind`'s `view`), in one write: a write per view
     * would leave every cut with the last.
     */
    setViews(views: (ClusterView | null)[]): void {
        if (views.length > MAX_CLUSTER_VIEWS) throw new Error(`${views.length} cluster views, at most ${MAX_CLUSTER_VIEWS}`);
        views.forEach((view, k) => packClusterView(this.viewBytes, k * CLUSTER_VIEW_BYTES, view));
        this.device.queue.writeBuffer(this.views, 0, this.viewBytes, 0, Math.max(views.length, 1) * CLUSTER_VIEW_BYTES);
    }

    /** Start a frame's cull: collect the counts a finished readback holds. */
    beginFrame(): void {
        this.feedback.beginFrame();
    }

    /**
     * Run the cuts `[clusters, view]` (each bound with `ClusterGpu.bind`) in one compute pass:
     * every prepare, then every cull (dispatched indirectly, a workgroup per visible instance),
     * then every finish; then copy what each claimed for the readback (call `submitted` once
     * `encoder` is submitted).
     */
    encode(encoder: GPUCommandEncoder, cuts: [ClusterGpu, number][]): void {
        const bound = cuts.flatMap(([gpu, view]) => {
            const cut = gpu.cut(view);
            return cut?.bound ? [cut] : [];
        });
        const pass = encoder.beginComputePass({ label: 'Renderer/ClusterCulling', timestampWrites: gpuPass('Renderer/ClusterCulling') });
        pass.setPipeline(this.prepare);
        for (const cut of bound) {
            pass.setBindGroup(0, cut.bound!.prepare);
            pass.dispatchWorkgroups(1);
        }
        pass.setPipeline(this.cull);
        pass.setBindGroup(1, this.viewBindGroup);
        for (const cut of bound) {
            pass.setBindGroup(0, cut.bound!.cull);
            pass.dispatchWorkgroupsIndirect(cut.dispatch, 0);
        }
        pass.setPipeline(this.finish);
        for (const cut of bound) {
            pass.setBindGroup(0, cut.bound!.prepare);
            pass.dispatchWorkgroups(1);
        }
        pass.end();
        // what each cut claimed, to size its draw list to (a few frames later)
        this.feedback.copy(this.device, encoder, cuts);
    }

    /** The encoder `encode` recorded into was submitted. */
    submitted(): void {
        this.feedback.submitted();
    }
}

/** A view's cut of a renderable: its parameters, the draw list and its indirect draw, and the cull's indirect dispatch. */
export class Cut {
    readonly params: GPUBuffer;
    written: ArrayBuffer | null = null;
    draws: GPUBuffer;
    /** `draws`' length in entries (0 before the first bind) */
    capacity = 0;
    readonly list = new Sizer();
    /** the cut's triangles, 3 indices each (`entry << 8 | local vertex`) */
    indices: GPUBuffer;
    /** `indices`' room in triangles (0 before the first bind) */
    triangleCapacity = 0;
    readonly triangles = new Sizer();
    /** The indirect draw (`CLUSTER_DRAW_ARGS_BYTES`). */
    readonly args: GPUBuffer;
    readonly dispatch: GPUBuffer;
    bound: { records: GPUBuffer | null; count: GPUBuffer | null; cull: GPUBindGroup; prepare: GPUBindGroup } | null = null;
    /** the vertex stage's group 2, and the buffers it was made with */
    draw: { key: (GPUBuffer | null)[]; group: GPUBindGroup } | null = null;

    constructor(device: GPUDevice) {
        const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ label, size, usage });
        this.params = buffer('Clusters/Params', CLUSTER_CULL_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.draws = buffer('Clusters/Draws', 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.indices = buffer('Clusters/Indices', 12, GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        // (COPY_SRC: read back by the stats and the feedback)
        this.args = buffer('Clusters/Args', CLUSTER_DRAW_ARGS_BYTES, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.dispatch = buffer('Clusters/Dispatch', 16, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE);
    }

    /** The vertex stage's group 2 for this cut, once `ClusterGpu.bindDraw` made it. */
    get drawBindGroup(): GPUBindGroup | null {
        return this.draw?.group ?? null;
    }

    destroy(): void {
        for (const b of [this.params, this.draws, this.indices, this.args, this.dispatch]) b.destroy();
    }
}

function sameBytes(a: ArrayBuffer | null, b: ArrayBuffer): boolean {
    if (!a || a.byteLength !== b.byteLength) return false;
    const x = new Uint32Array(a), y = new Uint32Array(b);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
}

/** A renderable's cluster mesh on the GPU, and its cuts: one per view that draws it. */
export class ClusterGpu {
    /** Names its cuts' readbacks, whatever the scene does with the renderable. */
    readonly id = nextGpuId++;
    readonly mesh: GPUBuffer;
    /** Vertices a cluster is drawn as (3 x the mesh's max triangles). */
    readonly vertexCount: number;
    readonly clusterCount: number;
    /** bound in place of a missing instance buffer or count */
    private readonly empty: GPUBuffer;
    /** by view; none for views that never drew it */
    private cuts: (Cut | null)[] = [];
    /** the most triangles an index buffer holds (tests make it small) */
    triangleLimit = 0xffffffff;

    constructor(private readonly device: GPUDevice, mesh: ClusterMesh) {
        const words = mesh.gpuWords();
        this.mesh = device.createBuffer({ label: 'Clusters/Mesh', size: words.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
        new Uint32Array(this.mesh.getMappedRange()).set(words);
        this.mesh.unmap();
        this.vertexCount = 3 * mesh.maxTriangles();
        this.clusterCount = mesh.clusters.length;
        this.empty = device.createBuffer({ label: 'Clusters/Empty', size: 32, usage: GPUBufferUsage.STORAGE });
    }

    /** View `view`'s cut, if it was ever bound. */
    cut(view: number): Cut | null {
        return this.cuts[view] ?? null;
    }

    /**
     * Bind view `view`'s cut to `source`, with a draw list of at most `params.capacity` entries
     * sized to what the cut needs (from `culling`'s readbacks), and write the parameters if they
     * changed. True when the draw list or index buffer was remade: draws recorded with the old one
     * are stale.
     */
    bind(culling: ClusterCulling, view: number, source: InstanceSource, params: ClusterCullParams): boolean {
        const device = this.device;
        params.view = view;
        while (this.cuts.length <= view) this.cuts.push(null);
        const cut = this.cuts[view] ??= new Cut(device);
        const max = Math.max(params.capacity, 1);
        const reading = culling.feedback.get(this.id, view);
        const needed = reading ? reading[0] : null;
        const trianglesSeen = reading ? trianglesNeeded(reading[0], cut.capacity, reading[1]) : null;
        const length = cut.list.sized(cut.capacity, INITIAL_DRAWN, MIN_DRAWN, max, needed);
        // (the shader lists no more than the list holds, nor than `max`)
        params.capacity = Math.min(length, max);
        let grown = length !== cut.capacity;
        if (grown) {
            cut.capacity = length;
            cut.draws.destroy();
            cut.draws = device.createBuffer({ label: 'Clusters/Draws', size: cut.capacity * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            cut.bound = null;
        }
        // every cluster it may list at its largest, within what a binding holds
        const limits = device.limits;
        const binding = Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / 12);
        const most = Math.max(Math.min(max * (this.vertexCount / 3), binding, this.triangleLimit), 1);
        const triangles = cut.triangles.sized(cut.triangleCapacity, INITIAL_TRIANGLES, MIN_TRIANGLES, most, trianglesSeen);
        params.triangleCapacity = Math.min(triangles, most);
        if (triangles !== cut.triangleCapacity) {
            cut.triangleCapacity = triangles;
            cut.indices.destroy();
            cut.indices = device.createBuffer({ label: 'Clusters/Indices', size: triangles * 12, usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            cut.bound = null;
            grown = true;
        }
        const records = source.kind === 'none' ? null : source.records;
        const count = source.kind === 'culled' ? source.args : null;
        if (!cut.bound || cut.bound.records !== records || cut.bound.count !== count) {
            const entry = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer } });
            cut.bound = {
                records,
                count,
                cull: device.createBindGroup({
                    label: 'Clusters/Cull',
                    layout: culling.cullLayout,
                    entries: [entry(0, cut.params), entry(1, this.mesh), entry(2, records ?? this.empty), entry(3, cut.draws), entry(4, cut.args), entry(5, cut.indices)],
                }),
                prepare: device.createBindGroup({
                    label: 'Clusters/Prepare',
                    layout: culling.prepareLayout,
                    entries: [entry(10, cut.params), entry(11, count ?? this.empty), entry(12, cut.args), entry(13, cut.dispatch)],
                }),
            };
        }
        const bytes = packClusterCull(params);
        if (!sameBytes(cut.written, bytes)) {
            device.queue.writeBuffer(cut.params, 0, bytes);
            cut.written = bytes;
        }
        return grown;
    }

    /**
     * View `view`'s vertex stage group 2 (`clusterMeshBindGroupLayoutEntries`): the renderer's
     * normal and world matrices, the mesh, the cut's draw list and its bound records. Remade when
     * any of them changed; true then (bundles recorded the old one). The cut must be bound.
     */
    bindDraw(layout: GPUBindGroupLayout, view: number, normal: GPUBuffer, world: GPUBuffer, worldBytes: number): boolean {
        const cut = this.cuts[view];
        if (!cut) throw new Error('ClusterGpu.bindDraw: the cut is bound first');
        const records = cut.bound?.records ?? null;
        const key = [normal, world, cut.draws, records];
        if (cut.draw && cut.draw.key.every((b, i) => b === key[i])) return false;
        cut.draw = {
            key,
            group: this.device.createBindGroup({
                label: 'Clusters/Draw',
                layout,
                entries: [
                    { binding: 0, resource: { buffer: normal, size: 64 } },
                    { binding: 1, resource: { buffer: world, size: worldBytes } },
                    { binding: 2, resource: { buffer: this.mesh } },
                    { binding: 3, resource: { buffer: cut.draws } },
                    { binding: 4, resource: { buffer: records ?? this.empty } },
                ],
            }),
        };
        return true;
    }

    destroy(): void {
        this.mesh.destroy();
        this.empty.destroy();
        for (const cut of this.cuts) cut?.destroy();
        this.cuts = [];
    }
}

/**
 * Cluster LOD for a renderable (`Renderable.clusters`; Rust `ClusterLod`). Each frame, every view
 * that draws it (the camera and its velocity pass, the directional, spot and cascaded shadow
 * maps, rendered planar reflections, the sky occlusion's top-down view, the voxel clipmap's
 * regions and the ray tracing grid's box) draws the cut of `mesh` that view needs
 * (`Renderer.setClusterErrorThreshold`, times the view's scale:
 * `Renderer.setShadowClusterErrorScale`, `PlanarReflection.lodErrorScale`,
 * `SkyOcclusionOptions.lodErrorScale`; the clipmap's `clusterErrorVoxels` and the grid's
 * `clusterErrorCells` set their own) instead of the geometry, through a vertex stage generated
 * around the material's `vertex_main`. The geometry, which must be the mesh `mesh` was built
 * from, is still what impostor bakes, point-light (cube) shadows and `SceneVoxelGi` draw.
 *
 * The instances are the geometry's one instance buffer (if any), culled by the renderable's
 * `InstanceCulling` when it has one and placed as `transform` says. Set it before the renderable
 * is first drawn, or call `Renderer.invalidateBundle` after.
 */
export class ClusterLod {
    /** How an instance record places the mesh. Null: the instances are drawn where the renderable is (or there are none). */
    transform: InstanceTransform | null = null;
    /** Skip clusters whose every triangle faces away (on by default), where the material culls back faces and isn't transparent. */
    coneCulling = true;
    /**
     * Clusters drawn per frame in each view, at most. By default every cluster of every instance,
     * up to 4 194 304. Each view's cut keeps a draw list (8 bytes an entry) and an index buffer of
     * its clusters' triangles (12 bytes each), sized to what it needs as the cull reads back: a
     * cut that suddenly needs more than they hold leaves the clusters past them undrawn until the
     * readback lands, 2-3 frames later.
     */
    capacity: number | null = null;
    /**
     * How much further than `transform` the material may stretch or sway an instance about its
     * origin (1 by default): no point moves more than `stretch - 1` times its distance from the
     * origin. The cull's errors grow by it, its spheres by it and by how far their centres may
     * move, and its cones are off past 1.
     */
    stretch = 1;
    /** The mesh on the GPU, once a renderer drew it. */
    gpu: ClusterGpu | null = null;

    constructor(public readonly mesh: ClusterMesh) { }

    withTransform(transform: InstanceTransform): this {
        this.transform = transform;
        return this;
    }

    withConeCulling(on: boolean): this {
        this.coneCulling = on;
        return this;
    }

    withStretch(stretch: number): this {
        this.stretch = stretch;
        return this;
    }

    withCapacity(clusters: number): this {
        this.capacity = clusters;
        return this;
    }

    /**
     * Ready this frame's cut for view `view`: the GPU state made once, the cut bound to `source`
     * (records of `stride` bytes) with the parameters, and its vertex stage group 2 (`layout`)
     * over the renderer's normal and world matrices. `backFacesCulled`: the material culls back
     * faces (the cone test only removes what it would). True when what bundles recorded changed.
     */
    prepare(device: GPUDevice, culling: ClusterCulling, layout: GPUBindGroupLayout, matrices: { normal: GPUBuffer; world: GPUBuffer; worldBytes: number }, view: number, source: InstanceSource, stride: number, world: ArrayLike<number>, backFacesCulled: boolean): boolean {
        const gpu = this.gpu ??= new ClusterGpu(device, this.mesh);
        const every = Math.min(sourceCapacity(source) * gpu.clusterCount, DEFAULT_MAX_DRAWN);
        const capacity = Math.max(this.capacity ?? every, 1);
        const params = clusterCullParams(world, this.transform, stride, source, capacity, gpu.vertexCount, this.coneCulling && backFacesCulled, this.stretch);
        const grown = gpu.bind(culling, view, source, params);
        return gpu.bindDraw(layout, view, matrices.normal, matrices.world, matrices.worldBytes) || grown;
    }

    /** Frees the GPU mesh and cuts (made again if drawn). */
    destroy(): void {
        this.gpu?.destroy();
        this.gpu = null;
    }
}

/**
 * The instance records' layout of `r`'s geometry (its one instance buffer's vertex layout, once
 * initialized), or null without instances: what its cluster vertex stage reads records with.
 */
export function instanceLayoutOf(r: Renderable): InstanceLayout | null {
    if (!r.geometry.isInstancedGeometry || (r.geometry as InstancedGeometry).extraBuffers.length === 0) return null;
    return [...r.geometry.vertexBuffersDescriptors][1] ?? null;
}

/**
 * A shadow pass's hook for cluster LOD (the renderer's): draw `renderable`'s cluster cut for
 * cull view `view` into a depth target of `depthFormat` with `bias`, its matrices at `offset`,
 * through its material's cluster depth pipeline, setting the pipeline and groups 0 and 2 (group 1
 * is the pass's). False, having set nothing, when it has no cut there: draw the geometry.
 */
export type ClusterDepthDraw = (pass: GPURenderPassEncoder, renderable: Renderable, view: number, depthFormat: GPUTextureFormat, bias: DepthBias, offset: number) => boolean;

/**
 * Draw `cut` with `pipeline`: the cut's group 2 at the renderable's matrix `offset`, and the
 * indirect draw the cull wrote (the material's group 0 is the caller's). Rust `draw_cut`.
 */
export function drawCut(encoder: GPURenderPassEncoder | GPURenderBundleEncoder, cut: Cut, offset: number): boolean {
    const group = cut.drawBindGroup;
    if (!group) return false;
    encoder.setBindGroup(2, group, [offset, offset]);
    encoder.setIndexBuffer(cut.indices, 'uint32');
    encoder.drawIndexedIndirect(cut.args, 0);
    return true;
}
