import { mat4 } from 'gl-matrix';
import type { Scene } from '../objects/Scene';
import type { Renderable } from '../objects/Renderable';
import type { InstancedGeometry } from '../geometries/InstancedGeometry';
import { GatherSource, RtGrid, RtGridHandle, RtGridOptions, RtGridStats, RtPlacement, effectiveEpsilon, rtSurfaceWord } from './RtGrid';
import { RtMesh, boxesMeet, transformBox } from './RtMesh';
import { CLAIMED_WORD, ClusterView, Cut } from '../clusters/ClusterLod';

type Vec3 = [number, number, number];

/** What `Renderer.enableRtGrid` builds. Rust: `rt::SceneRtGridOptions`. */
export interface SceneRtGridOptions {
    grid?: RtGridOptions;
    /**
     * The error budget of the cluster cuts gathered into the grid (`Renderable.clusters`), in
     * cells: 1 keeps every error within a cell. Default 1.
     */
    clusterErrorCells?: number;
    /**
     * Rebuild every frame, not only when the box moved or what it holds changed (`dynamic`
     * renderables in it rebuild it every frame anyway). Default false.
     */
    rebuildEveryFrame?: boolean;
}

/** What the grid did, for a stats overlay. Rust: `rt::SceneRtGridStats`. */
export interface SceneRtGridStats {
    grid: RtGridStats;
    /** Sources gathered at the last rebuild (a renderable's mesh each). */
    sources: number;
    /** Whether this frame rebuilt it, and how many frames have. */
    rebuilt: boolean;
    rebuilds: number;
    /** CPU time of the last rebuild's recording, ms. */
    cpuMs: number;
}

/** A renderable's mesh for the gather: its vertex and index counts when made, its bounds, its buffer and triangles. */
interface SceneMesh {
    geometry: unknown;
    counts: [number, number];
    min: Vec3;
    max: Vec3;
    buffer: GPUBuffer;
    triangles: number;
}

/**
 * The renderer's ray tracing grid (`Renderer.enableRtGrid`): an `RtGrid` round the camera holding
 * the triangles of the renderables with `Renderable.rt`, read with `RT_GRID_WGSL` from any compute
 * pass after the frame's culling (`grid.bindGroupEntries`).
 *
 * Its box is one more cull view: `InstanceCulling` compacts the instances meeting it (by
 * `rtLodRange`, the camera's bands by default), and a cluster view cuts cluster LOD renderables
 * there (orthographic, `clusterErrorCells` cells of error). The gather then reads on the GPU each
 * clustered renderable's draw list, each instanced renderable's culled records (times its mesh,
 * placed by `Renderable.rtPlacement` or its cluster LOD's transform; all of its records without
 * `instanceCulling`), and each single mesh whose world box meets the grid's (one
 * box test per renderable on the CPU). It rebuilds when the box moves, when the static
 * renderables in it change (transform, visibility, surface), on `invalidate`, and every frame
 * while a `dynamic` one is in the scene. Rust: `rt::SceneRtGrid`.
 */
export class SceneRtGrid {
    public readonly options: { clusterErrorCells: number; rebuildEveryFrame: boolean };
    public readonly grid: RtGrid;
    private readonly meshes = new Map<Renderable, SceneMesh>();
    private key: unknown[] | null = null;
    private force = true;
    private rebuild = false;
    private built = false;
    private readonly _stats = { sources: 0, rebuilt: false, rebuilds: 0, cpuMs: 0 };
    /** renderables left out (no CPU geometry, or instances with no placement), warned once */
    private readonly warned = new WeakSet<Renderable>();

    constructor(private readonly device: GPUDevice, options: SceneRtGridOptions = {}) {
        this.grid = new RtGrid(device, options.grid ?? {});
        this.options = { clusterErrorCells: options.clusterErrorCells ?? 1, rebuildEveryFrame: options.rebuildEveryFrame ?? false };
    }

    /** Rebuild every frame or only when needed (see `SceneRtGridOptions.rebuildEveryFrame`). */
    setRebuildEveryFrame(on: boolean): void {
        this.options.rebuildEveryFrame = on;
    }

    /** A handle to the grid's buffers for an effect (`RtReflectionsEffect`): it follows them when the grid grows. */
    get handle(): RtGridHandle {
        return this.grid.handle;
    }

    /**
     * Rebuild next frame (after changing what the renderer can't see, such as a GPU-written
     * instance buffer of a renderable that isn't `dynamic`).
     */
    invalidate(): void {
        this.force = true;
    }

    get stats(): SceneRtGridStats {
        return { grid: this.grid.stats, ...this._stats };
    }

    /** Bytes on the GPU: the grid's and the renderables' meshes. */
    memoryBytes(): number {
        let bytes = this.grid.memoryBytes();
        for (const m of this.meshes.values()) bytes += m.buffer.size;
        return bytes;
    }

    /**
     * Where a renderable's records put its mesh: its `rtPlacement`, its cluster LOD's transform, or
     * nowhere (a single mesh); null when it is instanced and neither says.
     */
    private static placement(r: Renderable): RtPlacement | null {
        if (r.rtPlacement) return r.rtPlacement;
        if (r.clusters?.transform) return RtPlacement.instance(r.clusters.transform);
        return r.geometry.isInstancedGeometry ? null : RtPlacement.NONE;
    }

    /**
     * The cluster view of the box: orthographic, a pixel a cell, the cut's errors within
     * `clusterErrorCells`. Rust: `SceneRtGrid::cluster_view`.
     */
    clusterView(): ClusterView {
        const [lo, hi] = this.grid.bounds();
        return {
            viewProj: this.cullViewProj(mat4.create()),
            eye: [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5],
            pixelsPerUnit: 1 / this.grid.options.cell,
            near: 0.01,
            threshold: this.options.clusterErrorCells,
            orthographic: true,
        };
    }

    /** A cut gathered from the box's view grew: rebuild next frame too (its clusters past the old list were missing). */
    cutGrew(): void {
        this.force = true;
    }

    /**
     * Plan the frame, before the culling: follow `eye`, and rebuild if the box moved, what the
     * static renderables with `rt` are made of changed, a dynamic one is there, the buffers
     * overflowed, or it was forced.
     */
    plan(scene: Scene, eye: ArrayLike<number>): void {
        this.grid.pollReadback();
        const key: unknown[] = [];
        let anyDynamic = false;
        for (const r of scene.getOrderedObjects()) {
            const surface = r.rt;
            if (!surface || !r.visible || !r.geometry.initialized) continue;
            if (r.dynamic) {
                anyDynamic = true;
                continue;
            }
            const instances = r.geometry.isInstancedGeometry ? (r.geometry as InstancedGeometry).instanceCount : 1;
            key.push(r, ...r.worldMatrix.internalMat4, ...surface.albedo, rtSurfaceWord(surface), r.geometry.vertexCount, instances, r.rtPlacement);
        }
        const moved = this.grid.follow(eye);
        const changed = !this.key || this.key.length !== key.length || this.key.some((k, i) => k !== key[i]);
        this.key = key;
        const s = this.grid.stats;
        const overflowed = s.triangles > s.triangleCapacity || s.references > s.referenceCapacity;
        this.rebuild = moved || changed || anyDynamic || overflowed || this.options.rebuildEveryFrame || this.force;
        this.force = false;
        this._stats.rebuilt = this.rebuild;
    }

    /** Whether this frame rebuilds it (`plan`). */
    get rebuilding(): boolean {
        return this.rebuild;
    }

    /** The cull view of the box: an orthographic view along -z whose frustum is the box. */
    cullViewProj(out: mat4 = mat4.create()): mat4 {
        const [lo, hi] = this.grid.bounds();
        return mat4.orthoZO(out, lo[0], hi[0], lo[1], hi[1], -hi[2], -lo[2]);
    }

    /**
     * Rebuild the grid from `scene`'s renderables with `rt`, as culled and cut (`clusterCut`, the
     * renderer's: a renderable's cluster cut for the slot, if it has one) for cull view `slot`
     * (the box), into `encoder` (after the frame's culling); `afterSubmit` reads back what it needed.
     */
    build(encoder: GPUCommandEncoder, scene: Scene, slot: number, clusterCut?: (r: Renderable) => { mesh: GPUBuffer; cut: Cut } | null): void {
        const t0 = performance.now();
        this._stats.rebuilds++;
        const [lo, hi] = this.grid.bounds();
        const eps = effectiveEpsilon(this.grid.options);
        const sources: GatherSource[] = [];
        let reserve = 0;
        for (const r of scene.getOrderedObjects()) {
            const surface = r.rt;
            if (!surface || !r.visible || !r.geometry.initialized) continue;
            const placement = SceneRtGrid.placement(r);
            if (!placement) {
                this.warnOnce(r, 'is instanced with no rtPlacement');
                continue;
            }
            const geometry = r.geometry;
            // its cut for the box, on the cluster path
            const clustered = clusterCut?.(r) ?? null;
            if (!clustered && (!geometry.vertices?.length || !geometry.indices?.length)) {
                this.warnOnce(r, 'has no CPU geometry');
                continue;
            }
            let m = this.meshes.get(r);
            if (!clustered) {
                const counts: [number, number] = [geometry.vertices!.length, geometry.indices!.length];
                if (!m || m.geometry !== geometry || m.counts[0] !== counts[0] || m.counts[1] !== counts[1]) {
                    m?.buffer.destroy();
                    const mesh = RtMesh.fromGeometry(geometry);
                    m = { geometry, counts, min: mesh.min, max: mesh.max, buffer: mesh.createBuffer(this.device), triangles: mesh.triangleCount };
                    this.meshes.set(r, m);
                }
            }
            const world = r.worldMatrix.internalMat4;
            let records: GatherSource['records'] = null;
            let firstRecord = 0;
            let count: GatherSource['count'] = { fixed: 1 };
            if (geometry.isInstancedGeometry) {
                const instanced = geometry as InstancedGeometry;
                const first = instanced.extraBuffers[0];
                const stride = first?.stride ?? 0;
                const culling = r.instanceCulling;
                const draw = culling?.view(slot) ?? null;
                if (culling && draw) {
                    // the instance records the renderable's culling left for the box
                    records = { buffer: draw.instances, stride };
                    firstRecord = Math.floor(draw.instancesOffset / culling.culledStride);
                    count = { args: draw.args, word: draw.offset / 4 + 1, atMost: culling.count };
                } else {
                    const buffer = first?.gpuBuffer;
                    if (!buffer || (buffer.usage & GPUBufferUsage.STORAGE) === 0) {
                        this.warnOnce(r, 'has no instance buffer with STORAGE usage');
                        continue;
                    }
                    records = { buffer, stride };
                    count = { fixed: instanced.instanceCount };
                }
            } else if (!clustered) {
                const [a, b] = transformBox(world, m!.min, m!.max);
                if (!boxesMeet(a, b, lo, hi, eps)) continue;
                reserve += m!.triangles;
            }
            if (clustered) {
                // the cut's draw list: as many entries as it claimed, at most its length
                const { mesh, cut } = clustered;
                sources.push({
                    mesh, triangles: 0, draws: cut.draws, records, firstRecord: 0,
                    count: { args: cut.args, word: CLAIMED_WORD, atMost: cut.capacity },
                    world, placement: placement.wgsl(), surface, id: Math.max(scene.slotOf(r), 0),
                });
                continue;
            }
            sources.push({
                mesh: m!.buffer,
                triangles: m!.triangles,
                records,
                firstRecord,
                count,
                world,
                placement: placement.wgsl(),
                surface,
                id: Math.max(scene.slotOf(r), 0),
            });
        }
        this.grid.begin(encoder);
        // (room for the single meshes now; instances, which may mostly lie outside the box, grow
        // the buffers by the readback)
        this.grid.reserveTriangles(Math.min(reserve, 0xffffffff));
        this.grid.gatherSources(encoder, sources);
        this.grid.finish(encoder);
        this.built = true;
        this._stats.sources = sources.length;
        this._stats.cpuMs = performance.now() - t0;
    }

    /** After the frame holding a rebuild is submitted: read back what it needed. */
    afterSubmit(): void {
        if (!this.built) return;
        this.built = false;
        this.grid.readBack();
    }

    private warnOnce(r: Renderable, why: string): void {
        if (this.warned.has(r)) return;
        this.warned.add(r);
        console.warn(`rt grid: renderable ${r.geometry.label} ${why}: left out`);
    }

    destroy(): void {
        this.grid.destroy();
        for (const m of this.meshes.values()) m.buffer.destroy();
        this.meshes.clear();
    }
}
