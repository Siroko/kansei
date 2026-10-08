import { CULL_ARGS_BYTES } from './InstanceCulling';

/**
 * What became of the instances culled for one view in one frame, summed over the renderables
 * that draw in it (`Renderer.cullingStats`).
 */
export interface CullStats {
    /** Instances culled: each renderable's whole list. */
    tested: number;
    /** Outside their renderable's LOD band. */
    lodCulled: number;
    /** In the band, outside the view's frustum. */
    frustumCulled: number;
    /** In view but hidden behind the rest of the scene (with occlusion culling). */
    occlusionCulled: number;
    /** Drawn. */
    drawn: number;
    /** Triangles drawn: each draw's instances times its mesh's. */
    triangles: number;
}

/** No instances. */
export function emptyCullStats(): CullStats {
    return { tested: 0, lodCulled: 0, frustumCulled: 0, occlusionCulled: 0, drawn: 0, triangles: 0 };
}

/**
 * A view the renderer culls instances for: the camera, the directional shadow map's light
 * (`Renderer.enableShadows`), a cascade of the sun's shadows by index
 * (`Renderer.enableCascadedShadows`), the sky occlusion's top-down view
 * (`Renderer.enableSkyOcclusion`), a layer of the spot shadow atlas
 * (`Renderer.enableSpotShadows`), or the ray tracing grid's box (`Renderer.enableRtGrid`).
 */
export type CullViewKind = 'camera' | 'shadow' | `cascade${number}` | 'skyOcclusion' | 'spot' | `voxelGi${number}` | 'rtGrid';

/** Instance culling statistics of a recent frame, per view. */
export class CullingStats {
    constructor(
        /** The camera's frame number (`Camera.frame`) they are from. */
        public readonly frame: number,
        /** Every view culled that frame, in the renderer's order. */
        public readonly views: [CullViewKind, CullStats][],
    ) { }

    view(kind: CullViewKind): CullStats | null {
        return this.views.find(([k]) => k === kind)?.[1] ?? null;
    }

    /** The camera's view (zero if nothing was culled for it). */
    camera(): CullStats {
        return this.view('camera') ?? emptyCullStats();
    }
}

interface Entry {
    view: number;
    tested: number;
    args: GPUBuffer;
    offset: number;
}

/**
 * Reads the culled draws' counters back asynchronously: at most one copy in flight, collected
 * when mapped (a few frames later), so the frame never waits for the GPU.
 */
export class StatsReadback {
    enabled = false;
    private kinds: CullViewKind[] = [];
    private entries: Entry[] = [];
    private pending: { frame: number; kinds: CullViewKind[]; entries: [number, number][]; state: 'mapping' | 'mapped' | 'failed' } | null = null;
    private staging: GPUBuffer | null = null;
    private latestStats: CullingStats | null = null;

    get latest(): CullingStats | null {
        return this.enabled ? this.latestStats : null;
    }

    /** Start a frame culling `kinds` (by view index): collect a finished readback. */
    beginFrame(kinds: CullViewKind[]): void {
        this.entries.length = 0;
        this.kinds = kinds;
        const pending = this.pending;
        if (!pending || pending.state === 'mapping') return;
        this.pending = null;
        if (pending.state === 'failed') return;
        const staging = this.staging!;
        const words = new Uint32Array(staging.getMappedRange(0, pending.entries.length * CULL_ARGS_BYTES));
        const views: [CullViewKind, CullStats][] = [];
        pending.entries.forEach(([view, tested], k) => {
            // (the draw's index count, then its instance count; the culled counts at words 5-7)
            const a = words.subarray(k * (CULL_ARGS_BYTES / 4));
            const stats: CullStats = {
                tested,
                drawn: a[1],
                lodCulled: a[5],
                frustumCulled: a[6],
                occlusionCulled: a[7],
                triangles: Math.floor(a[0] / 3) * a[1],
            };
            const kind = pending.kinds[view];
            const sum = views.find(([k]) => k === kind);
            if (!sum) {
                views.push([kind, stats]);
                return;
            }
            const s = sum[1];
            s.tested += stats.tested;
            s.drawn += stats.drawn;
            s.lodCulled += stats.lodCulled;
            s.frustumCulled += stats.frustumCulled;
            s.occlusionCulled += stats.occlusionCulled;
            s.triangles += stats.triangles;
        });
        views.sort((a, b) => pending.kinds.indexOf(a[0]) - pending.kinds.indexOf(b[0]));
        staging.unmap();
        this.latestStats = new CullingStats(pending.frame, views);
    }

    /** A draw culled this frame for view `view`: `tested` instances, counted into `args` at `offset`. */
    record(view: number, tested: number, args: GPUBuffer, offset: number): void {
        if (this.enabled) this.entries.push({ view, tested, args, offset });
    }

    /** After the frame's culling is submitted: copy its counters for reading, unless a copy is still in flight. */
    endFrame(device: GPUDevice, frame: number): void {
        if (!this.enabled || this.pending || this.entries.length === 0) return;
        const size = this.entries.length * CULL_ARGS_BYTES;
        if (!this.staging || this.staging.size < size) {
            this.staging?.destroy();
            let capacity = CULL_ARGS_BYTES;
            while (capacity < size) capacity *= 2;
            this.staging = device.createBuffer({
                label: 'InstanceCulling/StatsReadback',
                size: capacity,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
        const staging = this.staging;
        const encoder = device.createCommandEncoder({ label: 'InstanceCulling/Stats' });
        this.entries.forEach((entry, k) => encoder.copyBufferToBuffer(entry.args, entry.offset, staging, k * CULL_ARGS_BYTES, CULL_ARGS_BYTES));
        device.queue.submit([encoder.finish()]);
        const pending = {
            frame,
            kinds: this.kinds,
            entries: this.entries.map((e): [number, number] => [e.view, e.tested]),
            state: 'mapping' as 'mapping' | 'mapped' | 'failed',
        };
        this.entries.length = 0;
        this.pending = pending;
        staging.mapAsync(GPUMapMode.READ, 0, size).then(
            () => { pending.state = 'mapped'; },
            () => { pending.state = 'failed'; },
        );
    }
}
