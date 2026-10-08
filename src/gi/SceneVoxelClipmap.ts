import type { mat4 } from 'gl-matrix';
import type { Scene } from '../objects/Scene';
import type { Renderable } from '../objects/Renderable';
import type { InstancedGeometry } from '../geometries/InstancedGeometry';
import type { ShadowMap } from '../shadows/ShadowMap';
import type { CascadedShadowSource } from '../shadows/ComputeShadows';
import type { CubeMapShadowMap } from '../shadows/CubeMapShadowMap';
import { drawGeometry } from '../culling/InstanceCulling';
import { transformBox } from '../rt/RtMesh';
import { ClipmapGiSettings, ClipmapInjection, defaultClipmapGiSettings } from './ClipmapInjection';
import { ClipmapProbeOptions, ClipmapProbes, defaultClipmapProbeOptions, sameClipmapProbeOptions } from './ClipmapProbes';
import { ClipRegion, ClipSurfaces, ClipmapVoxelizer, clipRegionBounds, sameClipRegion } from './ClipmapVoxelizer';
import { GiSurface, MeshVoxelizer } from './MeshVoxelizer';
import { gradientSkyLighting } from './ParticleConeShading';
import { ClipmapLayout, VoxelClipmap } from './VoxelClipmap';
import type { Vec3 } from './VoxelVolume';

/** What `Renderer.enableVoxelClipmap` builds. Rust: `gi::SceneVoxelClipmapOptions`. */
export interface SceneVoxelClipmapOptions {
    /** Levels, at most `MAX_CLIPMAP_LEVELS`: each covers twice the extent of the one before. Default 5. */
    levels: number;
    /** Voxels across each level in x and z (a multiple of 8). Default 64. */
    resolution: number;
    /** Voxels of each level in y (a multiple of 8). Default 32. */
    heightResolution: number;
    /** The finest level's voxels, metres. Default 0.5. */
    voxelSize: number;
    /**
     * A level's window moves in steps of this many of its voxels, once the camera is that far from
     * its centre: larger steps move it less often, each a thicker slab. Default 4.
     */
    snapVoxels: number;
    /** The reference radiance is stored against (see `VoxelClipmap`). Default 1. */
    radianceScale: number;
    /**
     * The finest levels dynamic renderables (`Renderable.dynamic`) are voxelized into, every frame
     * (coarser levels leave them out). Default 3.
     */
    dynamicLevels: number;
    /**
     * Regions (a slab a level's window moved into, or a whole window) voxelized a frame at most.
     * Each is a view instance culling culls for (`InstanceCulling.giLodRange`). Default 1.
     */
    jobsPerFrame: number;
    /**
     * The error budget of the cluster cuts drawn into the voxels (`Renderable.clusters`, with
     * cluster LOD), in voxels of the level voxelized: 1 keeps every error within a voxel. Default 1.
     */
    clusterErrorVoxels: number;
}

export function defaultSceneVoxelClipmapOptions(): SceneVoxelClipmapOptions {
    return { levels: 5, resolution: 64, heightResolution: 32, voxelSize: 0.5, snapVoxels: 4, radianceScale: 1, dynamicLevels: 3, jobsPerFrame: 1, clusterErrorVoxels: 1 };
}

/** The layout `options` give. */
export function clipmapLayoutOf(options: SceneVoxelClipmapOptions): ClipmapLayout {
    return new ClipmapLayout(options.levels, [options.resolution, options.heightResolution, options.resolution], options.voxelSize);
}

/**
 * A job a frame may run: voxelize `region` (cleared first), after which its level's window is at
 * `origin`. Rust: `gi::clipmap_scene::ClipJob`.
 */
export interface ClipJob {
    region: ClipRegion;
    origin: Vec3;
}

/** Frames a job slot's cull view stays after its job. */
const KEEP_ALIVE = 8;

/** A GI renderable's draws this frame: its voxelization pipeline and matrices, and its world box (null: unknown). */
interface ClipmapDraw {
    renderable: Renderable;
    pipeline: GPURenderPipeline;
    meshOffset: number;
    bounds: [Vec3, Vec3] | null;
}

/**
 * Voxel GI for an open scene's meshes (the renderer's, `Renderer.enableVoxelClipmap`): a voxel
 * clipmap (`VoxelClipmap`) around the camera instead of `SceneVoxelGi`'s one fixed box. Each
 * frame, after the shadow maps and before the GBuffer:
 * 1. each level's window follows the camera in steps (`snapVoxels`); the static renderables with
 *    a `Renderable.gi` surface are voxelized a region at a time (`jobsPerFrame`): the slab a
 *    window moved into, or a whole window when it is first filled, when one of them changed, or
 *    on `invalidate`; a window moves when its slab is done, so the levels always hold what their
 *    windows cover. The dynamic ones are voxelized into the finest `dynamicLevels` every frame;
 * 2. the voxels are lit level by level (`settings.levelsPerFrame`) by the renderer's lights
 *    through their shadow maps, or cones through the clipmap where the maps don't reach, plus
 *    their emission and a bounce of last frame's light.
 *
 * Instanced renderables with `instanceCulling` are voxelized as culled for each region's own cull
 * view (`giLodRange` picks the LOD they voxelize); renderables with cluster LOD would draw their
 * cut there (`clusterErrorVoxels`), which comes with cluster LOD in the TS engine (V-5): until
 * then they voxelize their mesh.
 *
 * Read it with `VoxelGIEffect.withClipmap` (screen-space cones), or with `CLIPMAP_WGSL` from any
 * compute pass; `enableProbes` keeps irradiance probes traced through it. Rust:
 * `gi::SceneVoxelClipmap`.
 */
export class SceneVoxelClipmap {
    public readonly settings: ClipmapGiSettings = defaultClipmapGiSettings();
    public readonly options: SceneVoxelClipmapOptions;
    /** The clipmap of the scene's light, for its consumers. */
    public readonly clipmap: VoxelClipmap;
    public readonly voxelizer: ClipmapVoxelizer;
    public readonly injection: ClipmapInjection;
    private readonly sky: GPUBuffer;
    /** The sky the voxels' and the probes' cones see past the clipmap: `sky`, or the one `useSkyLighting` gave. */
    private skySource: GPUBuffer;
    private _probes: ClipmapProbes | null = null;
    /** Levels to voxelize anew over their whole window (first fill, invalidation). */
    private readonly stale: boolean[];
    /** This frame's jobs, by slot. */
    private readonly _jobs: ClipJob[] = [];
    /**
     * Per job slot, the region it voxelized last and the frames since: its cull view stays for
     * `KEEP_ALIVE` frames after (so the cluster cuts drawn into it grow to what it needs).
     */
    private readonly slots: { region: ClipRegion | null, age: number }[];
    /** Regions to voxelize again: a cluster cut drawn into them was too small. */
    private readonly redo: ClipRegion[] = [];
    /** The level the round of `levelsPerFrame` lights next. */
    private nextLevel = 1;
    /** Per renderable, its geometry's local bounds and the counts they were taken at. */
    private readonly bounds = new WeakMap<Renderable, { geometry: unknown, counts: [number, number], min: Vec3, max: Vec3 }>();

    constructor(private readonly device: GPUDevice, options: Partial<SceneVoxelClipmapOptions> = {}) {
        this.options = { ...defaultSceneVoxelClipmapOptions(), ...options };
        const layout = clipmapLayoutOf(this.options);
        this.clipmap = new VoxelClipmap(device, layout, this.options.radianceScale);
        this.voxelizer = new ClipmapVoxelizer(device, layout, Math.max(this.options.jobsPerFrame, 1), this.options.dynamicLevels);
        // no sky past the clipmap until one is set
        const black = gradientSkyLighting([0, 0, 0], [0, 0, 0]);
        this.sky = device.createBuffer({ label: 'VoxelClipmap/Sky', size: black.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.sky, 0, black);
        this.skySource = this.sky;
        this.injection = new ClipmapInjection(device, this.clipmap, this.sky);
        this.stale = Array.from({ length: layout.levels }, () => true);
        this.slots = Array.from({ length: this.voxelizer.jobSlots }, () => ({ region: null, age: 0 }));
    }

    /**
     * Voxelize every level's whole window again, the finest first (after changing something the
     * voxelizer can't see, such as a material's texture).
     */
    public invalidate(): void {
        this.stale.fill(true);
    }

    /**
     * The sky past the clipmap, from `down` to `up` (scene radiance): see `gradientSkyLighting`.
     * Black until set. Ignored after `useSkyLighting`.
     */
    public setSkyGradient(up: Vec3, down: Vec3): void {
        this.device.queue.writeBuffer(this.sky, 0, gradientSkyLighting(up, down));
    }

    /**
     * Take the sky from `skyLighting` (a `SkyLighting` uniform such as
     * `SkyAtmosphere.bindings.skyLighting`) instead of the gradient.
     */
    public useSkyLighting(skyLighting: GPUBuffer): void {
        this.injection.setSky(skyLighting);
        this.skySource = skyLighting;
    }

    /**
     * Keep irradiance probes traced through the clipmap (`ClipmapProbes`), updated each frame after
     * its light, following the camera. Read them with `VoxelGIEffect.setClipmapProbes`,
     * `VolumetricFogEffect.setClipmapProbes` or a material's `CLIPMAP_PROBES_WGSL`. Calling it
     * again with other options builds them anew.
     */
    public enableProbes(options: Partial<ClipmapProbeOptions> = {}): ClipmapProbes {
        const full = { ...defaultClipmapProbeOptions(), ...options };
        if (!this._probes || !sameClipmapProbeOptions(this._probes.options, full)) {
            this._probes?.destroy();
            this._probes = new ClipmapProbes(this.device, this.clipmap, full);
        }
        return this._probes;
    }

    public disableProbes(): void {
        this._probes?.destroy();
        this._probes = null;
    }

    /** The probes, if enabled (change their `options` between frames). */
    public get probes(): ClipmapProbes | null {
        return this._probes;
    }

    /** Bytes on the GPU: the levels' radiance, the surface buffers and the probes. */
    public memoryBytes(): number {
        return this.clipmap.memoryBytes() + this.voxelizer.memoryBytes() + (this._probes?.memoryBytes() ?? 0);
    }

    /** Whether some level is still to be filled for the first time, or again (`invalidate`). */
    public get filling(): boolean {
        return this.stale.some((s) => s);
    }

    /** This frame's jobs (after `plan`). */
    public get jobs(): readonly ClipJob[] {
        return this._jobs;
    }

    /**
     * Plan the frame (the renderer's, before its culling, whose views the regions are) for `scene`
     * and a camera at `eye`: whether the static GI renderables changed (every level is voxelized
     * again then), and up to `jobsPerFrame` regions, the most urgent first: a level never filled or
     * stale before any that moved, then the level whose window the eye is furthest out of (as a
     * share of the window), finer levels first.
     */
    public plan(scene: Scene, eye: Vec3): void {
        // what the static surfaces are made of
        const key: unknown[] = [];
        for (const r of scene.getOrderedObjects()) {
            const surface = r.gi;
            if (!surface || !r.visible || !r.geometry.initialized || r.dynamic) continue;
            const instances = r.geometry.isInstancedGeometry ? (r.geometry as InstancedGeometry).instanceCount : 1;
            key.push(r, r.geometry, r.geometry.vertexCount, instances, ...r.worldMatrix.internalMat4,
                ...surface.albedo, ...(surface.emission ?? [0, 0, 0]), surface.opacity ?? 1);
        }
        const staticChanged = this.voxelizer.staticChanged(key);
        this._jobs.length = 0;
        if (!this.settings.enabled) {
            for (const s of this.slots) s.age++;
            return;
        }
        if (staticChanged) this.invalidate();
        const layout = this.clipmap.layout;
        const origins = Array.from({ length: layout.levels }, (_, level) => this.clipmap.origin(level));
        const taken = Array.from({ length: layout.levels }, () => false);
        for (let slot = 0; slot < this.voxelizer.jobSlots; slot++) {
            // a region whose cluster cuts grew first, if its window still holds some of it
            let job: ClipJob | null = null;
            while (!job && this.redo.length > 0) {
                const region = this.redo.pop()!;
                const origin = origins[region.level];
                if (!origin || taken[region.level]) continue;
                const lo = region.lo.map((v, a) => Math.max(v, origin[a])) as Vec3;
                const hi = region.lo.map((v, a) => Math.min(v + region.size[a], origin[a] + layout.dims[a])) as Vec3;
                if (hi.every((h, a) => h > lo[a])) job = { region: { level: region.level, lo, size: hi.map((h, a) => h - lo[a]) as Vec3 }, origin };
            }
            job ??= nextJob(layout, origins, this.stale, taken, eye, this.options.snapVoxels);
            if (!job) break;
            taken[job.region.level] = true;
            this.voxelizer.setJob(slot, job.region);
            this._jobs.push(job);
        }
        this.slots.forEach((state, slot) => {
            const job = this._jobs[slot];
            if (job) { state.region = job.region; state.age = 0; } else state.age++;
        });
    }

    /**
     * Job slot `slot`'s cull view this frame (a box round the region it voxelizes, or voxelized in
     * the last `KEEP_ALIVE` frames) and the voxel size of its level; null when it has none.
     */
    public giView(slot: number): { viewProj: mat4, voxelSize: number } | null {
        const state = this.slots[slot];
        if (!state?.region || state.age >= KEEP_ALIVE) return null;
        return { viewProj: this.voxelizer.jobViewProj(slot), voxelSize: this.clipmap.layout.levelVoxelSize(state.region.level) };
    }

    /**
     * A cluster cut drawn into job slot `slot`'s view grew: voxelize its region again (once the cut
     * holds what it needs, a few frames on, the redo is complete). For cluster LOD (V-5).
     */
    public cutGrew(slot: number): void {
        const region = this.slots[slot]?.region;
        if (region && !this.redo.some((r) => sameClipRegion(r, region))) this.redo.push(region);
    }

    /** The levels lit this frame: all, or the finest and the next `levelsPerFrame - 1` in turn. */
    private levelsToLight(): number[] {
        const levels = this.clipmap.layout.levels;
        const perFrame = this.settings.levelsPerFrame;
        if (perFrame <= 0 || perFrame >= levels) return Array.from({ length: levels }, (_, k) => k);
        const lit = [0];
        for (let k = 1; k < perFrame; k++) {
            lit.push(this.nextLevel);
            this.nextLevel = this.nextLevel + 1 >= levels ? 1 : this.nextLevel + 1;
        }
        return lit;
    }

    /** The world box of `r`'s mesh (null: instanced, drawn indirectly, or no CPU vertices). */
    private worldBounds(r: Renderable): [Vec3, Vec3] | null {
        const g = r.geometry;
        if (g.isInstancedGeometry || g.indirectArgsBuffer || !g.vertices?.length) return null;
        const counts: [number, number] = [g.vertices.length, g.indices?.length ?? 0];
        let b = this.bounds.get(r);
        if (!b || b.geometry !== g || b.counts[0] !== counts[0] || b.counts[1] !== counts[1]) {
            const { min, max } = g.bounds();
            b = { geometry: g, counts, min, max };
            this.bounds.set(r, b);
        }
        return transformBox(r.worldMatrix.internalMat4, b.min, b.max);
    }

    /**
     * Record the frame's clipmap (the renderer's, after its culling and shadow maps, before the
     * GBuffer): voxelize this frame's regions (`plan`) with the visible static GI renderables of
     * `scene` (matrices at `meshOffset` of `meshBindGroup`; instanced ones as culled for cull view
     * `jobView(slot)`), the windows move, the dynamic renderables go into the finest levels, the
     * levels due are lit through `shadowMap` (a cascaded map's widest cascade, the directional map,
     * or null) and `pointShadows`, then the probes update round `eye`.
     */
    public encode(
        encoder: GPUCommandEncoder,
        scene: Scene,
        meshBindGroup: GPUBindGroup,
        meshOffset: (renderable: Renderable) => number,
        jobView: (slot: number) => number,
        shadowMap: ShadowMap | CascadedShadowSource | null,
        pointShadows: CubeMapShadowMap | null,
        eye: Vec3,
    ): void {
        if (!this.settings.enabled) return;
        const voxelizer = this.voxelizer;
        this.injection.syncShadowMaps(shadowMap, pointShadows);
        this.injection.shadows.updateLights(scene.directionalLights, scene.pointLights, scene.areaLights, false, false);

        // the renderables drawn into voxels: visible, with a surface; in slot order
        const draws: ClipmapDraw[] = [];
        for (const r of scene.getOrderedObjects()) {
            if (!r.gi || !r.visible || !r.geometry.initialized) continue;
            const pipeline = r.material.getVoxelPipeline(this.device, voxelizer.id, r.geometry.vertexBuffersDescriptors,
                voxelizer.bindGroupLayout, voxelizer.fragment, MeshVoxelizer.TARGET_FORMAT, voxelizer.sampleCount);
            draws.push({ renderable: r, pipeline, meshOffset: meshOffset(r), bounds: this.worldBounds(r) });
        }
        draws.sort((a, b) => a.meshOffset - b.meshOffset);
        voxelizer.writeDraws(draws.map((d) => d.renderable.gi as GiSurface));
        const anyDynamic = draws.some((d) => d.renderable.dynamic);

        // the static renderables into this frame's regions, each cleared first
        const layout = this.clipmap.layout;
        this._jobs.forEach((job, slot) => {
            voxelizer.encodeClear(encoder, slot, this.clipmap);
            const groups = voxelizer.groups(ClipSurfaces.Static, slot);
            if (!groups) return;
            const region = clipRegionBounds(job.region, layout);
            groups.forEach((group, axis) => {
                const pass = voxelizer.beginPass(encoder, ClipSurfaces.Static, slot, axis);
                this.drawVoxels(pass, draws, region, layout.levelVoxelSize(job.region.level), jobView(slot), group, meshBindGroup);
                pass.end();
            });
        });
        // the windows move with the regions done (the shaders read the new origins from now on)
        for (const job of this._jobs) {
            const level = job.region.level;
            this.clipmap.setOrigin(level, job.origin);
            if (job.region.size.every((s, a) => s === layout.dims[a])) this.stale[level] = false;
        }
        voxelizer.setDynamicWindows(Array.from({ length: voxelizer.dynamicLevels }, (_, level) => this.clipmap.origin(level)));
        this.clipmap.upload();
        // the dynamic renderables into the finest levels' whole windows
        if (anyDynamic) {
            voxelizer.ensureDynamic();
            voxelizer.clearDynamic(encoder);
            for (let level = 0; level < voxelizer.dynamicLevels; level++) {
                const groups = voxelizer.groups(ClipSurfaces.Dynamic, level);
                const window = this.clipmap.levelBounds(level);
                if (!groups || !window) continue;
                groups.forEach((group, axis) => {
                    const pass = voxelizer.beginPass(encoder, ClipSurfaces.Dynamic, level, axis);
                    this.drawVoxels(pass, draws, window, layout.levelVoxelSize(level), null, group, meshBindGroup);
                    pass.end();
                });
            }
        }
        const dynamicLevels = voxelizer.dynamicLevels;
        this.injection.encode(encoder, voxelizer, this.levelsToLight(), (level) => anyDynamic && level < dynamicLevels, this.settings);
        this._probes?.encode(encoder, this.skySource, eye);
    }

    /**
     * Draw the static renderables of `draws` (with a cull view) or the dynamic ones (`view` null)
     * into a voxelization pass over world box `region` (voxels `reach` wide), with `group` as
     * group 3:
     * - a mesh whose world box misses the region by more than a voxel and 2 % of its own size is
     *   skipped;
     * - in a static region, an instanced renderable with `instanceCulling` draws its instances
     *   culled for the region (cluster LOD would draw its cut there: V-5);
     * - a dynamic one draws its instances as they are, unless culling would change their layout
     *   (crossfades): it is left out.
     * Rust: `renderers::renderer::draw_clipmap_voxels`.
     */
    private drawVoxels(
        pass: GPURenderPassEncoder,
        draws: readonly ClipmapDraw[],
        region: [Vec3, Vec3],
        reach: number,
        view: number | null,
        group: GPUBindGroup,
        meshBindGroup: GPUBindGroup,
    ): void {
        const [regionLo, regionHi] = region;
        const dynamic = view === null;
        draws.forEach((d, k) => {
            const r = d.renderable;
            if (r.dynamic !== dynamic) return;
            if (d.bounds) {
                const [lo, hi] = d.bounds;
                // (a material may sway or bend its vertices a little past the mesh's bounds)
                const margin = reach + 0.02 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
                if (lo.some((v, a) => v - margin > regionHi[a]) || hi.some((v, a) => v + margin < regionLo[a])) return;
            }
            const culling = r.instanceCulling;
            let culled = null;
            if (view !== null && culling && r.geometry.isInstancedGeometry) {
                // (the cluster LOD path draws the renderable's cut for the region here: V-5)
                culled = culling.view(view);
                if (!culled) return;
            } else if (dynamic && culling && culling.culledStride !== culling.stride) {
                return;
            }
            pass.setPipeline(d.pipeline);
            pass.setBindGroup(0, r.material.getBindGroup(this.device));
            pass.setBindGroup(2, meshBindGroup, [d.meshOffset, d.meshOffset]);
            pass.setBindGroup(3, group, [this.voxelizer.drawOffset(k)]);
            pass.setVertexBuffer(0, r.geometry.vertexBuffer!);
            pass.setIndexBuffer(r.geometry.indexBuffer!, r.geometry.indexFormat!);
            drawGeometry(pass, r.geometry, culled);
        });
    }

    public destroy(): void {
        this.disableProbes();
        this.voxelizer.destroy();
        this.injection.destroy();
        this.clipmap.destroy();
        this.sky.destroy();
    }
}

/**
 * The most urgent job for a clipmap laid out as `layout` whose levels' windows are at `origins`
 * (null: never filled), for an eye at `eye`, among the levels not `taken`: a level never filled or
 * `stale` (a whole window, at the eye) before any that moved, finer first; then the level whose
 * window the eye is furthest out of, as a share of the window, its move along the axis it moved
 * most (the slab of the moved window outside the old one). Null when no level needs one.
 * Rust: `gi::clipmap_scene::next_job`.
 */
export function nextJob(
    layout: ClipmapLayout,
    origins: readonly (Vec3 | null)[],
    stale: readonly boolean[],
    taken: readonly boolean[],
    eye: Vec3,
    snap: number,
): ClipJob | null {
    const dims = layout.dims;
    let best: { urgency: number, job: ClipJob } | null = null;
    for (let level = 0; level < layout.levels; level++) {
        if (taken[level]) continue;
        const origin = origins[level];
        let urgency: number;
        let job: ClipJob;
        if (origin && !stale[level]) {
            const target = layout.follow(level, origin, eye, snap);
            const delta = target.map((t, a) => t - origin[a]);
            if (delta.every((d) => d === 0)) continue;
            // the axis it moved most along (the first of equals)
            let axis = 0;
            for (let a = 1; a < 3; a++) if (Math.abs(delta[a]) > Math.abs(delta[axis])) axis = a;
            const moved = [...origin] as Vec3;
            moved[axis] = target[axis];
            const d = delta[axis];
            let region: ClipRegion;
            if (Math.abs(d) >= dims[axis]) {
                region = { level, lo: moved, size: [...dims] as Vec3 };
            } else {
                const lo = [...moved] as Vec3;
                const size = [...dims] as Vec3;
                if (d > 0) lo[axis] = origin[axis] + dims[axis];
                size[axis] = Math.abs(d);
                region = { level, lo, size };
            }
            urgency = Math.abs(d) / dims[axis];
            job = { region, origin: moved };
        } else {
            const at = origin ? layout.follow(level, origin, eye, snap) : layout.centredOrigin(level, eye, snap);
            urgency = Infinity;
            job = { region: { level, lo: at, size: [...dims] as Vec3 }, origin: at };
        }
        if (!best || urgency > best.urgency) best = { urgency, job };
    }
    return best?.job ?? null;
}
