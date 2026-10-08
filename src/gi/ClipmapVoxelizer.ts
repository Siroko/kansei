import { mat4 } from 'gl-matrix';
import { CAMERA_TEMPORAL_BYTES, LIGHT_UNIFORM_BYTES, cameraBindGroupLayoutEntries } from '../renderers/SharedLayouts';
import { gpuPass } from '../profiling/Profiler';
import { CLIPMAP_CLEAR_WGSL, VOXEL_FRAGMENT_WGSL } from './GiWGSL';
import { GiSurface, MeshVoxelizer, VOXELIZE_PARAMS_BYTES, VOXEL_DRAW_BYTES, axisViews, voxelizeParams } from './MeshVoxelizer';
import type { ClipmapLayout, VoxelClipmap } from './VoxelClipmap';
import { SURFACE_WORDS_PER_VOXEL, Vec3 } from './VoxelVolume';

/**
 * u32 per voxel in a clipmap level's surface buffers: `SURFACE_WORDS_PER_VOXEL`'s, then the
 * surface's area in the voxel (voxel faces, fixed point 1/256), which makes its opacity.
 * Rust: `gi::CLIP_SURFACE_WORDS`.
 */
export const CLIP_SURFACE_WORDS = SURFACE_WORDS_PER_VOXEL + 1;

/** Pixels a clipmap voxelization draws along each side of a voxel face, single-sampled (voxel_write.wgsl). */
const PIXELS_PER_VOXEL = 2;

/** Bytes of the WGSL `ClearRegion` (clipmap_clear.wgsl). */
const CLEAR_REGION_BYTES = 48;

/** A box of a clipmap level's lattice: voxels `lo .. lo + size` of level `level`. Rust: `gi::ClipRegion`. */
export interface ClipRegion {
    level: number;
    lo: Vec3;
    size: Vec3;
}

/** The region's world box. */
export function clipRegionBounds(region: ClipRegion, layout: ClipmapLayout): [Vec3, Vec3] {
    const s = layout.levelVoxelSize(region.level);
    return [region.lo.map((v) => v * s) as Vec3, region.lo.map((v, a) => (v + region.size[a]) * s) as Vec3];
}

/** Whether two regions are the same. */
export function sameClipRegion(a: ClipRegion | null, b: ClipRegion | null): boolean {
    return a === b || (a !== null && b !== null && a.level === b.level
        && a.lo.every((v, i) => v === b.lo[i]) && a.size.every((v, i) => v === b.size[i]));
}

/**
 * The three axis cameras and parameters that voxelize a region (`ClipRegion`): uploaded when the
 * region changes. Rust: `gi::clipmap_voxelize::RegionView`.
 */
class RegionView {
    /** Per axis: the camera's group 1 (view, projection, scene lights and temporal data, the last two unused). */
    readonly cameraGroups: GPUBindGroup[];
    readonly params: GPUBuffer[];
    readonly viewports: [number, number][] = [[0, 0], [0, 0], [0, 0]];
    region: ClipRegion | null = null;
    private readonly cameraBuffers: GPUBuffer[] = [];
    private readonly viewProjection = mat4.create();

    constructor(private readonly device: GPUDevice, cameraLayout: GPUBindGroupLayout) {
        const uniform = (label: string, size: number) =>
            device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.cameraGroups = [0, 1, 2].map(() => {
            const buffers = [64, 64, LIGHT_UNIFORM_BYTES, CAMERA_TEMPORAL_BYTES].map((size) => uniform('VoxelClipmap/AxisCamera', size));
            this.cameraBuffers.push(...buffers);
            return device.createBindGroup({
                label: 'VoxelClipmap/AxisCameraBG',
                layout: cameraLayout,
                entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            });
        });
        this.params = [0, 1, 2].map(() => uniform('VoxelClipmap/VoxelizeParams', VOXELIZE_PARAMS_BYTES));
    }

    /** Point the cameras at `region` of a clipmap laid out as `layout` (written if it changed). */
    set(layout: ClipmapLayout, region: ClipRegion): void {
        if (sameClipRegion(this.region, region)) return;
        const size = layout.levelVoxelSize(region.level);
        const queue = this.device.queue;
        axisViews(region.lo.map((v) => v * size) as Vec3, size, region.size).forEach((axis, a) => {
            queue.writeBuffer(this.cameraBuffers[4 * a], 0, axis.view as Float32Array);
            queue.writeBuffer(this.cameraBuffers[4 * a + 1], 0, axis.projection as Float32Array);
            const viewport = axis.viewport.map((v) => v * PIXELS_PER_VOXEL) as [number, number];
            queue.writeBuffer(this.params[a], 0, voxelizeParams({ ...axis, viewport }, layout.dims, region.lo, region.size, CLIP_SURFACE_WORDS, PIXELS_PER_VOXEL));
            this.viewports[a] = viewport;
            if (a === 0) mat4.multiply(this.viewProjection, axis.projection, axis.view);
        });
        this.region = { level: region.level, lo: [...region.lo] as Vec3, size: [...region.size] as Vec3 };
    }

    /** The view-projection of one of its axes: a box frustum round the region. */
    viewProj(): mat4 {
        return this.viewProjection;
    }

    destroy(): void {
        for (const b of [...this.cameraBuffers, ...this.params]) b.destroy();
    }
}

/** Which of a clipmap level's surface buffers a pass writes. Rust: `gi::ClipSurfaces`. */
export const ClipSurfaces = {
    /** The renderables that don't move, voxelized a region at a time (`ClipmapVoxelizer` jobs). */
    Static: 0,
    /** `Renderable.dynamic` ones, cleared and voxelized over the whole window every frame. */
    Dynamic: 1,
} as const;
export type ClipSurfaces = typeof ClipSurfaces[keyof typeof ClipSurfaces];

/**
 * The scene's meshes into a voxel clipmap's levels (`SceneVoxelClipmap`): `MeshVoxelizer`'s raster
 * voxelization (each renderable drawn along x, y and z through its own `vertex_main`, a pixel per
 * voxel, its surface written with storage atomics by `VOXEL_WRITE_WGSL`) over a region of a
 * level's window at a time, each voxel stored in its toroidal texel.
 *
 * Static renderables go into each level's static buffer by jobs: the slab a level's window moved
 * into, or a whole window when it is first filled or invalidated; the region is cleared first.
 * Dynamic ones go into the dynamic buffers of the finest `dynamicLevels` levels, whole windows,
 * every frame. Rust: `gi::ClipmapVoxelizer`.
 */
export class ClipmapVoxelizer {
    /** Tells voxelizers apart in the materials' pipeline caches. */
    public readonly id = MeshVoxelizer.nextId();
    /** Group 3 of the voxelization pipelines (as `MeshVoxelizer`'s). */
    public readonly bindGroupLayout: GPUBindGroupLayout;
    /** The engine's fragment stage (`voxel_fragment`). */
    public readonly fragment: { module: GPUShaderModule, entryPoint: string };

    private readonly staticSurfaceBuffers: GPUBuffer[];
    private readonly dynamicSurfaceBuffers: (GPUBuffer | null)[];
    private readonly target: GPUTexture;
    private readonly targetView: GPUTextureView;
    private draws: GPUBuffer;
    private readonly drawStride: number;
    private drawCapacity = 16;
    /** Per job slot: its cameras and its region's clear. */
    private readonly jobs: { view: RegionView, clear: GPUBuffer }[];
    /** Per dynamic level: its cameras over the level's window. */
    private readonly dynamicViews: RegionView[];
    private readonly clearPipeline: GPUComputePipeline;
    private readonly clearBGL: GPUBindGroupLayout;
    /** What the static surfaces hold (`staticChanged`). */
    private staticKey: unknown[] | null = null;

    /**
     * A voxelizer for `layout`, with `jobSlots` regions voxelized a frame at most and dynamic
     * renderables in the finest `dynamicLevels` levels.
     */
    constructor(private readonly device: GPUDevice, public readonly layout: ClipmapLayout, jobSlots: number, dynamicLevels: number) {
        const fragment = GPUShaderStage.FRAGMENT;
        this.bindGroupLayout = device.createBindGroupLayout({
            label: 'VoxelClipmap/VoxelizeBGL',
            entries: [
                { binding: 100, visibility: fragment, buffer: { type: 'uniform' } },
                { binding: 101, visibility: fragment, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: VOXEL_DRAW_BYTES } },
                { binding: 102, visibility: fragment, buffer: { type: 'storage' } },
            ],
        });
        this.fragment = {
            module: device.createShaderModule({ label: 'VoxelClipmap/VoxelFragment', code: VOXEL_FRAGMENT_WGSL }),
            entryPoint: 'voxel_fragment',
        };
        const side = Math.max(...layout.dims) * PIXELS_PER_VOXEL;
        this.target = device.createTexture({
            label: 'VoxelClipmap/VoxelizeTarget',
            size: [side, side],
            sampleCount: 1,
            format: MeshVoxelizer.TARGET_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.targetView = this.target.createView();
        const alignment = device.limits.minUniformBufferOffsetAlignment;
        this.drawStride = Math.ceil(VOXEL_DRAW_BYTES / alignment) * alignment;
        this.draws = device.createBuffer({ label: 'VoxelClipmap/Draws', size: this.drawStride * this.drawCapacity, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

        const compute = GPUShaderStage.COMPUTE;
        this.clearBGL = device.createBindGroupLayout({
            label: 'VoxelClipmap/ClearBGL',
            entries: [
                { binding: 0, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 1, visibility: compute, buffer: { type: 'storage' } },
                { binding: 2, visibility: compute, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
            ],
        });
        this.clearPipeline = device.createComputePipeline({
            label: 'VoxelClipmap/Clear',
            layout: device.createPipelineLayout({ label: 'VoxelClipmap/Clear', bindGroupLayouts: [this.clearBGL] }),
            compute: { module: device.createShaderModule({ label: 'VoxelClipmap/Clear', code: CLIPMAP_CLEAR_WGSL }), entryPoint: 'main' },
        });

        const cameraLayout = device.createBindGroupLayout({ label: 'VoxelClipmap/CameraBGL', entries: cameraBindGroupLayoutEntries() });
        this.staticSurfaceBuffers = Array.from({ length: layout.levels }, () => this.surfaceBuffer('VoxelClipmap/StaticSurfaces'));
        const levels = Math.min(Math.max(dynamicLevels, 0), layout.levels);
        this.dynamicSurfaceBuffers = Array.from({ length: levels }, () => null);
        this.jobs = Array.from({ length: Math.max(jobSlots, 1) }, () => ({
            view: new RegionView(device, cameraLayout),
            clear: device.createBuffer({ label: 'VoxelClipmap/ClearParams', size: CLEAR_REGION_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        }));
        this.dynamicViews = Array.from({ length: levels }, () => new RegionView(device, cameraLayout));
    }

    private surfaceBuffer(label: string): GPUBuffer {
        return this.device.createBuffer({
            label,
            size: this.layout.voxelCount() * CLIP_SURFACE_WORDS * 4,
            // (COPY_SRC: readable in tests)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    /** Samples of its passes' target: one (it draws `PIXELS_PER_VOXEL` squared pixels a voxel face instead). */
    public get sampleCount(): number { return 1; }

    /**
     * Level `level`'s static surfaces: `CLIP_SURFACE_WORDS` u32 per voxel, by texel (x fastest),
     * each texel holding the voxel of the window congruent to it.
     */
    public staticSurfaces(level: number): GPUBuffer {
        return this.staticSurfaceBuffers[level];
    }

    /** Level `level`'s dynamic surfaces this frame (null until one is voxelized there). */
    public dynamicSurfaces(level: number): GPUBuffer | null {
        return this.dynamicSurfaceBuffers[level] ?? null;
    }

    /** The finest levels dynamic renderables go into. */
    public get dynamicLevels(): number {
        return this.dynamicSurfaceBuffers.length;
    }

    /** Regions voxelized a frame at most. */
    public get jobSlots(): number {
        return this.jobs.length;
    }

    /**
     * Whether the static surfaces must be voxelized again for `key` (a description of the static
     * renderables, as `MeshVoxelizer.staticChanged`); remembers it.
     */
    public staticChanged(key: unknown[]): boolean {
        const old = this.staticKey;
        if (old && old.length === key.length && old.every((v, i) => v === key[i])) return false;
        this.staticKey = key;
        return true;
    }

    /** Upload the draws' constant surfaces (one slot each, in draw order), growing their buffer. */
    public writeDraws(surfaces: readonly GiSurface[]): void {
        if (surfaces.length === 0) return;
        if (surfaces.length > this.drawCapacity) {
            this.drawCapacity = 2 ** Math.ceil(Math.log2(surfaces.length));
            this.draws.destroy();
            this.draws = this.device.createBuffer({ label: 'VoxelClipmap/Draws', size: this.drawStride * this.drawCapacity, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        }
        const data = new Float32Array(this.drawStride / 4 * surfaces.length);
        surfaces.forEach((s, k) => {
            const at = k * this.drawStride / 4;
            data.set(s.albedo, at);
            data[at + 3] = Math.max(s.opacity ?? 1, 0);
            data.set(s.emission ?? [0, 0, 0], at + 4);
        });
        this.device.queue.writeBuffer(this.draws, 0, data);
    }

    /** The dynamic offset of draw `k`'s surface. */
    public drawOffset(k: number): number {
        return k * this.drawStride;
    }

    /** Ready job slot `slot` for `region` (its cameras and parameters). */
    public setJob(slot: number, region: ClipRegion): void {
        const { view, clear } = this.jobs[slot];
        view.set(this.layout, region);
        const data = new ArrayBuffer(CLEAR_REGION_BYTES);
        const u32 = new Uint32Array(data);
        new Int32Array(data).set(region.lo, 0);
        u32[3] = CLIP_SURFACE_WORDS;
        u32.set(region.size, 4);
        u32.set(this.layout.dims, 8);
        this.device.queue.writeBuffer(clear, 0, data);
    }

    /** Job slot `slot`'s view-projection (a box frustum round its region). */
    public jobViewProj(slot: number): mat4 {
        return this.jobs[slot].view.viewProj();
    }

    /**
     * Ready the dynamic views over each dynamic level's window at `origins` (null: the level has no
     * window yet).
     */
    public setDynamicWindows(origins: readonly (Vec3 | null)[]): void {
        this.dynamicViews.forEach((view, level) => {
            const lo = origins[level];
            if (lo) view.set(this.layout, { level, lo, size: this.layout.dims });
        });
    }

    /** Make the dynamic buffers (the first frame a dynamic renderable is voxelized). */
    public ensureDynamic(): void {
        for (let k = 0; k < this.dynamicSurfaceBuffers.length; k++) {
            this.dynamicSurfaceBuffers[k] ??= this.surfaceBuffer('VoxelClipmap/DynamicSurfaces');
        }
    }

    /**
     * Record clearing job slot `slot`'s region of its level: its static surfaces and its texels of
     * the clipmap's radiance (no stale light there before the injection relights it).
     */
    public encodeClear(encoder: GPUCommandEncoder, slot: number, clipmap: VoxelClipmap): void {
        const { view, clear } = this.jobs[slot];
        const region = view.region;
        if (!region) return;
        const group = this.device.createBindGroup({
            label: 'VoxelClipmap/ClearBG',
            layout: this.clearBGL,
            entries: [
                { binding: 0, resource: { buffer: clear } },
                { binding: 1, resource: { buffer: this.staticSurfaceBuffers[region.level] } },
                { binding: 2, resource: clipmap.view(region.level) },
            ],
        });
        const pass = encoder.beginComputePass({ label: 'VoxelClipmap/Clear' });
        pass.setPipeline(this.clearPipeline);
        pass.setBindGroup(0, group);
        const [w, h, d] = region.size;
        pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        pass.end();
    }

    /** Record clearing every dynamic buffer. */
    public clearDynamic(encoder: GPUCommandEncoder): void {
        for (const buffer of this.dynamicSurfaceBuffers) if (buffer) encoder.clearBuffer(buffer);
    }

    private viewOf(set: ClipSurfaces, index: number): RegionView {
        return set === ClipSurfaces.Static ? this.jobs[index].view : this.dynamicViews[index];
    }

    /**
     * Group 3 for each axis of a pass into `set`: job slot `index`'s region (static), or dynamic
     * level `index`'s window; null when it has none.
     */
    public groups(set: ClipSurfaces, index: number): GPUBindGroup[] | null {
        const view = this.viewOf(set, index);
        if (!view.region) return null;
        const surfaces = set === ClipSurfaces.Static ? this.staticSurfaceBuffers[view.region.level] : this.dynamicSurfaceBuffers[index];
        if (!surfaces) return null;
        return view.params.map((params) => this.device.createBindGroup({
            label: 'VoxelClipmap/VoxelizeBG',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 100, resource: { buffer: params } },
                { binding: 101, resource: { buffer: this.draws, offset: 0, size: VOXEL_DRAW_BYTES } },
                { binding: 102, resource: { buffer: surfaces } },
            ],
        }));
    }

    /**
     * Begin the voxelization pass of `axis` (0 x, 1 y, 2 z) of `set` / `index` (as `groups`): the
     * axis' camera bound as group 1 and its viewport set.
     */
    public beginPass(encoder: GPUCommandEncoder, set: ClipSurfaces, index: number, axis: number): GPURenderPassEncoder {
        const view = this.viewOf(set, index);
        const label = set === ClipSurfaces.Static
            ? ['VoxelClipmap/VoxelizeX', 'VoxelClipmap/VoxelizeY', 'VoxelClipmap/VoxelizeZ'][axis]
            : ['VoxelClipmap/DynamicX', 'VoxelClipmap/DynamicY', 'VoxelClipmap/DynamicZ'][axis];
        const pass = encoder.beginRenderPass({
            label,
            colorAttachments: [{ view: this.targetView, clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'discard' }],
            timestampWrites: gpuPass(label),
        });
        const [w, h] = view.viewports[axis];
        pass.setViewport(0, 0, Math.max(w, 1), Math.max(h, 1), 0, 1);
        pass.setBindGroup(1, view.cameraGroups[axis]);
        return pass;
    }

    /** Bytes of the surface buffers. */
    public memoryBytes(): number {
        const buffers = this.staticSurfaceBuffers.length + this.dynamicSurfaceBuffers.filter((b) => b).length;
        return buffers * this.layout.voxelCount() * CLIP_SURFACE_WORDS * 4;
    }

    public destroy(): void {
        for (const b of [...this.staticSurfaceBuffers, ...this.dynamicSurfaceBuffers, this.draws]) b?.destroy();
        for (const { view, clear } of this.jobs) { view.destroy(); clear.destroy(); }
        for (const view of this.dynamicViews) view.destroy();
        this.target.destroy();
    }
}
