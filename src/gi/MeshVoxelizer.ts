import { mat4, vec3 } from 'gl-matrix';
import { drawGeometry } from '../culling/InstanceCulling';
import type { Renderable } from '../objects/Renderable';
import { CAMERA_TEMPORAL_BYTES, LIGHT_UNIFORM_BYTES, cameraBindGroupLayoutEntries } from '../renderers/SharedLayouts';
import { gpuPass } from '../profiling/Profiler';
import { VOXEL_FRAGMENT_WGSL } from './GiWGSL';
import { SURFACE_WORDS_PER_VOXEL, Vec3, VolumeLayout } from './VoxelVolume';

/**
 * A renderable's surface in voxel GI (`Renderable.gi`): the albedo it reflects and the light it
 * emits (scene radiance, cd/m²), the same over the whole renderable. For a textured surface, use
 * its mean colour, or give its material a `voxelFragmentEntry` that reads the texture.
 * Rust: `gi::GiSurface`.
 */
export interface GiSurface {
    albedo: Vec3;
    /** Default black. */
    emission?: Vec3;
    /**
     * Scales how much its area makes a clipmap's voxels opaque (1 by default). A volume
     * (`SceneVoxelGi`) keeps its voxels opaque.
     */
    opacity?: number;
}

/** Bytes of the WGSL `KanseiVoxelizeParams` (voxel_write.wgsl; Rust `VoxelizeParamsGpu`). */
export const VOXELIZE_PARAMS_BYTES = 144;
/** Bytes of the WGSL `KanseiVoxelDraw` (Rust `VoxelDrawGpu`). */
export const VOXEL_DRAW_BYTES = 32;

let nextVoxelizerId = 1;

/** Which surface buffer a voxelization pass writes. Rust: `gi::SurfaceSet`. */
export const SurfaceSet = {
    /** Renderables that don't move: voxelized again only when one of them changes. */
    Static: 0,
    /** `Renderable.dynamic` ones: cleared and voxelized every frame. */
    Dynamic: 1,
} as const;
export type SurfaceSet = typeof SurfaceSet[keyof typeof SurfaceSet];

/**
 * One axis of a voxelization: an orthographic camera looking along `look` over a box, a pixel per
 * voxel in a `viewport` of the box's cross-section, and the map from its clip space to the box's
 * voxel coordinates. Rust: `gi::voxelize::AxisView`.
 */
export interface AxisView {
    view: mat4;
    projection: mat4;
    look: Vec3;
    viewport: [number, number];
    clipToVoxel: mat4;
}

/**
 * The three axes (x, y, z) of a voxelization of the box from `lo` (world) of `dims` voxels of
 * `voxelSize`, a pixel per voxel. Rust: `gi::voxelize::axis_views`.
 */
export function axisViews(lo: Vec3, voxelSize: number, dims: Vec3): AxisView[] {
    const extent = dims.map((d) => d * voxelSize) as Vec3;
    const centre = lo.map((l, i) => l + extent[i] * 0.5) as Vec3;
    const worldToVoxel = mat4.create();
    mat4.fromScaling(worldToVoxel, [1 / voxelSize, 1 / voxelSize, 1 / voxelSize]);
    mat4.translate(worldToVoxel, worldToVoxel, [-lo[0], -lo[1], -lo[2]]);
    const [dx, dy, dz] = dims;
    // (looking along, up, viewport width and height in voxels, extents across, depth)
    const axes: [Vec3, Vec3, [number, number], [number, number], number][] = [
        [[-1, 0, 0], [0, 1, 0], [dz, dy], [extent[2], extent[1]], extent[0]],
        [[0, -1, 0], [0, 0, -1], [dx, dz], [extent[0], extent[2]], extent[1]],
        [[0, 0, -1], [0, 1, 0], [dx, dy], [extent[0], extent[1]], extent[2]],
    ];
    return axes.map(([look, up, viewport, across, depth]) => {
        const eye = vec3.scaleAndAdd(vec3.create(), centre, look, -depth * 0.5);
        const view = mat4.lookAt(mat4.create(), eye, centre, up);
        const projection = mat4.orthoZO(mat4.create(), -across[0] * 0.5, across[0] * 0.5, -across[1] * 0.5, across[1] * 0.5, 0, depth);
        const clipToVoxel = mat4.multiply(mat4.create(), projection, view);
        mat4.invert(clipToVoxel, clipToVoxel);
        mat4.multiply(clipToVoxel, worldToVoxel, clipToVoxel);
        return { view, projection, look, viewport, clipToVoxel };
    });
}

/**
 * The WGSL `KanseiVoxelizeParams` of an axis for region `[lo, lo + region)` of a volume of `dims`
 * voxels (stored toroidally, `words` u32 a voxel), drawn at `pixels` per voxel along each side.
 */
export function voxelizeParams(axis: AxisView, dims: Vec3, lo: Vec3, region: Vec3, words: number, pixels: number): ArrayBuffer {
    const data = new ArrayBuffer(VOXELIZE_PARAMS_BYTES);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);
    const i32 = new Int32Array(data);
    f32.set(axis.clipToVoxel, 0);
    f32.set(axis.look, 16);
    u32[19] = words;
    f32.set(axis.viewport, 20);
    f32[22] = 1 / (pixels * pixels);
    u32.set(dims, 24);
    i32.set(lo, 28);
    u32.set(region, 32);
    return data;
}

/**
 * Meshes into voxels through the rasterizer (miaumiau.cat/?p=1457's voxelization, without its CPU
 * triangle splitting or its "big triangle" pass): every GI renderable is drawn three times, with
 * an orthographic camera along x, y and z over the volume and a pixel per voxel, through its
 * material's own `vertex_main` (instancing and vertex animation voxelize as they draw), and a
 * fragment stage that writes the voxel it lands in with storage atomics (`VOXEL_WRITE_WGSL`): the
 * engine's, with the renderable's constant `GiSurface`, or the material's `voxelFragmentEntry`.
 *
 * Each voxel keeps its surfaces' average albedo and normal and their brightest emission
 * (`SURFACE_WORDS_PER_VOXEL`). Renderables that don't move go into the static buffer, again only
 * when one of them changes; `Renderable.dynamic` ones into the dynamic buffer, every frame.
 * `SceneVoxelGi` lights both into its volume. Rust: `gi::MeshVoxelizer`.
 */
export class MeshVoxelizer {
    /** The dummy target's format and samples: masked off, it only sets the fragments' coverage. */
    static readonly TARGET_FORMAT: GPUTextureFormat = 'r8unorm';
    static readonly SAMPLE_COUNT = 4;

    /** Tells voxelizers apart in the materials' pipeline caches. */
    public readonly id = nextVoxelizerId++;
    /** Group 3 of the voxelization pipelines: the axis' parameters (100), the draw's surface at a dynamic offset (101), the surfaces written (102). */
    public readonly bindGroupLayout: GPUBindGroupLayout;
    /** The engine's fragment stage (`voxel_fragment`). */
    public readonly fragment: { module: GPUShaderModule, entryPoint: string };
    /** The static renderables' voxels: `SURFACE_WORDS_PER_VOXEL` u32 per voxel, x fastest. */
    public readonly staticSurfaces: GPUBuffer;
    private _dynamicSurfaces: GPUBuffer | null = null;

    private readonly cameraGroups: GPUBindGroup[];
    private readonly cameraBuffers: GPUBuffer[] = [];
    private readonly params: GPUBuffer[];
    private draws: GPUBuffer;
    private readonly drawStride: number;
    private drawCapacity = 16;
    private staticGroups: GPUBindGroup[] = [];
    private dynamicGroups: GPUBindGroup[] = [];
    private readonly target: GPUTexture;
    private readonly targetView: GPUTextureView;
    /** What the static surfaces hold (`staticChanged`), once voxelized. */
    private staticKey: unknown[] | null = null;

    /** A voxelizer over `layout`. */
    constructor(private readonly device: GPUDevice, public readonly layout: VolumeLayout) {
        const fragment = GPUShaderStage.FRAGMENT;
        this.bindGroupLayout = device.createBindGroupLayout({
            label: 'VoxelGI/VoxelizeBGL',
            entries: [
                { binding: 100, visibility: fragment, buffer: { type: 'uniform' } },
                { binding: 101, visibility: fragment, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: VOXEL_DRAW_BYTES } },
                { binding: 102, visibility: fragment, buffer: { type: 'storage' } },
            ],
        });
        this.fragment = {
            module: device.createShaderModule({ label: 'VoxelGI/VoxelFragment', code: VOXEL_FRAGMENT_WGSL }),
            entryPoint: 'voxel_fragment',
        };

        // the three cameras: orthographic over the volume along x, y and z, one pixel per voxel
        const uniform = (label: string, size: number) =>
            device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const cameraLayout = device.createBindGroupLayout({ label: 'VoxelGI/CameraBGL', entries: cameraBindGroupLayoutEntries() });
        this.params = [];
        this.cameraGroups = axisViews(layout.origin, layout.voxelSize, layout.dims).map((axis) => {
            // view, projection, scene lights (unused) and temporal data (unused), as Camera binds them
            const buffers = [64, 64, LIGHT_UNIFORM_BYTES, CAMERA_TEMPORAL_BYTES].map((size) => uniform('VoxelGI/AxisCamera', size));
            device.queue.writeBuffer(buffers[0], 0, axis.view as Float32Array);
            device.queue.writeBuffer(buffers[1], 0, axis.projection as Float32Array);
            this.cameraBuffers.push(...buffers);
            const params = uniform('VoxelGI/VoxelizeParams', VOXELIZE_PARAMS_BYTES);
            device.queue.writeBuffer(params, 0, voxelizeParams(axis, layout.dims, [0, 0, 0], layout.dims, SURFACE_WORDS_PER_VOXEL, 1));
            this.params.push(params);
            return device.createBindGroup({
                label: 'VoxelGI/AxisCameraBG',
                layout: cameraLayout,
                entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            });
        });

        const side = Math.max(...layout.dims);
        this.target = device.createTexture({
            label: 'VoxelGI/VoxelizeTarget',
            size: [side, side],
            sampleCount: MeshVoxelizer.SAMPLE_COUNT,
            format: MeshVoxelizer.TARGET_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.targetView = this.target.createView();
        const alignment = device.limits.minUniformBufferOffsetAlignment;
        this.drawStride = Math.ceil(VOXEL_DRAW_BYTES / alignment) * alignment;
        this.draws = uniform('VoxelGI/Draws', this.drawStride * this.drawCapacity);
        this.staticSurfaces = this.surfaceBuffer('VoxelGI/StaticSurfaces');
        this.rebuildGroups();
    }

    private surfaceBuffer(label: string): GPUBuffer {
        return this.device.createBuffer({
            label,
            size: this.layout.voxelCount() * SURFACE_WORDS_PER_VOXEL * 4,
            // (COPY_SRC: readable in tests)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    private rebuildGroups(): void {
        const groups = (surfaces: GPUBuffer) => this.params.map((params) => this.device.createBindGroup({
            label: 'VoxelGI/VoxelizeBG',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 100, resource: { buffer: params } },
                { binding: 101, resource: { buffer: this.draws, offset: 0, size: VOXEL_DRAW_BYTES } },
                { binding: 102, resource: { buffer: surfaces } },
            ],
        }));
        this.staticGroups = groups(this.staticSurfaces);
        this.dynamicGroups = this._dynamicSurfaces ? groups(this._dynamicSurfaces) : [];
    }

    /** Samples of its passes' target. */
    public get sampleCount(): number { return MeshVoxelizer.SAMPLE_COUNT; }

    /** The dynamic renderables' voxels this frame (null until one is voxelized). */
    public get dynamicSurfaces(): GPUBuffer | null { return this._dynamicSurfaces; }

    /**
     * Voxelize the static renderables again next frame (they are otherwise redone only when one
     * of them changes its transform, visibility, geometry or surface).
     */
    public invalidate(): void {
        this.staticKey = null;
    }

    /** Upload the draws' constant surfaces (one slot each, in draw order), growing their buffer. */
    public writeDraws(surfaces: readonly GiSurface[]): void {
        if (surfaces.length === 0) return;
        if (surfaces.length > this.drawCapacity) {
            this.drawCapacity = 2 ** Math.ceil(Math.log2(surfaces.length));
            this.draws.destroy();
            this.draws = this.device.createBuffer({
                label: 'VoxelGI/Draws',
                size: this.drawStride * this.drawCapacity,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.rebuildGroups();
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

    /**
     * Whether the static surfaces must be voxelized again for `key` (a description of the static
     * renderables: see `SceneVoxelGi.encode`); remembers it.
     */
    public staticChanged(key: unknown[]): boolean {
        const old = this.staticKey;
        if (old && old.length === key.length && old.every((v, i) => v === key[i])) return false;
        this.staticKey = key;
        return true;
    }

    /** Make the dynamic buffer (the first frame a dynamic renderable is voxelized). */
    public ensureDynamic(): void {
        if (!this._dynamicSurfaces) {
            this._dynamicSurfaces = this.surfaceBuffer('VoxelGI/DynamicSurfaces');
            this.rebuildGroups();
        }
    }

    /** Record clearing `set`'s buffer. */
    public clear(encoder: GPUCommandEncoder, set: SurfaceSet): void {
        const buffer = set === SurfaceSet.Static ? this.staticSurfaces : this._dynamicSurfaces;
        if (buffer) encoder.clearBuffer(buffer);
    }

    /**
     * Begin the voxelization pass of `axis` (0 x, 1 y, 2 z): the axis' camera bound as group 1
     * and its viewport set. Bind group 3 per draw with `group(axis, set)` and `drawOffset`.
     */
    public beginPass(encoder: GPUCommandEncoder, axis: number): GPURenderPassEncoder {
        const label = ['VoxelGI/VoxelizeX', 'VoxelGI/VoxelizeY', 'VoxelGI/VoxelizeZ'][axis];
        const pass = encoder.beginRenderPass({
            label,
            colorAttachments: [{ view: this.targetView, clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'discard' }],
            timestampWrites: gpuPass(label),
        });
        const [dx, dy, dz] = this.layout.dims;
        const [w, h] = [[dz, dy], [dx, dz], [dx, dy]][axis];
        pass.setViewport(0, 0, w, h, 0, 1);
        pass.setBindGroup(1, this.cameraGroups[axis]);
        return pass;
    }

    /** Group 3 for `axis` into `set`. */
    public group(axis: number, set: SurfaceSet): GPUBindGroup {
        return set === SurfaceSet.Static ? this.staticGroups[axis] : this.dynamicGroups[axis];
    }

    /**
     * Record the voxelization of `set`'s buffer: cleared, then each of `draws` whose `dynamic`
     * matches the set drawn along the three axes with its `pipeline` (`Material.getVoxelPipeline`
     * for this voxelizer), its material's group 0 and its matrices at `meshOffset` of
     * `meshBindGroup` (group 2). `draws[k]`'s surface is draw slot `k` (`writeDraws`).
     */
    public encodeSet(
        encoder: GPUCommandEncoder,
        set: SurfaceSet,
        draws: readonly { renderable: Renderable, pipeline: GPURenderPipeline, meshOffset: number }[],
        meshBindGroup: GPUBindGroup,
    ): void {
        this.clear(encoder, set);
        const dynamic = set === SurfaceSet.Dynamic;
        if (!draws.some((d) => d.renderable.dynamic === dynamic)) return;
        for (let axis = 0; axis < 3; axis++) {
            const pass = this.beginPass(encoder, axis);
            draws.forEach(({ renderable: r, pipeline, meshOffset }, k) => {
                if (r.dynamic !== dynamic) return;
                const geometry = r.geometry;
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, r.material.getBindGroup(this.device));
                pass.setBindGroup(2, meshBindGroup, [meshOffset, meshOffset]);
                pass.setBindGroup(3, this.group(axis, set), [this.drawOffset(k)]);
                pass.setVertexBuffer(0, geometry.vertexBuffer!);
                pass.setIndexBuffer(geometry.indexBuffer!, geometry.indexFormat!);
                // every instance: no cull view is the voxelizer's
                drawGeometry(pass, geometry, null);
            });
            pass.end();
        }
    }

    /** Bytes of its surface buffers. */
    public memoryBytes(): number {
        return (this._dynamicSurfaces ? 2 : 1) * this.layout.voxelCount() * SURFACE_WORDS_PER_VOXEL * 4;
    }

    public destroy(): void {
        for (const b of [this.staticSurfaces, this._dynamicSurfaces, this.draws, ...this.params, ...this.cameraBuffers]) b?.destroy();
        this.target.destroy();
    }
}
