import { mat4 } from 'gl-matrix';
import { ComputeBuffer } from '../buffers/ComputeBuffer';
import { BufferBase } from '../buffers/BufferBase';
import type { Geometry } from '../buffers/Geometry';
import type { InstancedGeometry } from '../geometries/InstancedGeometry';
import { frustumPlanes } from './Frustum';
import instanceCullWgsl from '../../rust/kansei-core/src/shaders/instance_cull.wgsl?raw';
import lodFadeWgsl from '../../rust/kansei-core/src/shaders/lod_fade.wgsl?raw';

/**
 * The cull shader, shared with the Rust engine (`shaders/instance_cull.wgsl`): one thread per
 * instance, one dispatch for all of a renderable's views (y: the view). Group 0 is the
 * renderable's (`CullInstances`, the source, the compacted instances, the indirect draws), group 1
 * the frame's views.
 */
export const INSTANCE_CULL_WGSL: string = instanceCullWgsl;

/**
 * WGSL for materials of renderables culled with crossfades (`InstanceCulling.withCrossfade`):
 * `kansei_lod_fade_discard(fade, pixel, frame)`, whether to drop a pixel of an instance fading
 * between LODs. Rust: `culling::LOD_FADE_WGSL`.
 */
export const LOD_FADE_WGSL: string = lodFadeWgsl;

const NO_WORD = 0xffffffff;
/** Largest f32: a band to infinity (`lod_far.min(f32::MAX)` in Rust). */
const F32_MAX = 3.4028234663852886e38;

/**
 * Bytes of a view's indirect draw: `DrawIndexedIndirect`'s five words, then the instances culled
 * by the LOD band, the frustum and occlusion (counted when the renderer's culling stats are on).
 */
export const CULL_ARGS_BYTES = 32;
/** Bytes of `CullInstances` in instance_cull.wgsl. */
export const CULL_INSTANCES_BYTES = 192;
/** Bytes of `CullView` in instance_cull.wgsl. */
export const CULL_VIEW_BYTES = 256;

const FLAG_STATS = 1;
const FLAG_BOX = 2;
const FLAG_CASTS_SHADOW = 8;
const FLAG_VIEW = 32;
const FLAG_CASTERS_ONLY = 64;
const FLAG_LAYERED = 128;
const FLAG_REFLECTION = 256;
const FLAG_GI = 2048;
const FLAG_GI_SURFACE = 4096;
const FLAG_RT = 8192;
const FLAG_RT_SURFACE = 16384;

type Vec3 = [number, number, number];
/** A band of distances from the main camera, `[near, far)`. */
type LodRange = [number, number];

/**
 * A culled draw: the compacted instances (for the geometry's first instance buffer, from
 * `instancesOffset`) and the indirect draw, at `offset` in `args`.
 */
export interface CulledDraw {
    instances: GPUBuffer;
    instancesOffset: number;
    args: GPUBuffer;
    offset: number;
}

/**
 * A view the renderer culls for: its view-projection, whether it only draws shadow casters (a
 * shadow map, with `shadowLodRange`), whether it is a planar reflection (with
 * `reflectionLodRange`), voxel GI's (renderables with a GI surface, with `giLodRange`) or the ray
 * tracing grid's (with `rtLodRange`), the layers it draws if not all (a reflection's layer mask),
 * and how it scales the LOD distances (below 1 a view picks finer LODs than the camera would).
 */
export interface CullView {
    viewProj: ArrayLike<number>;
    castersOnly: boolean;
    reflection: boolean;
    gi: boolean;
    rt: boolean;
    layerMask: number | null;
    lodDistanceScale: number;
}

/** A view of every kind but `castersOnly`'s false, drawing all layers at the camera's LOD scale. */
export function cullView(viewProj: ArrayLike<number>, options: Partial<Omit<CullView, 'viewProj'>> = {}): CullView {
    return { viewProj, castersOnly: false, reflection: false, gi: false, rt: false, layerMask: null, lodDistanceScale: 1, ...options };
}

/**
 * Whether `view` draws a renderable that casts shadows or not, on `layers`, with a GI surface or
 * not, in the ray tracing grid or not (the cull shader's `drawnIn`).
 */
export function cullViewDraws(view: CullView, castsShadow: boolean, layers: number, giSurface: boolean = false, rtSurface: boolean = false): boolean {
    return (!view.castersOnly || castsShadow) && (!view.gi || giSurface) && (!view.rt || rtSurface)
        && (view.layerMask === null || (view.layerMask & layers) !== 0);
}

/**
 * Packs `view` into `out` (`CULL_VIEW_BYTES` at `byteOffset`) for the GPU: its frustum, the LOD
 * origin and its flags. `null` packs an unused view (zeros: the shader skips it).
 */
export function packCullView(out: ArrayBuffer, byteOffset: number, view: CullView | null, lodOrigin: ArrayLike<number>, stats: boolean): void {
    const f = new Float32Array(out, byteOffset, CULL_VIEW_BYTES / 4);
    const u = new Uint32Array(out, byteOffset, CULL_VIEW_BYTES / 4);
    f.fill(0);
    if (!view) return;
    let flags = FLAG_VIEW;
    if (view.castersOnly) flags |= FLAG_CASTERS_ONLY;
    if (view.layerMask !== null) flags |= FLAG_LAYERED;
    if (view.reflection) flags |= FLAG_REFLECTION;
    if (view.gi) flags |= FLAG_GI;
    if (view.rt) flags |= FLAG_RT;
    if (stats) flags |= FLAG_STATS;
    const planes = frustumPlanes(view.viewProj);
    for (let k = 0; k < 6; k++) f.set(planes[k], k * 4);
    // view and proj (occlusion only): identity
    for (let k = 0; k < 4; k++) {
        f[24 + k * 5] = 1;
        f[40 + k * 5] = 1;
    }
    f[56] = lodOrigin[0];
    f[57] = lodOrigin[1];
    f[58] = lodOrigin[2];
    u[59] = flags;
    // depth size (occlusion only)
    f[60] = 1;
    f[61] = 1;
    // a band [near, far) at distance x scale is the band [near, far) / scale at x
    f[62] = Math.max(view.lodDistanceScale, 1e-3);
    u[63] = view.layerMask === null ? 0xffffffff : view.layerMask >>> 0;
}

/**
 * Consecutive views culled by one dispatch: their compacted instances, a region of `capacity`
 * instances each in one buffer, and the bind group (with the chunk's parameters).
 */
interface Chunk {
    firstView: number;
    views: number;
    instances: GPUBuffer;
    bindGroup: GPUBindGroup;
}

/**
 * The renderable's culling state for every view: the instances' parameters (a copy per chunk),
 * written only when they change; the indirect draws, `CULL_ARGS_BYTES` apart in one buffer that a
 * frame resets with one clear; and the views' compacted instances, in chunks as large as a storage
 * binding allows (one, but for very many instances and views).
 */
interface Shared {
    params: GPUBuffer;
    paramsStride: number;
    written: Uint32Array | null;
    args: GPUBuffer;
    views: number;
    chunks: Chunk[];
    /** bytes per compacted instance the buffers were made for */
    outStride: number;
}

/**
 * Per-view GPU culling for an instanced renderable (set `Renderable.instanceCulling`). A port of
 * the Rust engine's `culling::InstanceCulling`, sharing its shader.
 *
 * `source` holds every instance, never culled: the per-instance vertex data of the geometry's
 * first instance buffer, `stride` bytes each. Every frame the renderer culls it on the GPU for
 * each view that draws the renderable: the main camera, and the directional shadow map's light
 * (`Renderer.enableShadows`), so things outside the picture still cast into it. An instance is
 * kept when its bounding sphere (centre at `centerOffset`, `radius` times the f32 at
 * `radiusScaleOffset` if set, both in the renderable's object space) is inside the view, and its
 * distance from the main camera is within `lodRange`. The survivors are compacted, a region per
 * view, and drawn indirectly; the geometry's own instance buffer only lends its vertex layout. One
 * dispatch culls all of a renderable's views. Point-light cube shadows, and shadow maps the caller
 * renders, draw every instance.
 *
 * LOD: one renderable per LOD mesh, all sharing `source`, each with its distance band. Bands are
 * measured from the main camera in every view, so shadows match what is on screen. Shadow maps
 * can have bands of their own (`withShadowLodRange`): a far LOD nearer there than on screen, say.
 * Keep each kind of view's bands of a mesh's LODs adjoining, as the camera's.
 *
 * Tighter bounds (`withBoundsShift`, `withBoundsBox`) cull more, in every view: a tree's sphere
 * round its base reaches a tree's height below the ground and to each side.
 *
 * ```ts
 * // 32-byte instances: position xyz + height, then yaw, ...; spheres of 0.6 x height
 * lod0.instanceCulling = new InstanceCulling(allTrees, treeCount, 32, 0, 0.6)
 *     .withRadiusScale(12)
 *     .withLodRange(0, 60);
 * ```
 */
export class InstanceCulling {
    /**
     * All instances: the `ComputeBuffer` the geometry reads them from (the same object, so one
     * GPU buffer), initialized on first use if the geometry has not been drawn yet; needs
     * `STORAGE` usage.
     */
    public readonly source: ComputeBuffer;
    /** Instances in `source` (at most the count it was created with grows the buffers). */
    public count: number;
    /** Bytes per instance, a multiple of 4. */
    public readonly stride: number;
    /** Byte offset of the instance's centre (3 x f32) within an instance. */
    public readonly centerOffset: number;
    /** Bounding radius, object space. */
    public radius: number;
    /** Byte offset of an f32 the bounds are multiplied by (a per-instance scale or height). */
    public radiusScaleOffset: number | null = null;
    /** Distances from the main camera at which the instances draw here: `[near, far)`. */
    public lodRange: LodRange = [0, Infinity];
    /** The band in shadow maps, if not `lodRange`. */
    public shadowLodRange: LodRange | null = null;
    /** The band in planar reflections, if not `lodRange`. */
    public reflectionLodRange: LodRange | null = null;
    /** The band in voxel GI's views, if not `lodRange`; an empty band (`near >= far`) is nowhere. */
    public giLodRange: LodRange | null = null;
    /** The band in the ray tracing grid's view, if not `lodRange`; an empty band is nowhere. */
    public rtLodRange: LodRange | null = null;
    /**
     * Object-space offset of the bounds' centre from the instance's centre, times the per-instance
     * scale: where the sphere (or box) sits. It moves the centre the LOD distance is measured from
     * too. Instances' own rotations are not applied: use it along an axis they turn about.
     */
    public boundsShift: Vec3 = [0, 0, 0];
    /**
     * Object-space half extents of a box (times the per-instance scale, round the shifted centre)
     * to cull with instead of the sphere. Instances' own rotations are not applied: make it wide
     * enough for them (for instances turned about y, equal x and z extents of the widest reach).
     */
    public boundsBox: Vec3 | null = null;
    /**
     * Width of the crossfades at the LOD bands' edges, in LOD distance (0: none); see
     * `withCrossfade`. Turning them on or off changes the compacted instances' layout.
     */
    public crossfade: number = 0;

    private capacity: number;
    private shared: Shared | null = null;
    private readonly scratch = new ArrayBuffer(CULL_INSTANCES_BYTES);

    /**
     * Culling for `count` instances of `stride` bytes in `source`, the `ComputeBuffer` the
     * geometry reads them from (or a GPU buffer created elsewhere, a simulation's), each a sphere
     * of `radius` around the 3 floats at `centerOffset`.
     */
    constructor(source: ComputeBuffer | GPUBuffer, count: number, stride: number, centerOffset: number, radius: number) {
        if (stride % 4 !== 0 || centerOffset % 4 !== 0 || centerOffset + 12 > stride) {
            throw new Error('InstanceCulling: the instance layout must be 4-byte words, with the centre inside');
        }
        this.source = source instanceof ComputeBuffer ? source : ComputeBuffer.fromExternal(source, BufferBase.BUFFER_TYPE_STORAGE);
        this.count = count;
        this.stride = stride;
        this.centerOffset = centerOffset;
        this.radius = radius;
        this.capacity = count;
    }

    /**
     * Culling for instances of one vec4 each, position in xyz and a scale in w that multiplies
     * `radius`: `new InstanceCulling(source, count, 16, 0, radius).withRadiusScale(12)`.
     */
    static forVec4Instances(source: ComputeBuffer | GPUBuffer, count: number, radius: number): InstanceCulling {
        return new InstanceCulling(source, count, 16, 0, radius).withRadiusScale(12);
    }

    withRadiusScale(offset: number): this {
        if (offset % 4 !== 0 || offset + 4 > this.stride) throw new Error('InstanceCulling: the radius scale must be a word inside the instance');
        this.radiusScaleOffset = offset;
        return this;
    }

    withLodRange(near: number, far: number): this {
        this.lodRange = [near, far];
        return this;
    }

    /** See `shadowLodRange`. */
    withShadowLodRange(near: number, far: number): this {
        this.shadowLodRange = [near, far];
        return this;
    }

    /** See `reflectionLodRange`. */
    withReflectionLodRange(near: number, far: number): this {
        this.reflectionLodRange = [near, far];
        return this;
    }

    /** See `giLodRange`. */
    withGiLodRange(near: number, far: number): this {
        this.giLodRange = [near, far];
        return this;
    }

    /** See `rtLodRange`. */
    withRtLodRange(near: number, far: number): this {
        this.rtLodRange = [near, far];
        return this;
    }

    /** See `boundsShift`. */
    withBoundsShift(shift: Vec3): this {
        this.boundsShift = [...shift];
        return this;
    }

    /** See `boundsBox`. */
    withBoundsBox(halfExtents: Vec3): this {
        this.boundsBox = [...halfExtents];
        return this;
    }

    /**
     * Dithered crossfades between LODs, over `width` of LOD distance (metres, times the view's
     * LOD distance scale): each band's edges widen by `width / 2`, and there the instances draw
     * in both LODs, each keeping a complementary share of the pixels, so an instance moving
     * across a band's edge dissolves from one LOD into the other instead of snapping, in every
     * view (the camera, the shadow map), by that view's band. A band from 0 has no near
     * crossfade, and one to infinity no far one: an impostor LOD fades in from the meshes. Give
     * every LOD of a set the same width, narrower than its bands.
     *
     * Each compacted instance is then followed by its fade, an f32: declare the geometry's
     * instance buffer `stride + 4` bytes wide with the fade as an attribute at offset `stride`
     * (`ComputeBuffer.withVertexLayout` over the source), pass it to the fragment shader, and
     * drop pixels with `LOD_FADE_WGSL`'s `kansei_lod_fade_discard` (in the material's
     * `shadowFragmentEntry` too, for the shadows to dissolve). An impostor baked from such a LOD
     * draws it with that layout too: end `ImpostorOptions.instance` with a fade of 1.
     *
     * The fade depends only on the distance, so there is no switch to flip back and forth: an
     * instance moving to and fro across a band's edge dissolves to and fro by as much as it
     * moves, and one held there stays part dissolved.
     */
    withCrossfade(width: number): this {
        this.crossfade = Math.max(width, 0);
        return this;
    }

    /** Bytes of a compacted instance: the source's, and with crossfades its fade. */
    get culledStride(): number {
        return this.stride + (this.crossfade > 0 ? 4 : 0);
    }

    /** The instances a dispatch tests. */
    get tested(): number {
        return Math.min(this.count, this.capacity);
    }

    /** The compacted instances and indirect draw of `view` (0 is the main camera), once culled. */
    view(view: number): CulledDraw | null {
        const shared = this.shared;
        if (!shared) return null;
        const chunk = shared.chunks.find((c) => view >= c.firstView && view < c.firstView + c.views);
        if (!chunk) return null;
        return {
            instances: chunk.instances,
            instancesOffset: (view - chunk.firstView) * this.regionBytes(),
            args: shared.args,
            offset: view * CULL_ARGS_BYTES,
        };
    }

    /** Bytes of a view's region of compacted instances. */
    private regionBytes(): number {
        return Math.max(this.capacity, 1) * this.culledStride;
    }

    private instancesBuffer(device: GPUDevice, views: number): GPUBuffer {
        // (COPY_SRC: readable for debugging)
        return device.createBuffer({
            label: 'InstanceCulling/Instances',
            size: views * this.regionBytes(),
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
    }

    /** A bind group: chunk `chunk`'s parameters, the source, compacted instances `instances`, every draw. */
    private bindGroup(device: GPUDevice, layout: GPUBindGroupLayout, chunk: number, instances: GPUBuffer): GPUBindGroup {
        const shared = this.shared!;
        return device.createBindGroup({
            label: 'InstanceCulling/BG',
            layout,
            entries: [
                { binding: 0, resource: { buffer: shared.params, offset: chunk * shared.paramsStride, size: CULL_INSTANCES_BYTES } },
                { binding: 1, resource: { buffer: this.source.gpuBuffer! } },
                { binding: 2, resource: { buffer: instances } },
                { binding: 3, resource: { buffer: shared.args } },
            ],
        });
    }

    /**
     * Make sure there are GPU slots for `count` views; true if buffers were (re)created (render
     * bundles that recorded the old ones are stale).
     */
    ensureViews(device: GPUDevice, layout: GPUBindGroupLayout, count: number): boolean {
        const limits = device.limits;
        return this.ensureViewsWithin(device, layout, count, Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize));
    }

    /** `ensureViews`, with at most `maxChunkBytes` of compacted instances per chunk. */
    ensureViewsWithin(device: GPUDevice, layout: GPUBindGroupLayout, count: number, maxChunkBytes: number): boolean {
        // the source, unless the geometry sharing it has been drawn already
        if (!this.source.initialized) this.source.initialize(device);
        const buffer = this.source.gpuBuffer;
        if (!buffer || (buffer.usage & GPUBufferUsage.STORAGE) === 0) {
            throw new Error('InstanceCulling: the source instance buffer needs STORAGE usage (and VERTEX, for the geometry to draw it)');
        }
        const shared = this.shared;
        if (this.count <= this.capacity && shared && shared.views >= count && shared.outStride === this.culledStride) {
            return false;
        }
        // recreate everything: the draws hold a slot per view
        this.destroy();
        this.capacity = Math.max(this.capacity, this.count);
        count = Math.max(count, shared ? shared.views : 0, 1);
        const perChunk = Math.min(Math.max(Math.floor(maxChunkBytes / this.regionBytes()), 1), count);
        const chunkCount = Math.ceil(count / perChunk);
        const align = device.limits.minUniformBufferOffsetAlignment;
        const paramsStride = Math.ceil(CULL_INSTANCES_BYTES / align) * align;
        this.shared = {
            params: device.createBuffer({
                label: 'InstanceCulling/Params',
                size: chunkCount * paramsStride,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),
            paramsStride,
            written: null,
            // (COPY_SRC: the stats read them back)
            args: device.createBuffer({
                label: 'InstanceCulling/Args',
                size: count * CULL_ARGS_BYTES,
                usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            }),
            views: count,
            chunks: [],
            outStride: this.culledStride,
        };
        for (let k = 0; k < chunkCount; k++) {
            const firstView = k * perChunk;
            const views = Math.min(perChunk, count - firstView);
            const instances = this.instancesBuffer(device, views);
            this.shared.chunks.push({ firstView, views, instances, bindGroup: this.bindGroup(device, layout, k, instances) });
        }
        return true;
    }

    /** The chunks the buffers are split in (one, but for very many instances and views). */
    get chunkCount(): number {
        return this.shared?.chunks.length ?? 0;
    }

    /**
     * Start a frame, before its cull pass (after `ensureViews`): write the instances' parameters
     * for a renderable with `world` matrix and `indexCount` indices, drawn in shadow maps if
     * `castsShadow`, on `layers`, if they changed, and clear every view's draw (the cull sets the
     * index count of those it culls into).
     */
    beginFrame(queue: GPUQueue, encoder: GPUCommandEncoder, world: mat4 | Float32Array, indexCount: number, castsShadow: boolean, layers: number, giSurface: boolean = false, rtSurface: boolean = false): void {
        const shared = this.shared;
        if (!shared) throw new Error('InstanceCulling: ensureViews first');
        const f = new Float32Array(this.scratch);
        const u = new Uint32Array(this.scratch);
        const scale = (c: number) => Math.hypot(world[c], world[c + 1], world[c + 2]);
        const band = (range: LodRange, at: number) => {
            f[at] = range[0];
            f[at + 1] = Math.min(range[1], F32_MAX);
        };
        // an empty band is nowhere, crossfades and all
        const kindBand = (range: LodRange | null): LodRange =>
            range && range[0] >= range[1] ? [F32_MAX, F32_MAX] : range ?? this.lodRange;
        let flags = 0;
        if (this.boundsBox) flags |= FLAG_BOX;
        if (castsShadow) flags |= FLAG_CASTS_SHADOW;
        if (giSurface) flags |= FLAG_GI_SURFACE;
        if (rtSurface) flags |= FLAG_RT_SURFACE;
        f.set(world, 0);
        f.set(this.boundsShift, 16);
        f[19] = this.lodRange[0];
        f.set(this.boundsBox ?? [0, 0, 0], 20);
        f[23] = Math.min(this.lodRange[1], F32_MAX);
        f[24] = this.radius;
        f[25] = Math.max(scale(0), scale(4), scale(8));
        u[26] = this.tested;
        u[27] = this.stride / 4;
        u[28] = this.centerOffset / 4;
        u[29] = this.radiusScaleOffset === null ? NO_WORD : this.radiusScaleOffset / 4;
        u[30] = flags;
        u[31] = indexCount;
        u[32] = 0; // firstView, per chunk below
        u[33] = this.capacity;
        u[34] = 0; // lateSlot (occlusion)
        u[35] = layers >>> 0;
        band(this.shadowLodRange ?? this.lodRange, 36);
        band(this.reflectionLodRange ?? this.lodRange, 38);
        u[40] = 0; // occlusionView
        f[41] = this.crossfade;
        band(kindBand(this.giLodRange), 42);
        band(kindBand(this.rtLodRange), 44);
        f[46] = 0;
        f[47] = 0;
        if (!shared.written || !sameWords(shared.written, u)) {
            // a copy per chunk, each with its first view
            const words = shared.paramsStride / 4;
            const bytes = new Uint32Array(shared.chunks.length * words);
            shared.chunks.forEach((chunk, k) => {
                bytes.set(u, k * words);
                bytes[k * words + 32] = chunk.firstView;
            });
            queue.writeBuffer(shared.params, 0, bytes);
            shared.written = u.slice();
        }
        encoder.clearBuffer(shared.args);
    }

    private workgroups(): number {
        return Math.max(Math.ceil(this.tested / 64), 1);
    }

    /**
     * Cull into every view the renderable is drawn in, a dispatch per chunk (with the cull
     * pipeline and the views' group 1 set).
     */
    dispatch(pass: GPUComputePassEncoder): void {
        const shared = this.shared;
        if (!shared) throw new Error('InstanceCulling: ensureViews first');
        for (const chunk of shared.chunks) {
            pass.setBindGroup(0, chunk.bindGroup);
            pass.dispatchWorkgroups(this.workgroups(), chunk.views, 1);
        }
    }

    /** Frees the GPU buffers (the source stays its owner's); the next frame makes new ones. */
    destroy(): void {
        if (!this.shared) return;
        this.shared.params.destroy();
        this.shared.args.destroy();
        for (const chunk of this.shared.chunks) chunk.instances.destroy();
        this.shared = null;
    }
}

function sameWords(a: Uint32Array, b: Uint32Array): boolean {
    for (let i = 0; i < b.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

/**
 * The cull compute pipeline, shared by every renderable, and the frame's views, in one buffer
 * (`setViews`, `viewBindGroup`). Raw pipelines with explicit layouts: `Compute` binds group 0
 * only.
 */
export class CullPipeline {
    readonly pipeline: GPUComputePipeline;
    /** params, source, compacted instances, indirect draws */
    readonly layout: GPUBindGroupLayout;
    /** group 1: the views */
    private readonly viewLayout: GPUBindGroupLayout;
    private views: { buffer: GPUBuffer; bindGroup: GPUBindGroup; capacity: number } | null = null;

    constructor(private readonly device: GPUDevice) {
        const entry = (binding: number, buffer: GPUBufferBindingLayout): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer });
        this.layout = device.createBindGroupLayout({
            label: 'InstanceCulling/BGL',
            entries: [
                entry(0, { type: 'uniform' }),
                entry(1, { type: 'read-only-storage' }),
                entry(2, { type: 'storage' }),
                entry(3, { type: 'storage' }),
            ],
        });
        this.viewLayout = device.createBindGroupLayout({ label: 'InstanceCulling/ViewBGL', entries: [entry(0, { type: 'read-only-storage' })] });
        const module = device.createShaderModule({ label: 'InstanceCulling', code: INSTANCE_CULL_WGSL });
        this.pipeline = device.createComputePipeline({
            label: 'InstanceCulling/main',
            layout: device.createPipelineLayout({ label: 'InstanceCulling', bindGroupLayouts: [this.layout, this.viewLayout] }),
            compute: { module, entryPoint: 'main' },
        });
    }

    /** Upload the frame's views (`packCullView` each, `CULL_VIEW_BYTES` apart), in one write. */
    setViews(views: ArrayBuffer, count: number): void {
        if (!this.views || this.views.capacity < count) {
            this.views?.buffer.destroy();
            let capacity = 8;
            while (capacity < count) capacity *= 2;
            const buffer = this.device.createBuffer({
                label: 'InstanceCulling/Views',
                size: capacity * CULL_VIEW_BYTES,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            const bindGroup = this.device.createBindGroup({
                label: 'InstanceCulling/Views',
                layout: this.viewLayout,
                entries: [{ binding: 0, resource: { buffer } }],
            });
            this.views = { buffer, bindGroup, capacity };
        }
        this.device.queue.writeBuffer(this.views.buffer, 0, views, 0, count * CULL_VIEW_BYTES);
    }

    /** Group 1 of the cull pipeline (after `setViews`). */
    get viewBindGroup(): GPUBindGroup {
        if (!this.views) throw new Error('CullPipeline: setViews first');
        return this.views.bindGroup;
    }
}

/**
 * Binds `geometry`'s instance buffers and issues its draw: with `culled`, its compacted instances
 * in place of the first instance buffer and the indirect draw; else every instance (or the
 * geometry's own indirect draw). The vertex and index buffers are the caller's to set.
 */
export function drawGeometry(encoder: GPURenderPassEncoder | GPURenderBundleEncoder, geometry: Geometry, culled: CulledDraw | null): void {
    if (geometry.isInstancedGeometry) {
        const geo = geometry as InstancedGeometry;
        geo.extraBuffers.forEach((buffer, i) => {
            if (culled && i === 0) encoder.setVertexBuffer(1, culled.instances, culled.instancesOffset);
            else encoder.setVertexBuffer(i + 1, buffer.resource.buffer);
        });
        if (culled) encoder.drawIndexedIndirect(culled.args, culled.offset);
        else encoder.drawIndexed(geo.vertexCount, geo.instanceCount, 0, 0, 0);
    } else if (geometry.indirectArgsBuffer) {
        encoder.drawIndexedIndirect(geometry.indirectArgsBuffer, 0);
    } else {
        encoder.drawIndexed(geometry.vertexCount);
    }
}
