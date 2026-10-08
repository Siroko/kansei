import { mat4, vec3 } from 'gl-matrix';
import type { Camera } from '../cameras/Camera';
import type { DepthBias } from '../materials/Material';
import { gpuPass } from '../profiling/Profiler';
import { CAMERA_TEMPORAL_BYTES, LIGHT_UNIFORM_BYTES, cameraBindGroupLayoutEntries } from '../renderers/SharedLayouts';
import buildWgsl from '../../rust/kansei-core/src/shaders/sky_occlusion_build.wgsl?raw';
import skyOcclusionWgsl from '../../rust/kansei-core/src/shaders/sky_occlusion.wgsl?raw';

/**
 * WGSL for materials dimmed by the renderer's sky occlusion: `skyVisibility(volume, sampler,
 * params, worldPos)` and `SkyOcclusionParams`. Bind `SkyOcclusion.volume` (a filterable
 * `texture_3d<f32>`), a linear clamping sampler and `SkyOcclusion.params` (uniform). Rust:
 * `shadows::SKY_OCCLUSION_WGSL`.
 */
export const SKY_OCCLUSION_WGSL: string = skyOcclusionWgsl;

/** Options of `Renderer.enableSkyOcclusion` (Rust `SkyOcclusionOptions`). */
export interface SkyOcclusionOptions {
    /** Side of the square area around the camera it covers, metres. */
    extentM?: number;
    /** Texels per side of the depth map the canopy is seen in from above. */
    resolution?: number;
    /** Voxels of the visibility volume: per side, and in height. */
    volumeSize?: [number, number];
    /** World heights the volume spans; the ground and the canopy should lie inside. */
    minHeightM?: number;
    maxHeightM?: number;
    /** How far the camera may move before the map is rebuilt around it, as a share of the extent. */
    recenter?: number;
    /** How much a ray is dimmed per metre through canopy that covers its whole footprint. */
    canopyExtinction?: number;
    /**
     * Frames the volume's build is spread over, after the top-down pass (a share of it on each).
     * Materials read the previous volume until the new one is done.
     */
    frames?: number;
    /**
     * Tiles per side the top-down pass is split into, one drawn per frame, each culled to its
     * part of the view (1: the whole map in one frame). A rebuild then takes `depthTiles` squared
     * frames of the top-down pass, one for the pyramid and `frames` for the volume, and no frame
     * carries the whole top-down pass.
     */
    depthTiles?: number;
    /**
     * Scales the camera distance `InstanceCulling` picks LOD bands by, for the top-down view:
     * above 1 it draws coarser LODs, whose detail the map's texels rarely resolve, but it also
     * drops instances beyond a last LOD band that ends at a finite distance sooner.
     */
    lodDistanceScale?: number;
    /**
     * Scales the cluster LOD budget (`Renderable.clusters`) of the top-down view: above 1 it
     * draws coarser cuts than the camera. 1 by default.
     */
    lodErrorScale?: number;
    /**
     * The layers (`Renderable.layers`) whose shadow casters occlude the sky; all by default.
     * Leave solid ground out (put the vegetation on a layer of its own): the volume counts what
     * is under a top as inside it, so where it is interpolated across the ground the voxels below
     * it darken the ground's surface.
     */
    layerMask?: number;
}

/** Rust `SkyOcclusionOptions::default()`. */
const DEFAULTS: Required<SkyOcclusionOptions> = {
    extentM: 160,
    resolution: 1024,
    volumeSize: [128, 16],
    minHeightM: -20,
    maxHeightM: 60,
    recenter: 0.125,
    canopyExtinction: 0.3,
    frames: 4,
    depthTiles: 2,
    lodDistanceScale: 1,
    lodErrorScale: 1,
    layerMask: 0xffffffff,
};

/** Bytes of the WGSL `SkyOcclusionParams` and `OcclusionBuild`. */
const PARAMS_BYTES = 32;
const BUILD_BYTES = 48;
/** Bytes per slot of the build's parameters (WebGPU's uniform offset alignment). */
const SLOT = 256;
const NEAR = 1;

/** The smallest power of two at or above `n`. */
function nextPowerOfTwo(n: number): number {
    return 2 ** Math.ceil(Math.log2(Math.max(n, 1)));
}

/** A projection that maps the NDC rectangle x0..x1, y0..y1 to the whole view (Rust `reflections::crop`). */
function crop(x0: number, x1: number, y0: number, y1: number): mat4 {
    const sx = 2 / Math.max(x1 - x0, 1e-6), sy = 2 / Math.max(y1 - y0, 1e-6);
    return mat4.fromValues(sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 0, -(x0 + x1) * sx * 0.5, -(y0 + y1) * sy * 0.5, 0, 1);
}

/**
 * Sky occlusion around the camera (`Renderer.enableSkyOcclusion`): how much of the sky each point
 * sees past the canopy, as Lumen occludes Unreal's sky light under trees. Rust:
 * `shadows::SkyOcclusion`, on its WGSL.
 *
 * The renderer draws the shadow casters on `SkyOcclusionOptions.layerMask`'s layers (alpha-tested
 * foliage included, through its shadow fragment) into a depth map from straight above, culled on
 * the GPU like a shadow view. From it the build makes a pyramid of the canopy's cover and height,
 * then a low-resolution volume of sky visibility: from each voxel, 24 cosine-weighted directions
 * cone-traced through the canopy. It rebuilds only when the camera has moved far enough
 * (`recenter`), or on `refresh`, spread over frames: the top-down pass a tile a frame
 * (`depthTiles`), the pyramid, the volume a slab a frame (`frames`); the scene is taken to stand
 * still in between. Materials read it with `SKY_OCCLUSION_WGSL`'s `skyVisibility` and dim their sky
 * ambient light by it.
 *
 * The canopy is seen as a height field: its top, and how much of each area it covers. What is
 * under a crown is taken to be inside it, so rays that would slip beneath a neighbouring crown
 * count as dimmed.
 */
export class SkyOcclusion {
    /** The depth format and bias of the top-down pass: the cascades' (Rust `CascadedShadowMap`). */
    static readonly FORMAT: GPUTextureFormat = 'depth32float';
    static readonly DEPTH_BIAS: DepthBias = { constant: 0, slopeScale: 2, clamp: 0 };

    /**
     * `resolution`, `volumeSize` and `frames` are fixed when it is created; the others apply from
     * the next rebuild.
     */
    readonly options: Required<SkyOcclusionOptions>;
    /** The visibility volume (rgba8unorm, r the visibility), for `skyVisibility`. */
    readonly volume: GPUTextureView;
    /** The volume's texture (for `Texture.fromView` in a material). */
    readonly volumeTexture: GPUTexture;
    /** Its placement (uniform `SkyOcclusionParams`); off (visibility 1) until the first build. */
    readonly params: GPUBuffer;

    private _device: GPUDevice;
    /** The volume being built, copied into `volume` once complete. */
    private _back: GPUTexture;
    private _backView: GPUTextureView;
    private _depth: GPUTexture;
    private _depthView: GPUTextureView;
    private _pyramid: GPUTexture;
    private _pyramidViews: GPUTextureView[];
    private _pyramidAll: GPUTextureView;
    /**
     * The build's parameters, a slot per pyramid level then one per slab of the volume, written at
     * once when a rebuild starts.
     */
    private _buildParams: GPUBuffer;
    /** Layers of the volume built per frame. */
    private _slab: number;
    private _sampler: GPUSampler;
    private _topPipeline: GPUComputePipeline;
    private _downPipeline: GPUComputePipeline;
    private _volumePipeline: GPUComputePipeline;
    // Bind groups that never change: the top level's, each pyramid level's from the one below and
    // each slab's.
    private _topBG: GPUBindGroup;
    private _downBGs: GPUBindGroup[];
    private _volumeBGs: GPUBindGroup[];
    /** The top-down view as a camera group (view, projection, unused lights and temporal data). */
    private _cameraBuffers: GPUBuffer[];
    private _cameraBG: GPUBindGroup;
    private _view = mat4.create();
    private _projection = mat4.create();
    private _viewProj = mat4.create();
    private _cullViewProj = mat4.create();
    /**
     * Where the map was last built; while it is being rebuilt, the tile of the top-down pass due
     * this frame, whether the pyramid (and the volume's first slab) is, and the volume's next layer.
     */
    private _center: [number, number] | null = null;
    private _tile: number | null = null;
    private _pyramidDue = false;
    private _nextLayer: number | null = null;

    constructor(device: GPUDevice, options: SkyOcclusionOptions = {}) {
        this._device = device;
        const o = this.options = { ...DEFAULTS, ...options };
        const res = nextPowerOfTwo(Math.min(Math.max(o.resolution, 16), 8192));
        this._depth = device.createTexture({
            label: 'SkyOcclusion/Depth',
            size: [res, res],
            format: SkyOcclusion.FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this._depthView = this._depth.createView();
        const levels = Math.log2(res) + 1;
        this._pyramid = device.createTexture({
            label: 'SkyOcclusion/Pyramid',
            size: [res, res],
            format: 'rgba16float',
            mipLevelCount: levels,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this._pyramidViews = Array.from({ length: levels }, (_, l) => this._pyramid.createView({ baseMipLevel: l, mipLevelCount: 1 }));
        this._pyramidAll = this._pyramid.createView();
        const side = Math.max(o.volumeSize[0], 2), height = Math.max(o.volumeSize[1], 2);
        const volumeSize = [side, height, side];
        this.volumeTexture = device.createTexture({
            label: 'SkyOcclusion/Volume',
            size: volumeSize,
            dimension: '3d',
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
        });
        this.volume = this.volumeTexture.createView();
        this._back = device.createTexture({
            label: 'SkyOcclusion/VolumeBuild',
            size: volumeSize,
            dimension: '3d',
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this._backView = this._back.createView();
        const uniform = (label: string, size: number) =>
            device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.params = uniform('SkyOcclusion/Params', PARAMS_BYTES);
        // a slot per dispatch's parameters: one rewritten per dispatch would hold only the last
        // write when the passes run
        this._slab = Math.ceil(height / Math.min(Math.max(o.frames, 1), height));
        const slabs = Math.ceil(height / this._slab);
        this._buildParams = uniform('SkyOcclusion/Build', (levels + slabs) * SLOT);

        const compute = GPUShaderStage.COMPUTE;
        const uniformEntry: GPUBindGroupLayoutEntry = { binding: 0, visibility: compute, buffer: { type: 'uniform' } };
        const storageEntry = (binding: number, format: GPUTextureFormat, viewDimension: GPUTextureViewDimension): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: compute, storageTexture: { access: 'write-only', format, viewDimension } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });
        const topBGL = bgl('SkyOcclusion/TopBGL', [
            uniformEntry,
            { binding: 1, visibility: compute, texture: { sampleType: 'depth' } },
            storageEntry(3, 'rgba16float', '2d'),
        ]);
        const downBGL = bgl('SkyOcclusion/DownBGL', [
            uniformEntry,
            { binding: 2, visibility: compute, texture: { sampleType: 'unfilterable-float' } },
            storageEntry(3, 'rgba16float', '2d'),
        ]);
        const volumeBGL = bgl('SkyOcclusion/VolumeBGL', [
            uniformEntry,
            { binding: 4, visibility: compute, texture: { sampleType: 'float' } },
            { binding: 5, visibility: compute, sampler: { type: 'filtering' } },
            storageEntry(6, 'rgba8unorm', '3d'),
        ]);
        const module = device.createShaderModule({ label: 'SkyOcclusion/Build', code: buildWgsl });
        const pipeline = (label: string, entryPoint: string, layout: GPUBindGroupLayout) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module, entryPoint },
        });
        this._topPipeline = pipeline('SkyOcclusion/Top', 'top', topBGL);
        this._downPipeline = pipeline('SkyOcclusion/Down', 'down', downBGL);
        this._volumePipeline = pipeline('SkyOcclusion/Volume', 'volume', volumeBGL);
        this._sampler = device.createSampler({ label: 'SkyOcclusion/Sampler', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });

        const slot = (k: number): GPUBindingResource => ({ buffer: this._buildParams, offset: k * SLOT, size: BUILD_BYTES });
        const group = (layout: GPUBindGroupLayout, entries: [number, GPUBindingResource][]) => device.createBindGroup({
            label: 'SkyOcclusion/BG',
            layout,
            entries: entries.map(([binding, resource]) => ({ binding, resource })),
        });
        this._topBG = group(topBGL, [[0, slot(0)], [1, this._depthView], [3, this._pyramidViews[0]]]);
        this._downBGs = this._pyramidViews.slice(1).map((dst, i) =>
            group(downBGL, [[0, slot(i + 1)], [2, this._pyramidViews[i]], [3, dst]]));
        this._volumeBGs = Array.from({ length: slabs }, (_, s) =>
            group(volumeBGL, [[0, slot(levels + s)], [4, this._pyramidAll], [5, this._sampler], [6, this._backView]]));

        // view, projection, scene lights (unused) and temporal data (unused), as Camera binds them
        this._cameraBuffers = [64, 64, LIGHT_UNIFORM_BYTES, CAMERA_TEMPORAL_BYTES].map((size, i) => device.createBuffer({
            label: `SkyOcclusion/Camera${i}`,
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }));
        this._cameraBG = device.createBindGroup({
            label: 'SkyOcclusion/Camera BG',
            layout: device.createBindGroupLayout({ label: 'SkyOcclusion/Camera BGL', entries: cameraBindGroupLayoutEntries() }),
            entries: this._cameraBuffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
    }

    /** Rebuild on the next frame (the scene under the map changed). */
    refresh(): void {
        this._center = null;
    }

    private get _eyeY(): number {
        return this.options.maxHeightM + NEAR;
    }

    private get _far(): number {
        return this.options.maxHeightM - this.options.minHeightM + NEAR;
    }

    private get _resolution(): number {
        return this._pyramid.width;
    }

    /**
     * Before culling: whether the camera left the map's middle, and if so the top-down view of the
     * area around it (snapped to whole texels).
     */
    update(camera: Camera): void {
        const eye = camera.inverseViewMatrix.internalMat4;
        const hx = eye[12], hz = eye[14];
        const o = this.options;
        const c = this._center;
        if (c && Math.max(Math.abs(hx - c[0]), Math.abs(hz - c[1])) <= o.extentM * o.recenter) return;
        this._tile = 0;
        this._pyramidDue = false;
        this._nextLayer = null;
        const texel = o.extentM / this._resolution;
        const center: [number, number] = [Math.round(hx / texel) * texel, Math.round(hz / texel) * texel];
        this._center = center;
        const from = vec3.fromValues(center[0], this._eyeY, center[1]);
        mat4.lookAt(this._view, from, vec3.fromValues(center[0], this._eyeY - 1, center[1]), vec3.fromValues(0, 0, -1));
        const half = o.extentM * 0.5;
        mat4.orthoZO(this._projection, -half, half, -half, half, NEAR, this._far);
        mat4.multiply(this._viewProj, this._projection, this._view);
        const queue = this._device.queue;
        queue.writeBuffer(this._cameraBuffers[0], 0, this._view as Float32Array);
        queue.writeBuffer(this._cameraBuffers[1], 0, this._projection as Float32Array);
    }

    private get _tiles(): number {
        return Math.min(Math.max(this.options.depthTiles, 1), this._resolution);
    }

    /** Tile `tile`'s texels of the depth map: x from, x to, y from, y to (rows go down the map). */
    private _tileTexels(tile: number): [number, number, number, number] {
        const n = this._tiles, res = this._resolution;
        const x = tile % n, y = Math.floor(tile / n);
        return [Math.floor(x * res / n), Math.floor((x + 1) * res / n), Math.floor(y * res / n), Math.floor((y + 1) * res / n)];
    }

    /**
     * The top-down view to cull for this frame, while a tile of it is due: that tile's part of the
     * view (the tile's texels in normalized device coordinates, y up).
     */
    cullView(): Float32Array | null {
        if (this._tile === null) return null;
        const res = this._resolution;
        const [x0, x1, y0, y1] = this._tileTexels(this._tile).map((t) => t / res * 2 - 1);
        return mat4.multiply(this._cullViewProj, crop(x0, x1, -y1, -y0), this._viewProj) as Float32Array;
    }

    /** The top-down view's view matrix (its camera's, as `update` placed it). */
    get viewMatrix(): Float32Array {
        return this._view as Float32Array;
    }

    /** The top-down view's (orthographic) projection. */
    get projectionMatrix(): Float32Array {
        return this._projection as Float32Array;
    }

    /** Texels per side of its depth map. */
    get resolution(): number {
        return this._resolution;
    }

    /** Whether a tile of the top-down pass is due this frame. */
    get pending(): boolean {
        return this._tile !== null;
    }

    /** Whether this frame's tile is the first of a rebuild (its pass clears the map). */
    get firstTile(): boolean {
        return this._tile === 0;
    }

    /** This frame's tile of the depth map (x, y, width, height in texels), while one is due. */
    tileScissor(): [number, number, number, number] | null {
        if (this._tile === null) return null;
        const [x0, x1, y0, y1] = this._tileTexels(this._tile);
        return [x0, y0, x1 - x0, y1 - y0];
    }

    /** Whether `build` has work this frame: a tile of the top-down pass, the pyramid, or a slab of the volume. */
    get building(): boolean {
        return this._tile !== null || this._pyramidDue || this._nextLayer !== null;
    }

    /** The depth map the renderer draws the top-down pass into. */
    get depthView(): GPUTextureView {
        return this._depthView;
    }

    /** The top-down view as the depth pipelines' camera group (group 1). */
    get cameraBindGroup(): GPUBindGroup {
        return this._cameraBG;
    }

    /**
     * After a tile of the top-down pass, on to the next (nothing else that frame); on the frame
     * after the last, the pyramid and the volume's first slab; on the frames after it, the
     * volume's next slabs. Once the volume is complete, it replaces the one materials read, with
     * the parameters that place it (written before this frame's submission, so its materials read
     * the new volume).
     */
    build(encoder: GPUCommandEncoder): void {
        const center = this._center;
        if (!center) return;
        if (this._tile !== null) {
            const last = this._tiles * this._tiles - 1;
            const tile = this._tile;
            this._tile = tile < last ? tile + 1 : null;
            this._pyramidDue = tile >= last;
            return;
        }
        if (!this.building) return;
        const o = this.options;
        const queue = this._device.queue;
        const levels = this._pyramidViews.length;
        const side = this._back.width, height = this._back.height;
        const firstLayer = this._pyramidDue ? 0 : this._nextLayer ?? 0;
        const slab = this._slab;
        // with the pyramid, every dispatch's parameters in one write: the pyramid's levels, then
        // the volume's slabs
        if (this._pyramidDue) {
            const slabs = Math.ceil(height / slab);
            const data = new ArrayBuffer((levels + slabs) * SLOT);
            for (let k = 0; k < levels + slabs; k++) {
                const f = new Float32Array(data, k * SLOT, BUILD_BYTES / 4);
                const u = new Uint32Array(data, k * SLOT, BUILD_BYTES / 4);
                f.set([center[0], center[1], o.extentM, o.minHeightM, o.maxHeightM, this._eyeY, NEAR, this._far, Math.max(o.canopyExtinction, 0)]);
                u[9] = levels;
                u[10] = Math.min(k, levels);
                u[11] = Math.max(k - levels, 0) * slab;
            }
            queue.writeBuffer(this._buildParams, 0, data);
        }
        const pass = encoder.beginComputePass({ label: 'SkyOcclusion/Build', timestampWrites: gpuPass('SkyOcclusion/Build') });
        if (this._pyramidDue) {
            const res = this._resolution;
            pass.setPipeline(this._topPipeline);
            pass.setBindGroup(0, this._topBG);
            pass.dispatchWorkgroups(Math.ceil(res / 8), Math.ceil(res / 8));
            pass.setPipeline(this._downPipeline);
            for (let level = 1; level < levels; level++) {
                const size = Math.max(res >> level, 1);
                pass.setBindGroup(0, this._downBGs[level - 1]);
                pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));
            }
        }
        pass.setPipeline(this._volumePipeline);
        pass.setBindGroup(0, this._volumeBGs[firstLayer / slab]);
        pass.dispatchWorkgroups(Math.ceil(side / 4), Math.ceil(Math.min(slab, height - firstLayer) / 4), Math.ceil(side / 4));
        pass.end();
        this._pyramidDue = false;
        const next = firstLayer + slab;
        if (next < height) {
            this._nextLayer = next;
            return;
        }
        this._nextLayer = null;
        encoder.copyTextureToTexture({ texture: this._back }, { texture: this.volumeTexture }, [side, height, side]);
        queue.writeBuffer(this.params, 0, new Float32Array([
            center[0], center[1], 1 / o.extentM, o.minHeightM, 1 / Math.max(o.maxHeightM - o.minHeightM, 1e-3), 1, 0, 0,
        ]));
    }

    destroy(): void {
        for (const t of [this._depth, this._pyramid, this.volumeTexture, this._back]) t.destroy();
        for (const b of [this.params, this._buildParams, ...this._cameraBuffers]) b.destroy();
    }
}
