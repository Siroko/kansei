import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Camera } from "../cameras/Camera";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { Vector4 } from "../main";
import { Compute } from "../materials/Compute";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { GBuffer } from "../postprocessing/GBuffer";
import { ShadowMap, ShadowMapOptions } from "../shadows/ShadowMap";
import { CubeMapShadowMap, CubeMapShadowMapOptions } from "../shadows/CubeMapShadowMap";
import { SkyOcclusion, SkyOcclusionOptions } from "../shadows/SkyOcclusion";
import { CascadedShadowMap, CascadedShadowOptions, MAX_CASCADES } from "../shadows/CascadedShadowMap";
import { LightUniforms } from "../lights/LightUniforms";
import { BufferBase } from "../buffers/BufferBase";
import { SpotShadowAtlas } from "../shadows/SpotShadowAtlas";
import { SpotLightsGpu } from "../lights/SpotLightsGpu";
import { LightClusters } from "../lights/LightClusters";
import {
    BindGroupSlot, CASCADES_BYTES, CLUSTER_PARAMS_BYTES, LIGHT_UNIFORM_BYTES, MESH_TRANSFORMS_BYTES, SHADOW_UNIFORM_BYTES, SPOT_LIGHTS_BUFFER_BYTES,
    meshBindGroupLayoutEntries, meshSlotStride, shadowBindGroupLayoutEntries,
} from "./SharedLayouts";
import { FrameProfile, cpuScope, endProfiledFrame, gpuPass, setProfilingEnabled, takeProfile } from "../profiling/Profiler";
import { CULL_VIEW_BYTES, CullPipeline, CullView, cullView, cullViewDraws, drawGeometry, packCullView } from "../culling/InstanceCulling";
import { CullViewKind, CullingStats, StatsReadback } from "../culling/CullingStats";
import { mat4 } from "gl-matrix";
import { SceneVoxelGi, SceneVoxelGiOptions } from "../gi/SceneVoxelGi";
import { bakeImpostor } from "../impostors/bakeImpostor";
import type { Impostor, ImpostorOptions } from "../impostors/Impostor";
import { SceneRtGrid, SceneRtGridOptions } from "../rt/SceneRtGrid";

/**
 * Device limits the renderer requests from the adapter (Rust's `RequiredLimits`).
 * - `'default'`: WebGPU's default limits, which every adapter supports (for example 16 sampled
 *   textures and 8 storage buffers per shader stage), plus the adapter's
 *   `maxStorageBufferBindingSize`, which TS examples have always relied on.
 * - `'adapter'`: everything the adapter supports. Query what was granted with `Renderer.limits`.
 * - an object: exactly these limits (WebGPU's defaults for any left out); device creation fails
 *   if the adapter cannot meet them.
 */
export type RequiredLimits = 'default' | 'adapter' | Record<string, number>;

/**
 * The block-compressed texture formats a device can sample, from its enabled features
 * (Rust's `loaders::ktx2::CompressionSupport`; ASTC HDR has no WebGPU feature).
 */
export interface CompressionSupport {
    /** `texture-compression-bc`: BC1-7 (desktop GPUs, Apple Silicon). */
    bc: boolean;
    /** `texture-compression-astc`: ASTC LDR (Apple, most mobile). */
    astc: boolean;
    /** `texture-compression-etc2`: ETC2/EAC (mobile, Apple). */
    etc2: boolean;
}

/** Features requested only where the adapter offers them, so devices without them still initialize. */
const OPTIONAL_FEATURES: GPUFeatureName[] = [
    'timestamp-query',
    'texture-compression-bc',
    'texture-compression-astc',
    'texture-compression-etc2',
];

/** The draw sets of a scene pass, in the order they are drawn: `Scene.opaque`, `transmissive`, `transparent`. */
const DrawSet = { Opaque: 0, Transmissive: 1, Transparent: 2 } as const;
type DrawSet = typeof DrawSet[keyof typeof DrawSet];

/** The targets of a render pass, which pick each material's pipeline. */
interface PassTargets {
    colorFormats: GPUTextureFormat[];
    depthFormat: GPUTextureFormat;
    sampleCount: number;
}

/** A draw set's cached bundle (`null` when it records no draw) and what it recorded (`key`). */
interface SetBundle {
    bundle: GPURenderBundle | null;
    key: unknown[];
    valid: boolean;
}

/**
 * A pass's cached render bundles, one per draw set, and the shared bind groups and targets they
 * were recorded with (`shared`). A set's bundle is re-recorded only when its own draws change.
 */
class PassBundles {
    readonly sets: SetBundle[] = [0, 1, 2].map(() => ({ bundle: null, key: [], valid: false }));
    shared: unknown[] = [];

    invalidate(): void {
        for (const set of this.sets) set.valid = false;
    }
}

/**
 * Whether the cached bundles record `r` when it is visible. Dynamic renderables, and indirect
 * ones (whose draw counts the GPU writes), are drawn live after the bundle of their set.
 */
function isBundled(r: Renderable): boolean {
    return !r.dynamic && !r.geometry.indirectArgsBuffer;
}

function sameKey(a: unknown[], b: unknown[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

/** The cull view of the main camera (`Renderable.instanceCulling`). */
const MAIN_VIEW = 0;
/** The cull view of the directional shadow map the renderer owns (`enableShadows`). */
const SHADOW_VIEW = 1;

/** Encoder state already set while recording draws, so repeated state is not set again. */
interface DrawState {
    pipeline: GPURenderPipeline | null;
    materialBindGroup: GPUBindGroup | null;
    indexBuffer: GPUBuffer | null;
    vertexBuffer: GPUBuffer | null;
}

/**
 * Configuration options for the WebGPU renderer.
 * @interface RendererOptions
 * @property {boolean} [antialias] - Enable antialiasing
 * @property {boolean} [premultipliedAlpha] - Enable premultiplied alpha
 * @property {GPUCanvasAlphaMode} [alphaMode] - Canvas alpha mode configuration
 * @property {Vector4} [clearColor] - Clear color for the renderer
 */
export interface RendererOptions {
    antialias?: boolean;
    premultipliedAlpha?: boolean;
    alphaMode?: GPUCanvasAlphaMode;
    clearColor?: Vector4;
    width?: number;
    height?: number;
    sampleCount?: number;
    devicePixelRatio?: number;
    /** Device limits to request (see `RequiredLimits`; `'default'` unless raised). */
    requiredLimits?: RequiredLimits;
    /**
     * Require `float32-filterable` (linear sampling of r32float/rg32float/rgba32float textures,
     * which the voxel GI distance field needs), as the Rust renderer does. Initialization fails
     * with a clear error on adapters without it; pass `false` to run there without it.
     * Default `true`.
     */
    requireFloat32Filterable?: boolean;
    /**
     * Draw to this canvas instead of a new one (`Renderer.canvas`), e.g. a web `Canvas`'s
     * element, which sizes it: `web/Canvas.renderer` passes it with the drawing-buffer size and
     * a `devicePixelRatio` of 1.
     */
    canvas?: HTMLCanvasElement;
}

/**
 * Core WebGPU renderer class that handles initialization, rendering, and compute operations.
 * @class Renderer
 */
class Renderer {
    /**
     * Creates a new Renderer instance.
     * @constructor
     * @param {RendererOptions} options - Configuration options for the renderer
     * @throws {Error} Throws if WebGPU is not supported in the browser
     */

    public canvas: HTMLCanvasElement;
    public context: GPUCanvasContext | null;
    public device?: GPUDevice;
    private _presentationFormat?: GPUTextureFormat;
    private sampleCount: number = 4;
    private devicePixelRatio: number = window.devicePixelRatio;
    private colorTexture?: GPUTexture;
    private depthTexture?: GPUTexture;
    private width: number = 320;
    private height: number = 240;
    private clearColor: Vector4 = new Vector4(0, 0, 0, 0);
    private _renderScale: number = 1;

    // ── Public read-only accessors ──────────────────────────────────────────
    /** The initialised GPU device. Undefined before initialize() resolves. */
    public get gpuDevice(): GPUDevice { return this.device!; }
    /** Canvas colour format negotiated with the platform. */
    public get presentationFormat(): GPUTextureFormat { return this._presentationFormat!; }
    /** Canvas (display) width in physical pixels (includes devicePixelRatio). */
    public get renderWidth(): number { return this.width; }
    /** Canvas (display) height in physical pixels (includes devicePixelRatio). */
    public get renderHeight(): number { return this.height; }

    /**
     * Render the scene at `scale` times the canvas size (clamped to 0.25..1) through a
     * `PostProcessingVolume` (Rust `set_render_scale`). The GBuffer and every scene pass run at
     * `renderSize`; the chain's upscaler (`TemporalAAEffect`) reconstructs the canvas size from
     * the jittered frames, and the effects after it run at the canvas size. Without an upscaler
     * in the chain the blit stretches the image to the canvas. `render()` ignores it.
     */
    public setRenderScale(scale: number): void {
        this._renderScale = Number.isFinite(scale) ? Math.min(Math.max(scale, 0.25), 1) : 1;
    }

    /** See `setRenderScale`. */
    public get renderScale(): number { return this._renderScale; }

    /**
     * The size a `PostProcessingVolume` renders the scene at: the canvas size times the render
     * scale, rounded, at least 1 x 1.
     */
    public get renderSize(): [number, number] {
        const scaled = (n: number) => Math.max(Math.round(n * this._renderScale), 1);
        return [scaled(this.width), scaled(this.height)];
    }
    /** The shared per-object mesh bind group (normal + world matrices, dynamic offsets). */
    public get sharedMeshBindGroup(): GPUBindGroup | null { return this._sharedMeshBG; }
    /** Layout used for the shared mesh bind group. */
    public get sharedMeshBindGroupLayout(): GPUBindGroupLayout | null { return this._sharedMeshBGLayout; }
    /** Bytes between two objects' slots in the shared matrix buffers (the 128-byte mesh window rounded up to the device's offset alignment, typically 256). */
    public get matrixAlignment(): number { return this._matrixAlignment; }

    /**
     * Create a new GPU command encoder. Use this instead of touching the device.
     * Submit the encoder's finished command buffer with `submit()`.
     */
    public createCommandEncoder(label?: string): GPUCommandEncoder {
        return this.device!.createCommandEncoder(label ? { label } : undefined);
    }

    /** Submit one or more finished command buffers to the GPU queue. */
    public submit(commands: GPUCommandBuffer[] | GPUCommandBuffer): void {
        const list = Array.isArray(commands) ? commands : [commands];
        this.device!.queue.submit(list);
    }

    // Cached render bundles of the canvas pass and of the GBuffer pass: one per draw set,
    // holding the visible static renderables, each drawn at its scene slot. Dynamic and
    // indirect renderables are drawn live after their set's bundle.
    private _canvasBundles = new PassBundles();
    private _gbufferBundles = new PassBundles();
    private _bundleKeyScratch: unknown[] = [];
    private _sharedKeyScratch: unknown[] = [];
    private _bundleRecords: number = 0;

    /**
     * How many render bundles the renderer has recorded so far. A bundle is recorded per draw
     * set (opaque, transmissive, transparent) of a pass when what it draws changes: a renderable
     * shown, hidden, added, removed, given another material or geometry, or re-sorted among the
     * transparent ones.
     */
    public get bundleRecordCount(): number { return this._bundleRecords; }

    // Depth-copy pipeline: resolves MSAA depth from depthMSAATexture → depthTexture.
    // A fullscreen render pass reads texture_depth_multisampled_2d (sample 0) and
    // writes it via @builtin(frag_depth) into the non-MSAA depth attachment so that
    // compute shaders can sample it as texture_depth_2d.
    private _depthCopyPipeline: GPURenderPipeline | null = null;
    private _depthCopyBGL: GPUBindGroupLayout | null = null;
    private _depthCopyBG: GPUBindGroup | null = null;
    private _depthCopyBGSource: GPUTexture | null = null;

    // Shared matrix buffers — all objects' world and normal matrices packed into
    // two large GPU buffers (one per type) with 256-byte aligned strides, a slot
    // per renderable at its scene slot (`Scene.slotOf`). A world slot holds the
    // world matrix then last frame's (KanseiMeshTransforms). The renderer uploads
    // all matrices in exactly 2 writeBuffer calls per frame instead of 2×N.
    private _matrixAlignment: number = 256;  // meshSlotStride(device)
    private _worldMatricesBuf: GPUBuffer | null = null;
    private _normalMatricesBuf: GPUBuffer | null = null;
    private _worldMatricesStaging: Float32Array | null = null;
    private _normalMatricesStaging: Float32Array | null = null;
    private _sharedMeshBGLayout: GPUBindGroupLayout | null = null;
    private _sharedMeshBG: GPUBindGroup | null = null;
    private _meshSlotCapacity: number = 0;
    // The renderable each slot was last written for: its world matrix there is the previous one.
    private _slotOwners: (Renderable | null)[] = [];

    // The scene's directional and point lights, packed each frame (`LightUniforms`) into one
    // uniform that every camera drawn with binds at group 1 binding 2.
    private _lightPacker = new LightUniforms();
    private _lightData = new Float32Array(LIGHT_UNIFORM_BYTES / 4);
    private _lightUniforms = new ComputeBuffer({
        type: BufferBase.BUFFER_TYPE_UNIFORM,
        usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
        buffer: this._lightData,
    });

    /**
     * The scene light uniform (`KanseiLights` in `LIGHTS_WGSL`) the renderer binds in every
     * camera's group 1 at binding 2, rewritten when the scene's lights change. Effects that light
     * by the scene's lights can bind it too.
     */
    public get lightUniforms(): ComputeBuffer { return this._lightUniforms; }

    // ── Shadow resources (group 3, see SharedLayouts.shadowBindGroupLayoutEntries) ──
    /**
     * Whether materials sample the directional shadow map (`kansei_shadow_map`). `enableShadows`
     * turns it on; turning it off leaves the map rendering for effects that read it (the fog).
     */
    public shadowsEnabled: boolean = false;
    private _shadowMap: ShadowMap | null = null;
    private _cubeMapShadowMap: CubeMapShadowMap | null = null;
    // Whether the renderer renders the maps each frame (enableShadows, enablePointShadows), or
    // the caller does (the shadowMap and cubeMapShadowMap setters).
    private _ownsShadowMap: boolean = false;
    private _ownsCubeMapShadowMap: boolean = false;
    private _shadowBGL: GPUBindGroupLayout | null = null;
    private _shadowBG: GPUBindGroup | null = null;
    private _shadowUniformBuf: GPUBuffer | null = null;
    private _shadowComparisonSampler: GPUSampler | null = null;
    private _dummyShadowDepthTex: GPUTexture | null = null;
    private _dummyCubeShadowTex: GPUTexture | null = null;
    private _cubeShadowSampler: GPUSampler | null = null;
    // The cascaded shadow map (group 3 bindings 10-11, and its widest cascade at binding 0),
    // once enabled.
    private _cascadedShadowMap: CascadedShadowMap | null = null;
    // Bindings 5-12: a 1x1 depth array for the atlases not enabled (spot shadows, cascades) and
    // zeroed buffers (clusters off until a frame has spot lights, no cascades).
    private _dummyDepthArrayTex: GPUTexture | null = null;
    private _spotShadowSampler: GPUSampler | null = null;
    private _noClusterParamsBuf: GPUBuffer | null = null;
    private _noClusterLightsBuf: GPUBuffer | null = null;
    private _cascadesBuf: GPUBuffer | null = null;
    private _shadowBGDirty: boolean = true;
    private _skyOcclusion: SkyOcclusion | null = null;

    // GPU instance culling (renderables with `instanceCulling`): the cull pipeline, the frame's
    // views packed for the GPU, and the culling statistics.
    private _cullPipeline: CullPipeline | null = null;
    private _cullViewBytes = new ArrayBuffer(4 * CULL_VIEW_BYTES);
    private _cullStats = new StatsReadback();
    private _viewProjScratch = mat4.create();
    private _cascadeViewProjs = Array.from({ length: MAX_CASCADES }, () => mat4.create());

    /**
     * Read back each view's instance culling statistics (instances tested, outside their LOD band,
     * outside the frustum, drawn, triangles), a few frames late (`cullingStats`). Off by default.
     */
    public setCullingStats(enabled: boolean): void {
        this._cullStats.enabled = enabled;
    }

    /** The latest instance culling statistics while `setCullingStats` is on, or null. */
    public get cullingStats(): CullingStats | null {
        return this._cullStats.latest;
    }

    // Spot lights (group 3 bindings 5-9): the scene's, packed each frame into a fixed-capacity
    // storage buffer (so bind groups never go stale), the shadow atlas once enabled, and the
    // light clusters, created by the first frame that has spot lights to cluster.
    private _spotLightsBuf: GPUBuffer | null = null;
    private _spotLights = new SpotLightsGpu();
    private _spotLightsUploaded: number = 0;
    private _spotShadowAtlas: SpotShadowAtlas | null = null;
    private _lightClusters: LightClusters | null = null;
    private _clusteredLights: boolean = true;

    /** The directional shadow map materials sample (group 3 binding 0), or null. */
    public get shadowMap(): ShadowMap | null { return this._shadowMap; }
    /** Binds a shadow map the caller renders each frame (`ShadowMap.render`); see `enableShadows`. */
    public set shadowMap(value: ShadowMap | null) {
        if (this._ownsShadowMap && value !== this._shadowMap) this._shadowMap?.destroy();
        this._shadowMap = value;
        this._ownsShadowMap = false;
        this._shadowBGDirty = true;
    }

    /** The point-light cube shadow materials sample (group 3 binding 3), or null. */
    public get cubeMapShadowMap(): CubeMapShadowMap | null { return this._cubeMapShadowMap; }
    /**
     * Binds a cube shadow map the caller renders each frame (`CubeMapShadowMap.render`, then
     * `setPointShadowParams`); see `enablePointShadows`.
     */
    public set cubeMapShadowMap(value: CubeMapShadowMap | null) {
        if (this._ownsCubeMapShadowMap && value !== this._cubeMapShadowMap) this._cubeMapShadowMap?.destroy();
        this._cubeMapShadowMap = value;
        this._ownsCubeMapShadowMap = false;
        this._shadowBGDirty = true;
    }

    /**
     * Enables the directional shadow map (Rust's `enable_shadows`): each frame, before the scene
     * pass, the renderer renders it from the scene's first directional light with `castShadow`
     * (failing that, its first area light with `castShadow`, as a perspective map) over the
     * camera's view, drawing the `castShadow` renderables through their materials' depth
     * pipelines, and materials sample it through `kansei_shadow_map`. Returns the map, whose
     * options (`maxShadowDistance`, `bias`, ...) stay adjustable.
     */
    public enableShadows(options: ShadowMapOptions = {}): ShadowMap {
        if (this._ownsShadowMap) this._shadowMap?.destroy();
        this._shadowMap = new ShadowMap(this.device!, options);
        this._ownsShadowMap = true;
        this.shadowsEnabled = true;
        this._shadowBGDirty = true;
        return this._shadowMap;
    }

    /**
     * Enables point-light cube shadows (Rust's `enable_point_shadows`): each frame, before the
     * scene pass, the renderer renders the faces of the scene's point lights with `castShadow`
     * (up to `maxLights`), and materials sample the first one's through `kansei_point_shadow`.
     * Effects such as the fog read every light's faces (`CubeMapShadowMap.lights`).
     */
    public enablePointShadows(options: CubeMapShadowMapOptions = {}): CubeMapShadowMap {
        if (this._ownsCubeMapShadowMap) this._cubeMapShadowMap?.destroy();
        this._cubeMapShadowMap = new CubeMapShadowMap(this.device!, options);
        this._ownsCubeMapShadowMap = true;
        this._shadowBGDirty = true;
        return this._cubeMapShadowMap;
    }

    // Voxel GI of the scene's meshes (`enableVoxelGI`)
    private _voxelGI: SceneVoxelGi | null = null;
    // The voxel GI and spot atlas its injection was last given the spot lights for.
    private _voxelGISpots: { gi: SceneVoxelGi, atlas: SpotShadowAtlas | null } | null = null;

    /**
     * Voxel GI for the scene's meshes over a box (`SceneVoxelGi`, Rust's `enable_voxel_gi`): every
     * frame, after the shadow maps, the renderables with a `Renderable.gi` surface are voxelized
     * (the static ones when they change) and lit by the scene's lights through their shadow maps,
     * with the bounces adding up over frames. Read the result with `VoxelGIEffect`. Calling it
     * again replaces the volume.
     *
     * ```ts
     * const gi = renderer.enableVoxelGI({ boundsMin, boundsMax });
     * const effect = new VoxelGIEffect(gi.volume, { quality: gi.quality });
     * ```
     */
    public enableVoxelGI(options: SceneVoxelGiOptions): SceneVoxelGi {
        this._voxelGI?.destroy();
        this._voxelGI = new SceneVoxelGi(this.device!, options);
        const gi = this._voxelGI;
        console.info(`voxel GI: ${gi.quality}, [${gi.volume.dims.join(', ')}] voxels, ${(gi.memoryBytes() / (1 << 20)).toFixed(1)} MiB`);
        return gi;
    }

    /** Turn voxel GI off and free its volume. */
    public disableVoxelGI(): void {
        this._voxelGI?.destroy();
        this._voxelGI = null;
    }

    /** The scene's voxel GI, once `enableVoxelGI` has been called. */
    public get voxelGI(): SceneVoxelGi | null { return this._voxelGI; }

    /**
     * Sky occlusion around the camera: how much of the sky each point sees past the canopy
     * (`SkyOcclusion`), for materials to dim their sky ambient light by with
     * `SKY_OCCLUSION_WGSL`. The shadow casters on its layers are drawn from straight above when
     * the camera has moved far enough, culled on the GPU; call `SkyOcclusion.refresh` (through
     * `skyOcclusion`) after changing the scene under it. Rust: `enable_sky_occlusion`.
     */
    public enableSkyOcclusion(options: SkyOcclusionOptions = {}): SkyOcclusion {
        this._skyOcclusion?.destroy();
        this._skyOcclusion = new SkyOcclusion(this.device!, options);
        return this._skyOcclusion;
    }

    /** The sky occlusion, once `enableSkyOcclusion` has been called. */
    public get skyOcclusion(): SkyOcclusion | null { return this._skyOcclusion; }

    /**
     * Enables perspective shadow maps for spot lights (Rust's `enable_spot_shadows`): each frame,
     * before the scene pass, the first `maxLights` spot lights with `castShadow` (in scene order)
     * render a `resolution`² layer of the spot shadow atlas, drawing every visible `castShadow`
     * renderable through its material's depth pipeline. Materials sample it through
     * `SPOT_LIGHTS_WGSL` (`kansei_spot_shadow`, PCSS by each light's `sourceRadius`).
     */
    public enableSpotShadows(resolution: number, maxLights: number): SpotShadowAtlas {
        this._spotShadowAtlas?.destroy();
        this._spotShadowAtlas = new SpotShadowAtlas(this.device!, resolution, maxLights);
        this._shadowBGDirty = true;
        return this._spotShadowAtlas;
    }

    /** The spot-light shadow atlas, once `enableSpotShadows` has been called. */
    public get spotShadowAtlas(): SpotShadowAtlas | null { return this._spotShadowAtlas; }

    /**
     * The storage buffer holding the scene's spot lights (`KanseiSpotLights` in
     * `SPOT_LIGHT_TYPES_WGSL`), rewritten every frame. Materials see it at group 3 binding 6;
     * effects bind it themselves.
     */
    public get spotLightsBuffer(): GPUBuffer {
        this._ensureShadowResources();
        return this._spotLightsBuf!;
    }

    /**
     * Shade spot lights through the clustered light lists (the default; Rust's
     * `set_clustered_lights`): each fragment visits only the lights whose range and cone reach
     * its cluster. Off, every fragment visits every light (for comparisons and debugging).
     */
    public setClusteredLights(enabled: boolean): void {
        this._clusteredLights = enabled;
    }

    /** Whether spot lights are shaded through the clustered light lists (`setClusteredLights`). */
    public get clusteredLights(): boolean { return this._clusteredLights; }

    // The ray tracing grid of the scene's triangles (`enableRtGrid`)
    private _rtGrid: SceneRtGrid | null = null;
    private _rtViewProj = mat4.create();

    /**
     * A ray tracing grid of the scene's triangles round the camera (`SceneRtGrid`, Rust's
     * `enable_rt_grid`), for passes that trace rays through it (`RT_GRID_WGSL`): the renderables
     * with `Renderable.rt`, culled for its box on the GPU, rebuilt after the frame's culling when
     * the box moves or what it holds changes. Calling it again replaces the grid.
     *
     * ```ts
     * const rt = renderer.enableRtGrid({ grid: { dims: [128, 64, 128], cell: 0.5 } });
     * const reflections = new RtReflectionsEffect(renderer.voxelGI!.volume, rt.handle);
     * ```
     */
    public enableRtGrid(options: SceneRtGridOptions = {}): SceneRtGrid {
        this._rtGrid?.destroy();
        this._rtGrid = new SceneRtGrid(this.device!, options);
        return this._rtGrid;
    }

    /** Turn the ray tracing grid off and free it. */
    public disableRtGrid(): void {
        this._rtGrid?.destroy();
        this._rtGrid = null;
    }

    /** The ray tracing grid, once `enableRtGrid` has been called. */
    public get rtGrid(): SceneRtGrid | null { return this._rtGrid; }

    /**
     * Enables cascaded shadow maps for the scene's first directional light, when it has
     * `castShadow` (the sun, or the moon at night; Rust's `enable_cascaded_shadows`): each frame,
     * stable cascades fitted to the camera are drawn through the casters' own vertex shaders
     * (`Material.getDepthPipeline`), the instanced ones culled per cascade on the GPU, and
     * materials that include `CASCADED_SHADOWS_WGSL` sample them with contact-hardening PCSS
     * (`kansei_sun_shadow`). It replaces `enableShadows`: the widest cascade also serves shaders
     * that read the single directional map (group 3 binding 0, `kansei_shadow_map`), and a map
     * from `enableShadows` is not rendered while the cascades are on.
     */
    public enableCascadedShadows(options: CascadedShadowOptions = {}): CascadedShadowMap {
        this._cascadedShadowMap?.destroy();
        this._cascadedShadowMap = new CascadedShadowMap(this.device!, options);
        this.shadowsEnabled = true;
        this._shadowBGDirty = true;
        return this._cascadedShadowMap;
    }

    /**
     * The cascaded shadow map, once `enableCascadedShadows` has been called: the volumetric fog
     * takes it (`setCascadedShadowMap`) for shafts from its widest cascade.
     */
    public get cascadedShadowMap(): CascadedShadowMap | null { return this._cascadedShadowMap; }

    constructor(
        private options: RendererOptions = {}
    ) {
        this.canvas = this.options.canvas ?? document.createElement('canvas');
        this.context = this.canvas.getContext('webgpu');
        this.sampleCount = this.options.sampleCount || this.sampleCount;
        this.devicePixelRatio = this.options.devicePixelRatio || this.devicePixelRatio;
        this.width = this.options.width || this.width;
        this.height = this.options.height || this.height;
        this.clearColor = this.options.clearColor || this.clearColor;

        if (!this.context || navigator.gpu == null) {
            throw new Error('WebGPU is not supported');
        }
    }

    private async getDevice(): Promise<GPUDevice> {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter == null) {
            throw new Error("No WebGPU adapter found");
        }

        const requiredFeatures: GPUFeatureName[] = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
        if (adapter.features.has('float32-filterable')) {
            requiredFeatures.push('float32-filterable');
        } else if (this.options.requireFloat32Filterable ?? true) {
            throw new Error(
                "This GPU does not support 'float32-filterable'; " +
                "create the Renderer with { requireFloat32Filterable: false } to run without it " +
                "(features that filter 32-bit float textures, such as the voxel GI distance field, will not work)"
            );
        }

        const device = await adapter.requestDevice({
            label: 'Kansei Device',
            requiredFeatures,
            requiredLimits: Renderer.resolveLimits(this.options.requiredLimits ?? 'default', adapter.limits),
        });
        const c = Renderer.compressionSupportOf(device);
        // Same lines as the Rust renderer's device log (WebGPU has no ASTC HDR feature).
        console.info(`texture compression: CompressionSupport { bc: ${c.bc}, astc: ${c.astc}, etc2: ${c.etc2}, astc_hdr: false }`);
        const limits = device.limits;
        console.info(
            `device limits: ${limits.maxSampledTexturesPerShaderStage} sampled textures, ` +
            `${limits.maxSamplersPerShaderStage} samplers, ` +
            `${limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; ` +
            `textures up to ${limits.maxTextureDimension2D}`
        );
        return device;
    }

    /** The limits to request from an adapter with `adapter` limits under `policy`. */
    private static resolveLimits(policy: RequiredLimits, adapter: GPUSupportedLimits): Record<string, number> {
        if (policy === 'default') {
            return { maxStorageBufferBindingSize: adapter.maxStorageBufferBindingSize };
        }
        if (policy === 'adapter') {
            // GPUSupportedLimits exposes its values as prototype getters, so copy them by name.
            const all: Record<string, number> = {};
            for (const key in adapter) {
                const value = (adapter as unknown as Record<string, unknown>)[key];
                if (typeof value === 'number') all[key] = value;
            }
            return all;
        }
        return policy;
    }

    private static compressionSupportOf(device: GPUDevice): CompressionSupport {
        return {
            bc: device.features.has('texture-compression-bc'),
            astc: device.features.has('texture-compression-astc'),
            etc2: device.features.has('texture-compression-etc2'),
        };
    }

    /** The limits the device was created with (see `RendererOptions.requiredLimits`). */
    public get limits(): GPUSupportedLimits { return this.device!.limits; }

    /** The features the device was created with (`timestamp-query`, `float32-filterable`, ...). */
    public get features(): GPUSupportedFeatures { return this.device!.features; }

    /**
     * The block-compressed texture formats the device can sample, which decide what KTX2
     * textures transcode to.
     */
    public get compressionSupport(): CompressionSupport { return Renderer.compressionSupportOf(this.device!); }

    /**
     * Initializes the WebGPU device and context.
     * @async
     * @returns {Promise<void>}
     */
    public async initialize(): Promise<void> {
        if (!this.device) {
            const device = await this.getDevice();
            if (!this.device) {
                this.device = device;
                this._presentationFormat = navigator.gpu.getPreferredCanvasFormat();
                this.context?.configure({
                    device: this.device,
                    format: this._presentationFormat,
                    alphaMode: this.options?.alphaMode || "opaque",
                });

                this.setSize(
                    this.options.width || this.canvas.width,
                    this.options.height || this.canvas.height
                );
            }
        }
        return Promise.resolve();
    }

    /**
     * Sets the size of the rendering canvas and updates related resources.
     * @param {number} width - Canvas width in pixels
     * @param {number} height - Canvas height in pixels
     */
    public setSize(width: number, height: number) {
        this.resize(width * this.devicePixelRatio, height * this.devicePixelRatio);
    }

    /**
     * Sets the drawing-buffer size in pixels (no device pixel ratio applied) and recreates the
     * size-dependent targets: what `web/run`'s `Frame.resize` calls when the canvas changes size.
     * @param {number} width - Drawing-buffer width in pixels
     * @param {number} height - Drawing-buffer height in pixels
     */
    public resize(width: number, height: number) {
        this.width = width;
        this.height = height;
        this.canvas.width = this.width;
        this.canvas.height = this.height;

        this.depthTexture?.destroy();
        this.depthTexture = this.device!.createTexture({
            size: [this.canvas.width, this.canvas.height],
            sampleCount: this.sampleCount,
            dimension: '2d',
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.colorTexture?.destroy();
        this.colorTexture = this.device!.createTexture({
            size: [this.canvas.width, this.canvas.height],
            sampleCount: this.sampleCount,
            format: this.presentationFormat!,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    /**
     * Profile frames (see `profiling/Profiler`): each labelled pass's GPU time, from timestamp
     * queries (where the adapter offers `timestamp-query`), and each labelled section's CPU time.
     * Off by default; costs a null check per pass and section while off. `render()` and
     * `PostProcessingVolume.render()` end a profiled frame; other frames call `endProfiledFrame()`.
     */
    public setProfiling(enabled: boolean): void {
        setProfilingEnabled(this.device!, enabled);
    }

    /**
     * End a profiled frame of work recorded without `render()` or a `PostProcessingVolume`
     * (offscreen tools and tests), after its last submit.
     */
    public endProfiledFrame(): void {
        endProfiledFrame();
    }

    /**
     * The frames profiled since the last call, averaged per frame (the GPU's arrive a few frames
     * late); `FrameProfile.report()` formats them.
     */
    public takeProfile(): FrameProfile {
        return takeProfile();
    }

    /**
     * Forces every cached render bundle to be re-recorded on the next frame. The renderer
     * already re-records a draw set's bundle when what it draws changes (renderables shown,
     * hidden, added, removed, or given another material or geometry); call this after changing
     * something a bundle holds that it cannot see, such as a geometry's vertex buffers.
     */
    public invalidateBundle() {
        this._canvasBundles.invalidate();
        this._gbufferBundles.invalidate();
    }

    /**
     * Creates or grows the shared per-object matrix GPU buffers to hold `slotCount` slots.
     *
     * All objects' world and normal matrices are packed into two large GPU
     * buffers (one per type) with 256-byte aligned strides so every object's
     * slice is accessible via a dynamic uniform buffer offset.  This lets the
     * renderer upload ALL matrices in exactly 2 writeBuffer calls per frame.
     * The buffers only grow, keeping what the slots held.
     */
    private _ensureSharedMeshResources(slotCount: number) {
        if (slotCount <= this._meshSlotCapacity && this._sharedMeshBG !== null) return;

        const alignment = meshSlotStride(this.device!);
        this._matrixAlignment = alignment;

        const capacity = Math.max(slotCount, 1); // at least one slot
        const floatsPerSlot = alignment / 4; // 64 floats for 256-byte alignment

        this._worldMatricesBuf?.destroy();
        this._normalMatricesBuf?.destroy();

        this._worldMatricesBuf = this.device!.createBuffer({
            label: 'WorldMatrices',
            size: capacity * alignment,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._normalMatricesBuf = this.device!.createBuffer({
            label: 'NormalMatrices',
            size: capacity * alignment,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Padding bytes stay zero every frame; the slots keep their matrices (the previous world).
        const world = new Float32Array(capacity * floatsPerSlot);
        const normal = new Float32Array(capacity * floatsPerSlot);
        if (this._worldMatricesStaging) world.set(this._worldMatricesStaging);
        if (this._normalMatricesStaging) normal.set(this._normalMatricesStaging);
        this._worldMatricesStaging = world;
        this._normalMatricesStaging = normal;
        this._slotOwners.length = capacity;
        this._slotOwners.fill(null, this._meshSlotCapacity);

        // Create the layout once; all subsequent bind groups reuse it.
        if (!this._sharedMeshBGLayout) {
            this._sharedMeshBGLayout = this.device!.createBindGroupLayout({
                label: 'SharedMesh BindGroupLayout',
                entries: meshBindGroupLayoutEntries(),
            });
        }

        // Bind the first slot's normal matrix and world + previous world; the
        // dynamic offset shifts that window to the i-th object's slot at draw time.
        // A new bind group re-records the cached bundles.
        this._sharedMeshBG = this.device!.createBindGroup({
            label: 'SharedMesh BindGroup',
            layout: this._sharedMeshBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this._normalMatricesBuf, size: 64 } },
                { binding: 1, resource: { buffer: this._worldMatricesBuf,  size: MESH_TRANSFORMS_BYTES } },
            ],
        });

        this._meshSlotCapacity = capacity;
    }

    /**
     * Copies a renderable's matrices into its slot of the staging arrays: the
     * normal matrix, and the world matrix after the one it had there last frame
     * (or the world matrix itself when the slot was last written for another renderable).
     */
    private _stageMeshSlot(slot: number, renderable: Renderable) {
        const base = slot * (this._matrixAlignment / 4);
        const world = this._worldMatricesStaging!;
        if (this._slotOwners[slot] === renderable) {
            world.copyWithin(base + 16, base, base + 16);
        } else {
            world.set(renderable.worldMatrix.internalMat4, base + 16);
            this._slotOwners[slot] = renderable;
        }
        world.set(renderable.worldMatrix.internalMat4, base);
        this._normalMatricesStaging!.set(renderable.normalMatrix.internalMat4, base);
    }

    /** Uploads every slot's matrices in one write per buffer. */
    private _uploadMeshSlots() {
        this.device!.queue.writeBuffer(this._worldMatricesBuf!,  0, this._worldMatricesStaging!.buffer as ArrayBuffer);
        this.device!.queue.writeBuffer(this._normalMatricesBuf!, 0, this._normalMatricesStaging!.buffer as ArrayBuffer);
    }

    /**
     * Creates the shadow GPU resources (the group 3 layout, its uniform buffer and samplers, and
     * the dummies bound for what is not enabled) on first use.
     */
    private _ensureShadowResources(): void {
        if (this._shadowBGL) return;

        this._dummyShadowDepthTex = this.device!.createTexture({
            label: 'Shadow/DummyDepth',
            size: [1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        this._shadowComparisonSampler = this.device!.createSampler({
            label: 'Shadow/ComparisonSampler',
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // mat4(64) + bias(4) + normalBias(4) + shadowEnabled(4) + pointShadowEnabled(4)
        // + pointLightPos(12) + pointShadowFar(4) = 96 bytes
        this._shadowUniformBuf = this.device!.createBuffer({
            label: 'Shadow/Uniforms',
            size: SHADOW_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Dummy 1×1 6-layer r32float texture for cubemap shadow (filled with large distance)
        this._dummyCubeShadowTex = this.device!.createTexture({
            label: 'Shadow/DummyCube',
            size: [1, 1, 6],
            format: 'r32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        // Fill with large distance so everything is lit by default
        const farDist = new Float32Array([1e10]);
        for (let i = 0; i < 6; i++) {
            this.device!.queue.writeTexture(
                { texture: this._dummyCubeShadowTex, origin: [0, 0, i] },
                farDist,
                { bytesPerRow: 4 },
                [1, 1, 1],
            );
        }

        this._cubeShadowSampler = this.device!.createSampler({
            label: 'Shadow/CubeSampler',
            magFilter: 'nearest',
            minFilter: 'nearest',
        });

        this._dummyDepthArrayTex = this.device!.createTexture({
            label: 'Shadow/DummyDepthArray',
            size: [1, 1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        const zeroed = (label: string, size: number, usage: GPUBufferUsageFlags) =>
            this.device!.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
        this._spotLightsBuf = zeroed('Renderer/SpotLights', SPOT_LIGHTS_BUFFER_BYTES, GPUBufferUsage.STORAGE);
        this._noClusterParamsBuf = zeroed('Shadow/NoLightClusters', CLUSTER_PARAMS_BYTES, GPUBufferUsage.UNIFORM);
        this._noClusterLightsBuf = zeroed('Shadow/NoClusterLights', 16, GPUBufferUsage.STORAGE);
        this._cascadesBuf = zeroed('Shadow/NoCascades', CASCADES_BYTES, GPUBufferUsage.UNIFORM);
        this._spotShadowSampler = this.device!.createSampler({
            label: 'Renderer/SpotShadowSampler',
            compare: 'less-equal',
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });

        this._shadowBGL = this.device!.createBindGroupLayout({
            label: 'Shadow BindGroupLayout',
            entries: shadowBindGroupLayoutEntries(),
        });

        this._shadowBGDirty = true;
    }

    /**
     * Creates or recreates the shadow bind group (group 3) when a shadow map changes, with the
     * dummies standing in for what is not enabled.
     */
    private _updateShadowBindGroup(): void {
        if (!this._shadowBGDirty) return;
        this._ensureShadowResources();

        // the cascaded map's widest cascade stands in for the single directional map
        const csm = this._cascadedShadowMap;
        const depthView = csm?.farView
            ?? (this._shadowMap ? this._shadowMap.depthTexture : this._dummyShadowDepthTex!).createView();

        const cubeTex = this._cubeMapShadowMap
            ? this._cubeMapShadowMap.distanceTexture
            : this._dummyCubeShadowTex!;
        const depthArray = this._dummyDepthArrayTex!.createView({ dimension: '2d-array' });
        const spotAtlas = this._spotShadowAtlas?.arrayView ?? depthArray;
        const clusters = this._lightClusters;

        this._shadowBG = this.device!.createBindGroup({
            label: 'Shadow BindGroup',
            layout: this._shadowBGL!,
            entries: [
                { binding: 0, resource: depthView },
                { binding: 1, resource: this._shadowComparisonSampler! },
                { binding: 2, resource: { buffer: this._shadowUniformBuf! } },
                { binding: 3, resource: cubeTex.createView({ dimension: '2d-array' }) },
                { binding: 4, resource: this._cubeShadowSampler! },
                { binding: 5, resource: spotAtlas },
                { binding: 6, resource: { buffer: this._spotLightsBuf! } },
                { binding: 7, resource: this._spotShadowSampler! },
                { binding: 8, resource: { buffer: clusters?.params ?? this._noClusterParamsBuf! } },
                { binding: 9, resource: { buffer: clusters?.lights ?? this._noClusterLightsBuf! } },
                { binding: 10, resource: csm?.arrayView ?? depthArray },
                { binding: 11, resource: { buffer: csm?.uniform ?? this._cascadesBuf! } },
                { binding: 12, resource: this._spotShadowSampler! },
            ],
        });

        this._shadowBGDirty = false;
    }

    /**
     * Group 3 as the scene passes bind it: the shadow maps and lights enabled, dummies for the
     * rest (after `initialize`).
     */
    public get shadowBindGroup(): GPUBindGroup {
        this._updateShadowBindGroup();
        return this._shadowBG!;
    }

    /**
     * Bakes an octahedral impostor of the renderables `parts` (the parts of one object), drawn
     * together with their own materials' GBuffer pipelines (see `Impostor`, `bakeImpostor`). The
     * lights and shadows bound are the renderer's. Rust: `Renderer::bake_impostor`.
     */
    public bakeImpostor(parts: readonly Renderable[], options: ImpostorOptions = {}): Impostor {
        const device = this.device!;
        if (!this._lightUniforms.initialized) this._lightUniforms.initialize(device);
        return bakeImpostor({ device, lightBuffer: this._lightUniforms.gpuBuffer!, shadowBindGroup: this.shadowBindGroup }, parts, options);
    }

    /**
     * Renders a stack using the specified camera.
     *
     * Each frame has three phases:
     *  1. Update — compute the visible renderables' matrices on the CPU, copy them into
     *     their scene slots of the staging arrays, then upload via exactly 2 writeBuffer calls.
     *     Then the shadow maps the renderer owns are rendered (`_encodeShadowPasses`).
     *  2. Bundle — re-record a draw set's bundle (opaque, transmissive, transparent) only when
     *     the static renderables it draws changed (dynamic offsets into the shared buffers are
     *     baked per object, at its stable slot).
     *  3. Execute — replay each set's bundle, then draw its dynamic and indirect renderables live.
     *
     * @param {Scene} stack - The stack to render
     * @param {Camera} camera - The camera to use for rendering
     */
    public render(stack: Scene, camera: Camera) {
        stack.prepare(camera);
        camera.updateViewMatrix();

        const cameraBindGroup = this._prepareCamera(stack, camera);
        const targets: PassTargets = {
            colorFormats: [this.presentationFormat],
            depthFormat: 'depth24plus',
            sampleCount: this.sampleCount,
        };

        // Phase 1 — update matrices and upload them.
        this._updateRenderables(stack, camera, (renderable) => {
            if (!renderable.material.initialized) {
                renderable.material.initialize(
                    this.device!,
                    renderable.geometry.vertexBuffersDescriptors!,
                    this.presentationFormat!,
                    this.sampleCount
                );
            }
        });

        // The instances each view draws, then the shadow views and their uniforms, then the
        // light clusters for the camera.
        const commandRenderEncoder = this.device!.createCommandEncoder();
        this._planShadowViews(stack, camera);
        this._runInstanceCulling(commandRenderEncoder, stack, camera);
        this._encodeShadowPasses(commandRenderEncoder, stack);
        this._encodeVoxelGI(commandRenderEncoder, stack, camera);
        this._uploadShadowUniforms();
        this._encodeLightClusters(commandRenderEncoder, camera, this.width, this.height);
        this._updateShadowBindGroup();

        // Phase 2 — (re-)record the bundles whose draws changed.
        const sets = [stack.opaque, stack.transmissive, stack.transparent];
        this._syncBundles(this._canvasBundles, sets, stack, cameraBindGroup, targets);

        // Phase 3 — execute the bundles and the live draws in a fresh render pass.
        const textureView = this.context!.getCurrentTexture().createView();

        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.sampleCount > 1 ? this.colorTexture!.createView() : textureView,
                    resolveTarget: this.sampleCount > 1 ? textureView : undefined,
                    clearValue: {
                        r: this.options.clearColor?.x || 0.0,
                        g: this.options.clearColor?.y || 0.0,
                        b: this.options.clearColor?.z || 0.0,
                        a: this.options.clearColor?.w || 0.0
                    },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthTexture!.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        } as GPURenderPassDescriptor;

        renderPassDescriptor.label = 'Renderer/MainPass';
        renderPassDescriptor.timestampWrites = gpuPass('Renderer/MainPass');
        const passRenderEncoder = commandRenderEncoder.beginRenderPass(renderPassDescriptor);
        for (const set of [DrawSet.Opaque, DrawSet.Transmissive, DrawSet.Transparent]) {
            this._drawSet(passRenderEncoder, this._canvasBundles, set, sets[set], stack, cameraBindGroup, targets);
        }
        passRenderEncoder.end();
        this.device!.queue.submit([commandRenderEncoder.finish()]);
        this._endCulledFrame(camera);
        camera.endFrame();
        endProfiledFrame();
    }

    /**
     * Packs the scene's lights into the shared light uniform (written only when they changed),
     * binds it in `camera`'s group, writes the camera's jittered projection and temporal data,
     * and returns its bind group with everything uploaded.
     */
    private _prepareCamera(stack: Scene, camera: Camera): GPUBindGroup {
        const packed = this._lightPacker.data;
        this._lightPacker.pack(stack.directionalLights, stack.pointLights);
        const current = this._lightData;
        for (let i = 0; i < packed.length; i++) {
            if (packed[i] !== current[i]) {
                current.set(packed);
                this._lightUniforms.needsUpdate = true;
                break;
            }
        }
        camera.useLightUniforms(this._lightUniforms);
        camera.uploadTemporal();
        return camera.getBindGroup(this.device!);
    }

    /**
     * Phase 1 of a scene pass: initialises the visible renderables' geometry (`prepareMaterial`
     * readies their material for the pass), updates their matrices, stages them at their scene
     * slots and uploads every slot.
     */
    private _updateRenderables(stack: Scene, camera: Camera, prepareMaterial: (renderable: Renderable) => void) {
        const orderedObjects = stack.getOrderedObjects();
        this._ensureSharedMeshResources(stack.slotCapacity);

        for (let i = 0; i < orderedObjects.length; i++) {
            const renderable = orderedObjects[i];

            if (!renderable.geometry.initialized) {
                renderable.geometry.initialize(this.device!);
            }
            prepareMaterial(renderable);
            if (renderable.geometry.isInstancedGeometry) {
                const geo = renderable.geometry as InstancedGeometry;
                for (const extraBuffer of geo.extraBuffers) {
                    if (!extraBuffer.initialized) extraBuffer.initialize(this.device!);
                }
            }

            renderable.updateModelMatrix();
            renderable.updateNormalMatrix(camera.viewMatrix);

            this._stageMeshSlot(stack.slotOf(renderable), renderable);

            // Flush any dirty material-level buffers (textures, material uniforms).
            renderable.material.getBindGroup(this.device!);
            // A swapped material re-records its set's bundle through the bundle key.
            renderable.materialDirty = false;
        }

        this._uploadMeshSlots();
    }

    // ── Depth-copy helpers ───────────────────────────────────────────────────

    /**
     * Builds the depth-copy render pipeline on first use.
     * The pipeline draws a fullscreen triangle, reads sample 0 from a
     * texture_depth_multisampled_2d, and writes it as @builtin(frag_depth)
     * into a non-MSAA depth32float attachment.
     */
    private _ensureDepthCopyPipeline(): void {
        if (this._depthCopyPipeline) return;

        const shader = /* wgsl */`
            @group(0) @binding(0) var msaaDepth: texture_depth_multisampled_2d;

            @vertex
            fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
                const pos = array<vec2f, 3>(
                    vec2f(-1.0, -1.0),
                    vec2f( 3.0, -1.0),
                    vec2f(-1.0,  3.0),
                );
                return vec4f(pos[vi], 0.0, 1.0);
            }

            struct DepthOut { @builtin(frag_depth) depth: f32 }

            @fragment
            fn fs(@builtin(position) fragPos: vec4f) -> DepthOut {
                let coord = vec2i(i32(fragPos.x), i32(fragPos.y));
                // Resolve MSAA depth by taking the closest (min) sample.
                // Using only sample 0 would create wrong depth at silhouette edges,
                // causing DoF to alternate between sharp and max-blur on edge pixels.
                let d0 = textureLoad(msaaDepth, coord, 0);
                let d1 = textureLoad(msaaDepth, coord, 1);
                let d2 = textureLoad(msaaDepth, coord, 2);
                let d3 = textureLoad(msaaDepth, coord, 3);
                return DepthOut(min(min(d0, d1), min(d2, d3)));
            }
        `;

        const module = this.device!.createShaderModule({ code: shader });

        this._depthCopyBGL = this.device!.createBindGroupLayout({
            label: 'DepthCopy/BGL',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'depth', multisampled: true },
            }],
        });

        this._depthCopyPipeline = this.device!.createRenderPipeline({
            label: 'DepthCopy/Pipeline',
            layout: this.device!.createPipelineLayout({ bindGroupLayouts: [this._depthCopyBGL] }),
            vertex: { module, entryPoint: 'vs' },
            fragment: { module, entryPoint: 'fs', targets: [] },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'always',
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    /** Returns (and lazily creates) the depth-copy bind group for the given MSAA depth texture. */
    private _getDepthCopyBindGroup(msaaDepth: GPUTexture): GPUBindGroup {
        if (this._depthCopyBGSource !== msaaDepth) {
            this._depthCopyBG = this.device!.createBindGroup({
                label: 'DepthCopy/BindGroup',
                layout: this._depthCopyBGL!,
                entries: [{ binding: 0, resource: msaaDepth.createView() }],
            });
            this._depthCopyBGSource = msaaDepth;
        }
        return this._depthCopyBG!;
    }

    /**
     * Renders the scene into a GBuffer for post-processing.
     *
     * This is a drop-in replacement for render() when a PostProcessingVolume is in use.
     * It performs the same three-phase matrix-upload (then shadow views) / bundle-record / execute loop but
     * targets the GBuffer's four MRT targets and depth32float depth at the GBuffer's sample
     * count (1 unless the volume asked for MSAA; temporal AA handles aliasing then), then draws
     * the velocity pass into `GBuffer.velocityTexture`.
     *
     * It does not end the camera's frame: `PostProcessingVolume.render` calls `camera.endFrame()`
     * after its effects, which may read this frame's and last frame's view; call it yourself when
     * driving a GBuffer without the volume.
     *
     * @param stack   - The scene to render.
     * @param camera  - The camera to use.
     * @param gbuffer - The GBuffer to write colour and depth data into.
     */
    public renderToGBuffer(stack: Scene, camera: Camera, gbuffer: GBuffer): void {
        const sceneScope = cpuScope('scene');
        let t = cpuScope('scene/prepare');
        stack.prepare(camera);
        camera.updateViewMatrix();

        const cameraBindGroup = this._prepareCamera(stack, camera);

        // MRT format array: one entry per color attachment.
        const mrtFormats: GPUTextureFormat[] = [
            'rgba16float',  // @location(0) color
            'rgba16float',  // @location(1) emissive
            'rgba16float',  // @location(2) normal
            'rgba8unorm',   // @location(3) albedo
        ];
        const targets: PassTargets = {
            colorFormats: mrtFormats,
            depthFormat: 'depth32float',
            sampleCount: gbuffer.msaaSampleCount,
        };

        t?.end();

        // Phase 1 — identical to render(): upload matrices.
        t = cpuScope('scene/upload');
        this._updateRenderables(stack, camera, (renderable) => {
            // Ensure the material has a pipeline compiled for the GBuffer MRT config.
            this._pipelineFor(renderable, targets);
            if (renderable.material.outputsVelocity && !renderable.material.transparent) {
                renderable.material.getVelocityPipeline(this.device!, renderable.geometry.vertexBuffersDescriptors);
            }
            // Mark initialized to skip initialize() which would build a
            // canvas-format pipeline that fails for shaders with @location(1).
            renderable.material.initialized = true;
        });
        t?.end();

        // The instances each view draws (after the shadow maps' lights are placed), then the
        // shadow views and their uniforms.
        t = cpuScope('scene/culling');
        const commandEncoder = this.device!.createCommandEncoder();
        this._planShadowViews(stack, camera);
        // the ray tracing grid's box is a view the culling serves
        this._planRtGrid(stack, camera);
        this._runInstanceCulling(commandEncoder, stack, camera);
        this._runRtGrid(commandEncoder, stack);
        t?.end();
        t = cpuScope('scene/shadows');
        this._encodeShadowPasses(commandEncoder, stack);
        this._uploadShadowUniforms();
        t?.end();
        t = cpuScope('scene/voxel_gi');
        this._encodeVoxelGI(commandEncoder, stack, camera);
        t?.end();

        // The light clusters for the camera, over the GBuffer's pixels (the render size).
        t = cpuScope('scene/clusters');
        this._encodeLightClusters(commandEncoder, camera, gbuffer.width, gbuffer.height);
        this._updateShadowBindGroup();
        t?.end();

        // With transmissive objects the pass splits: opaque, a snapshot of the colour into
        // backgroundTexture, then transmissive and transparent on top.
        const hasTransmissive = stack.transmissive.length > 0;

        // Phase 2 — (re-)record the bundles whose draws changed.
        t = cpuScope('scene/bundles');
        const sets = [stack.opaque, stack.transmissive, stack.transparent];
        this._syncBundles(this._gbufferBundles, sets, stack, cameraBindGroup, targets);
        t?.end();
        const drawSets = (pass: GPURenderPassEncoder, which: DrawSet[]) => {
            for (const set of which) {
                this._drawSet(pass, this._gbufferBundles, set, sets[set], stack, cameraBindGroup, targets);
            }
        };

        // Phase 3 — execute into the GBuffer render pass(es).
        t = cpuScope('scene/gbuffer');

        const clearColor = {
            r: this.options.clearColor?.x || 0.0,
            g: this.options.clearColor?.y || 0.0,
            b: this.options.clearColor?.z || 0.0,
            a: this.options.clearColor?.w || 0.0,
        };
        const emissiveClear = { r: 0, g: 0, b: 0, a: 0 };
        const blackClear = { r: 0, g: 0, b: 0, a: 0 };

        // Build a pass descriptor configured for the given stage.
        // `loadExisting` = true for the second (transmissive) pass — it loads the
        // previously-rendered MSAA samples and continues drawing on top.
        const makePassDescriptor = (loadExisting: boolean): GPURenderPassDescriptor => {
            const colorLoad: GPULoadOp = loadExisting ? 'load' : 'clear';
            // Depth must be preserved in the MSAA path between pass 1 and pass 2,
            // AND kept until the depth-copy pass runs afterwards.
            const depthStoreOp: GPUStoreOp = 'store';

            if (gbuffer.msaaSampleCount > 1 && gbuffer.colorMSAATexture && gbuffer.depthMSAATexture) {
                // MSAA path: samples are kept between passes 1 and 2 via storeOp:'store'
                // and consumed on pass 2 via loadOp:'load'. Resolve happens at end of each pass.
                const colorStoreOp: GPUStoreOp = hasTransmissive && !loadExisting ? 'store' : 'discard';
                return {
                    colorAttachments: [
                        {
                            view: gbuffer.colorMSAATexture.createView(),
                            resolveTarget: gbuffer.colorTexture.createView(),
                            clearValue: clearColor,
                            loadOp: colorLoad,
                            storeOp: colorStoreOp,
                        },
                        {
                            view: gbuffer.emissiveMSAATexture!.createView(),
                            resolveTarget: gbuffer.emissiveTexture.createView(),
                            clearValue: emissiveClear,
                            loadOp: colorLoad,
                            storeOp: colorStoreOp,
                        },
                        {
                            view: gbuffer.normalMSAATexture!.createView(),
                            resolveTarget: gbuffer.normalTexture.createView(),
                            clearValue: blackClear,
                            loadOp: colorLoad,
                            storeOp: colorStoreOp,
                        },
                        {
                            view: gbuffer.albedoMSAATexture!.createView(),
                            resolveTarget: gbuffer.albedoTexture.createView(),
                            clearValue: blackClear,
                            loadOp: colorLoad,
                            storeOp: colorStoreOp,
                        },
                    ],
                    depthStencilAttachment: {
                        view: gbuffer.depthMSAATexture.createView(),
                        depthClearValue: 1.0,
                        depthLoadOp: loadExisting ? 'load' : 'clear',
                        depthStoreOp,
                    },
                };
            }

            // Non-MSAA path.
            return {
                colorAttachments: [
                    {
                        view: gbuffer.colorTexture.createView(),
                        clearValue: clearColor,
                        loadOp: colorLoad,
                        storeOp: 'store',
                    },
                    {
                        view: gbuffer.emissiveTexture.createView(),
                        clearValue: emissiveClear,
                        loadOp: colorLoad,
                        storeOp: 'store',
                    },
                    {
                        view: gbuffer.normalTexture.createView(),
                        clearValue: blackClear,
                        loadOp: colorLoad,
                        storeOp: 'store',
                    },
                    {
                        view: gbuffer.albedoTexture.createView(),
                        clearValue: blackClear,
                        loadOp: colorLoad,
                        storeOp: 'store',
                    },
                ],
                depthStencilAttachment: {
                    view: gbuffer.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: loadExisting ? 'load' : 'clear',
                    depthStoreOp,
                },
            };
        };

        if (hasTransmissive) {
            // Pass 1 — opaque objects.
            const opaquePass = commandEncoder.beginRenderPass({ ...makePassDescriptor(false), label: 'Renderer/GBufferOpaquePass', timestampWrites: gpuPass('Renderer/GBufferOpaquePass') });
            drawSets(opaquePass, [DrawSet.Opaque]);
            opaquePass.end();

            // Snapshot the opaque-only colour into backgroundTexture so that a
            // downstream transmission effect can sample the undistorted background.
            commandEncoder.copyTextureToTexture(
                { texture: gbuffer.colorTexture },
                { texture: gbuffer.backgroundTexture },
                { width: gbuffer.width, height: gbuffer.height, depthOrArrayLayers: 1 },
            );

            // Pass 2 — transmissive then transparent objects (continues drawing on top of the opaque result).
            const transmissivePass = commandEncoder.beginRenderPass({ ...makePassDescriptor(true), label: 'Renderer/GBufferIndirectPass', timestampWrites: gpuPass('Renderer/GBufferIndirectPass') });
            drawSets(transmissivePass, [DrawSet.Transmissive, DrawSet.Transparent]);
            transmissivePass.end();
        } else {
            const pass = commandEncoder.beginRenderPass({ ...makePassDescriptor(false), label: 'Renderer/GBufferOpaquePass', timestampWrites: gpuPass('Renderer/GBufferOpaquePass') });
            drawSets(pass, [DrawSet.Opaque, DrawSet.Transparent]);
            pass.end();
        }

        // Depth-copy pass: resolve MSAA depth → non-MSAA depthTexture for compute shaders.
        if (gbuffer.msaaSampleCount > 1 && gbuffer.depthMSAATexture) {
            this._ensureDepthCopyPipeline();
            const depthCopyPass = commandEncoder.beginRenderPass({
                label: 'DepthCopy',
                colorAttachments: [],
                depthStencilAttachment: {
                    view: gbuffer.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
                timestampWrites: gpuPass('DepthCopy'),
            });
            depthCopyPass.setPipeline(this._depthCopyPipeline!);
            depthCopyPass.setBindGroup(0, this._getDepthCopyBindGroup(gbuffer.depthMSAATexture));
            depthCopyPass.draw(3);
            depthCopyPass.end();
        }

        // Motion vectors of the materials that write them, against the GBuffer depth; the rest
        // of the velocity texture keeps NO_VELOCITY.
        this._drawVelocity(commandEncoder, stack, cameraBindGroup, gbuffer);
        t?.end();

        t = cpuScope('scene/submit');
        this.device!.queue.submit([commandEncoder.finish()]);
        this._rtGrid?.afterSubmit();
        this._endCulledFrame(camera);
        t?.end();
        sceneScope?.end();
    }

    /**
     * The velocity pass (Rust `draw_velocity`): clears the GBuffer's velocity texture to
     * `GBuffer.NO_VELOCITY`, then redraws the visible non-transparent renderables whose material
     * has `outputsVelocity` with its velocity pipeline (the same shader, only @location(4) kept),
     * depth-tested against the GBuffer's single-sample depth. Drawn live, at their scene slots.
     */
    private _drawVelocity(encoder: GPUCommandEncoder, stack: Scene, cameraBindGroup: GPUBindGroup, gbuffer: GBuffer): void {
        const noVelocity = GBuffer.NO_VELOCITY;
        const colorAttachments: (GPURenderPassColorAttachment | null)[] = new Array(GBuffer.VELOCITY_TARGET).fill(null);
        colorAttachments.push({
            view: gbuffer.velocityTexture.createView(),
            clearValue: { r: noVelocity, g: noVelocity, b: 0, a: 0 },
            loadOp: 'clear',
            storeOp: 'store',
        });
        const pass = encoder.beginRenderPass({
            label: 'Renderer/VelocityPass',
            colorAttachments,
            depthStencilAttachment: {
                view: gbuffer.depthTexture.createView(),
                depthLoadOp: 'load',
                depthStoreOp: 'store',
            },
            timestampWrites: gpuPass('Renderer/VelocityPass'),
        });
        pass.setBindGroup(BindGroupSlot.Camera, cameraBindGroup);
        if (this._shadowBG) pass.setBindGroup(BindGroupSlot.Shadow, this._shadowBG);
        const state: DrawState = { pipeline: null, materialBindGroup: null, indexBuffer: null, vertexBuffer: null };
        for (const set of [stack.opaque, stack.transmissive]) {
            for (const renderable of set) {
                if (!renderable.material.outputsVelocity || !renderable.geometry.initialized) continue;
                const pipeline = renderable.material.getVelocityPipeline(this.device!, renderable.geometry.vertexBuffersDescriptors);
                this._encodeDraw(pass, renderable, stack.slotOf(renderable), null, state, pipeline);
            }
        }
        pass.end();
    }

    /**
     * Places the shadow views of a frame before its culling, which culls for them too: the
     * cascades fit the camera for the scene's first directional light if it has `castShadow`
     * (Rust's `update_cascaded_shadows`), else are off this frame; the directional map the
     * renderer owns follows the first directional light with `castShadow` (else the first area
     * light with it), or has no light this frame (nor while the cascades are on); the scene's
     * spot lights are packed and uploaded, the first `castShadow` ones taking the spot atlas'
     * layers.
     */
    private _planShadowViews(stack: Scene, camera: Camera): void {
        const csm = this._cascadedShadowMap;
        if (csm) {
            const sun = stack.directionalLights[0];
            if (sun?.castShadow) {
                const eye = camera.inverseViewMatrix.internalMat4;
                csm.fit(camera, sun.direction);
                csm.upload(sun.direction, sun.effectiveColor, [eye[12], eye[13], eye[14]]);
            } else {
                csm.disable();
            }
        }
        this._skyOcclusion?.update(camera);
        this._uploadSpotLights(stack);
        const shadowMap = this._ownsShadowMap ? this._shadowMap : null;
        if (!shadowMap) return;
        const light = csm ? null : stack.directionalLights.find((l) => l.castShadow)
            ?? stack.areaLights.find((l) => l.castShadow);
        if (light) shadowMap.update(camera, light);
        else shadowMap.light = null;
    }

    /**
     * The shadow views of a frame, in the Rust renderer's order (`renderer.rs` `render`): after
     * the matrix upload and the culling, before the scene pass, into the frame's encoder. The
     * directional map (placed by `_planShadowViews`) draws the instances culled for its light,
     * the cube map the point lights with `castShadow` (every instance), each spot atlas layer the
     * instances culled for its light, the cascades the instances culled for each cascade; each
     * draws the visible `castShadow` renderables at their scene slots of the shared mesh buffers,
     * with the light as the camera (group 1); then the sky occlusion's tile while it rebuilds.
     * Maps the caller binds through the setters are left to the caller.
     */
    private _encodeShadowPasses(encoder: GPUCommandEncoder, stack: Scene): void {
        const objects = stack.getOrderedObjects();
        const meshOffset = (r: Renderable) => stack.slotOf(r) * this._matrixAlignment;

        const shadowMap = this._ownsShadowMap ? this._shadowMap : null;
        if (shadowMap?.light) {
            shadowMap.encode(encoder, objects, this._sharedMeshBG!, meshOffset, SHADOW_VIEW);
        }

        const cubeMap = this._ownsCubeMapShadowMap ? this._cubeMapShadowMap : null;
        if (cubeMap && cubeMap.update(stack.pointLights.filter((l) => l.castShadow)) > 0) {
            cubeMap.encode(encoder, objects, this._sharedMeshBG!, meshOffset);
            // Materials sample the first light's faces (kansei_point_shadow).
            const first = cubeMap.lights[0];
            const wm = first.worldMatrix.internalMat4;
            this.setPointShadowParams(wm[12], wm[13], wm[14], first.radius);
        }

        const atlas = this._spotShadowAtlas;
        if (atlas && this._spotLights.shadows.length > 0) {
            atlas.encode(encoder, this._spotLights.shadows, objects, this._sharedMeshBG!, meshOffset, this._spotViewBase());
        }

        // the sun's cascades
        const csm = this._cascadedShadowMap;
        if (csm && csm.slots.length > 0) {
            csm.encode(encoder, objects, this._sharedMeshBG!, meshOffset, (c) => this._cascadeView(c));
        }

        this._encodeSkyOcclusionPass(encoder, objects, meshOffset);
    }

    /**
     * While the sky occlusion is being rebuilt: a tile of the top-down pass a frame (the visible
     * shadow casters on its layers seen from above, through their materials' depth pipelines,
     * culled to that tile), then its pyramid and its volume's slabs (`SkyOcclusion.build`). Rust:
     * `run_sky_occlusion_pass`. Renderables with `shadowVertexCode` are left out: their material's
     * own vertex stage is not what they cast with.
     */
    private _encodeSkyOcclusionPass(encoder: GPUCommandEncoder, objects: readonly Renderable[], meshOffset: (r: Renderable) => number): void {
        const sky = this._skyOcclusion;
        if (!sky?.building) return;
        const scissor = sky.tileScissor();
        if (scissor) {
            const device = this.device!;
            const pass = encoder.beginRenderPass({
                label: 'Renderer/SkyOcclusionPass',
                timestampWrites: gpuPass('Renderer/SkyOcclusionPass'),
                colorAttachments: [],
                depthStencilAttachment: {
                    view: sky.depthView,
                    depthClearValue: 1.0,
                    depthLoadOp: sky.firstTile ? 'clear' : 'load',
                    depthStoreOp: 'store',
                },
            });
            pass.setScissorRect(...scissor);
            pass.setBindGroup(1, sky.cameraBindGroup);
            const view = this._skyOcclusionView();
            for (const r of objects) {
                if (!r.castShadow || !r.geometry.initialized || r.shadowVertexCode || (r.layers & sky.options.layerMask) === 0) continue;
                const pipeline = r.material.getDepthPipeline(device, r.geometry.vertexBuffersDescriptors, SkyOcclusion.FORMAT, SkyOcclusion.DEPTH_BIAS);
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, r.material.getBindGroup(device));
                const offset = meshOffset(r);
                pass.setBindGroup(2, this._sharedMeshBG!, [offset, offset]);
                pass.setVertexBuffer(0, r.geometry.vertexBuffer!);
                pass.setIndexBuffer(r.geometry.indexBuffer!, r.geometry.indexFormat!);
                drawGeometry(pass, r.geometry, r.instanceCulling?.view(view) ?? null);
            }
            pass.end();
        }
        sky.build(encoder);
    }

    /** The cull view of the sky occlusion's top-down view: after the cascades'. */
    private _skyOcclusionView(): number {
        return this._cascadeView(this._cascadedShadowMap?.slots.length ?? 0);
    }

    /**
     * Packs the scene's spot lights (the first `castShadow` ones get the atlas layers) and
     * uploads them, skipping the write while the scene has none and the buffer says so.
     */
    private _uploadSpotLights(stack: Scene): void {
        this._ensureShadowResources();
        const atlas = this._spotShadowAtlas;
        this._spotLights.pack(stack.spotLights, atlas?.layers ?? 0, atlas?.resolution ?? 0);
        if (this._spotLights.count === 0 && this._spotLightsUploaded === 0) return;
        this.device!.queue.writeBuffer(this._spotLightsBuf!, 0, this._spotLights.bytes);
        this._spotLightsUploaded = this._spotLights.count;
    }

    /**
     * Builds the light clusters for `camera` over a `width` x `height` target (Rust's
     * `LightClusters::build`), or shades with every light while clustering is off. Nothing to
     * build without spot lights; the clusters (and their buffers) are created by the first frame
     * with some.
     */
    private _encodeLightClusters(encoder: GPUCommandEncoder, camera: Camera, width: number, height: number): void {
        if (!this._clusteredLights || this._spotLights.count === 0) {
            this._lightClusters?.disable();
            return;
        }
        if (!this._lightClusters) {
            this._lightClusters = new LightClusters(this.device!, this._spotLightsBuf!);
            this._shadowBGDirty = true;
        }
        this._lightClusters.encode(encoder, camera, width, height);
    }

    /** The cull view of spot atlas layer 0 (layer l culls at `_spotViewBase() + l`), after the sky occlusion's. */
    private _spotViewBase(): number {
        return this._skyOcclusionView() + (this._skyOcclusion ? 1 : 0);
    }

    /**
     * The views instance culling runs for, at fixed indices: `MAIN_VIEW`, then while the renderer
     * owns the directional shadow map `SHADOW_VIEW` (`null` when no light casts this frame),
     * then this frame's cascades (`_cascadeView`), then with sky occlusion `_skyOcclusionView`
     * (`null` unless a tile of its top-down pass is due), then every layer of the spot shadow
     * atlas from `_spotViewBase()` (`null` when no light uses it), then with a ray tracing grid
     * its box (`_rtView`).
     */
    private _cullViews(camera: Camera): (CullView | null)[] {
        // the unjittered projection: the frustum, not where pixels sample
        const views: (CullView | null)[] = [cullView(camera.viewProjection(this._viewProjScratch))];
        if (this._ownsShadowMap && this._shadowMap) {
            const map = this._shadowMap;
            views.push(map.light ? cullView(map.lightViewProjMatrix, { castersOnly: true }) : null);
        }
        this._cascadedShadowMap?.slots.forEach((slot, c) => {
            const viewProj = mat4.multiply(this._cascadeViewProjs[c], slot.projection, slot.view);
            views.push(cullView(viewProj, { castersOnly: true }));
        });
        // then the sky occlusion's top-down view (`_skyOcclusionView`), while it is being rebuilt
        const sky = this._skyOcclusion;
        if (sky) {
            const viewProj = sky.cullView();
            views.push(viewProj ? cullView(viewProj, { castersOnly: true, layerMask: sky.options.layerMask, lodDistanceScale: sky.options.lodDistanceScale }) : null);
        }
        if (this._spotShadowAtlas) {
            const base = views.length;
            for (let l = 0; l < this._spotShadowAtlas.layers; l++) views.push(null);
            for (const slot of this._spotLights.shadows) views[base + slot.layer] = cullView(slot.viewProj, { castersOnly: true });
        }
        // then the ray tracing grid's box (`_rtView`)
        if (this._rtGrid) views.push(cullView(this._rtGrid.cullViewProj(this._rtViewProj), { rt: true }));
        return views;
    }

    /** The cull view of the ray tracing grid's box, after the spot atlas's layers (Rust's order). */
    private _rtView(): number {
        return this._spotViewBase() + (this._spotShadowAtlas?.layers ?? 0);
    }

    /** Cull view of cascade `index`, after the camera and the directional map's. */
    private _cascadeView(index: number): number {
        return 1 + (this._ownsShadowMap && this._shadowMap ? 1 : 0) + index;
    }

    /** What each cull view is, in `_cullViews` order (for the stats). */
    private _cullViewKinds(): CullViewKind[] {
        const kinds: CullViewKind[] = ['camera'];
        if (this._ownsShadowMap && this._shadowMap) kinds.push('shadow');
        this._cascadedShadowMap?.slots.forEach((_, c) => kinds.push(`cascade${c}`));
        if (this._skyOcclusion) kinds.push('skyOcclusion');
        for (let l = 0; l < (this._spotShadowAtlas?.layers ?? 0); l++) kinds.push('spot');
        if (this._rtGrid) kinds.push('rtGrid');
        return kinds;
    }

    /**
     * Culls every visible renderable with `instanceCulling` for every view that draws it, into
     * `encoder` (after the frame's uploads and `_planShadowViews`, before its shadow and main
     * passes): the frame's views in one write, then one compute pass, a dispatch per renderable
     * (per chunk of views). Buffers made for more views re-record the cached bundles.
     */
    private _runInstanceCulling(encoder: GPUCommandEncoder, stack: Scene, camera: Camera): void {
        this._cullStats.beginFrame(this._cullViewKinds());
        const culled = stack.getOrderedObjects().filter((r) => r.instanceCulling && r.geometry.initialized && r.geometry.isInstancedGeometry);
        if (culled.length === 0) return;
        const device = this.device!;
        const views = this._cullViews(camera);
        const pipeline = this._cullPipeline ??= new CullPipeline(device);
        // the frame's views, in one write; distances are measured from the camera in every view
        if (this._cullViewBytes.byteLength < views.length * CULL_VIEW_BYTES) this._cullViewBytes = new ArrayBuffer(views.length * CULL_VIEW_BYTES);
        const eye = camera.inverseViewMatrix.internalMat4;
        const lodOrigin = [eye[12], eye[13], eye[14]];
        views.forEach((view, k) => packCullView(this._cullViewBytes, k * CULL_VIEW_BYTES, view, lodOrigin, this._cullStats.enabled));
        pipeline.setViews(this._cullViewBytes, views.length);

        let staleBundles = false;
        for (const r of culled) {
            const culling = r.instanceCulling!;
            staleBundles = culling.ensureViews(device, pipeline.layout, views.length) || staleBundles;
            culling.beginFrame(device.queue, encoder, r.worldMatrix.internalMat4, r.geometry.vertexCount, r.castShadow, r.layers, r.gi !== null, r.rt !== null);
        }
        const pass = encoder.beginComputePass({ label: 'Renderer/InstanceCulling', timestampWrites: gpuPass('Renderer/InstanceCulling') });
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(1, pipeline.viewBindGroup);
        // a renderable's views in one dispatch; the shader skips those it is not drawn in
        for (const r of culled) {
            const culling = r.instanceCulling!;
            culling.dispatch(pass);
            views.forEach((view, slot) => {
                if (!view || !cullViewDraws(view, r.castShadow, r.layers, r.gi !== null, r.rt !== null)) return;
                const draw = culling.view(slot)!;
                this._cullStats.record(slot, culling.tested, draw.args, draw.offset);
            });
        }
        pass.end();
        if (staleBundles) this.invalidateBundle();
    }

    /**
     * Plan the ray tracing grid's frame before the culling, whose view its box is: follow the
     * camera, and whether to rebuild.
     */
    private _planRtGrid(stack: Scene, camera: Camera): void {
        if (!this._rtGrid) return;
        const eye = camera.inverseViewMatrix.internalMat4;
        this._rtGrid.plan(stack, [eye[12], eye[13], eye[14]]);
    }

    /** The ray tracing grid's rebuild, after the culling (its view's culled instances). */
    private _runRtGrid(encoder: GPUCommandEncoder, stack: Scene): void {
        if (!this._rtGrid?.rebuilding) return;
        this._rtGrid.build(encoder, stack, this._rtView());
    }

    /** After the frame's culling is submitted: read its statistics back (when on). */
    private _endCulledFrame(camera: Camera): void {
        this._cullStats.endFrame(this.device!, camera.frame);
    }

    /**
     * Voxel GI's frame (`enableVoxelGI`), after the shadow views and before the scene pass, in
     * the Rust renderer's order: the GI renderables voxelized at their scene slots, lit through
     * the shadow maps materials sample (the cascades' widest when they are on, else the
     * directional one while it has a light) and the spot lights with their atlas, the mips rebuilt, then its distance field and probes (the
     * probes around `camera`).
     */
    private _encodeVoxelGI(encoder: GPUCommandEncoder, stack: Scene, camera: Camera): void {
        if (!this._voxelGI) return;
        const spots = this._voxelGISpots;
        if (spots?.gi !== this._voxelGI || spots.atlas !== this._spotShadowAtlas) {
            this._ensureShadowResources();
            this._voxelGI.injection.shadows.setSpotLights(this._spotLightsBuf, this._spotShadowAtlas?.arrayView ?? null);
            this._voxelGISpots = { gi: this._voxelGI, atlas: this._spotShadowAtlas };
        }
        const sm = this._shadowMap;
        // the cascades' widest when they are on (Rust's `sync_voxel_gi_shadows`)
        const shadowMap = this._cascadedShadowMap
            ?? (sm && this.shadowsEnabled && (!this._ownsShadowMap || sm.light !== null) ? sm : null);
        const eye = camera.inverseViewMatrix.internalMat4;
        this._voxelGI.encode(encoder, stack, this._sharedMeshBG!, (r) => stack.slotOf(r) * this._matrixAlignment,
            shadowMap, this._cubeMapShadowMap, [eye[12], eye[13], eye[14]]);
    }

    /**
     * Uploads the shadow uniform (group 3 binding 2): the directional map's view-projection and
     * biases while materials sample it, and the point shadow's light.
     */
    private _uploadShadowUniforms(): void {
        this._ensureShadowResources();
        // 24 floats: mat4(16) + bias(1) + normalBias(1) + shadowEnabled(1) + pointShadowEnabled(1)
        //          + pointLightPos(3) + pointShadowFar(1)
        const staging = new Float32Array(24);
        const sm = this._shadowMap;
        const csm = this._cascadedShadowMap;
        const far = csm?.farViewProjection();
        if (csm) {
            // the widest cascade, for shaders that read the single map (off when no sun casts)
            if (far) {
                staging.set(far, 0);
                staging[16] = 0.0005;  // bias
                staging[17] = 2 * csm.options.maxDistance / csm.options.resolution;  // normal bias
                staging[18] = 1.0;     // shadowEnabled
            }
        } else if (sm && this.shadowsEnabled && (!this._ownsShadowMap || sm.light !== null)) {
            // A map the renderer owns has nothing to show when no light casts.
            staging.set(sm.lightViewProjMatrix, 0);
            staging[16] = sm.bias;
            staging[17] = sm.normalBias;
            staging[18] = 1.0;    // shadowEnabled
        } else {
            staging[18] = 0.0;    // shadowEnabled = off
        }
        // Point shadow: [enabled, posX, posY, posZ, shadowFar] — set each frame
        staging.set(this._pointShadowParams, 19);
        this.device!.queue.writeBuffer(this._shadowUniformBuf!, 0, staging);
        // Reset: the renderer sets it again for its own cube map, a caller-driven one every frame
        this._pointShadowParams.fill(0);
    }

    /** Point shadow params: [enabled, posX, posY, posZ, shadowFar]. Reset after each upload. */
    private _pointShadowParams = new Float32Array(5);

    /**
     * Sets the point light whose cube shadow materials sample this frame (`kansei_point_shadow`).
     * Only for a cube map the caller renders (the `cubeMapShadowMap` setter); the renderer sets
     * it for its own (`enablePointShadows`).
     */
    public setPointShadowParams(posX: number, posY: number, posZ: number, shadowFar: number): void {
        this._pointShadowParams[0] = 1.0;  // enabled
        this._pointShadowParams[1] = posX;
        this._pointShadowParams[2] = posY;
        this._pointShadowParams[3] = posZ;
        this._pointShadowParams[4] = shadowFar;
    }

    /** The pipeline of `renderable`'s material for a pass with `targets`. */
    private _pipelineFor(renderable: Renderable, targets: PassTargets): GPURenderPipeline {
        const formats = targets.colorFormats;
        return renderable.material.getPipelineForConfig(
            this.device!,
            renderable.geometry.vertexBuffersDescriptors,
            formats[0],
            targets.sampleCount,
            targets.depthFormat,
            formats.length,
            formats.length > 1 ? formats : undefined
        );
    }

    /**
     * Re-records the bundle of each of a pass's draw sets whose bundled draws changed since it was
     * recorded, and all of them when the shared bind groups or the targets did.
     */
    private _syncBundles(
        cache: PassBundles,
        sets: readonly (readonly Renderable[])[],
        stack: Scene,
        cameraBindGroup: GPUBindGroup,
        targets: PassTargets,
    ) {
        const shared = this._sharedKeyScratch;
        shared.length = 0;
        shared.push(cameraBindGroup, this._sharedMeshBG, this._shadowBG, targets.depthFormat, targets.sampleCount, ...targets.colorFormats);
        if (!sameKey(cache.shared, shared)) {
            cache.invalidate();
            this._sharedKeyScratch = cache.shared;
            cache.shared = shared;
        }

        for (let set = 0; set < cache.sets.length; set++) {
            const entry = cache.sets[set];
            const key = this._bundleKeyScratch;
            key.length = 0;
            for (const r of sets[set]) {
                if (!isBundled(r)) continue;
                const geo = r.geometry;
                key.push(r, stack.slotOf(r), r.material, r.material.currentBindGroup, geo, geo.initialized, geo.vertexCount,
                    geo.isInstancedGeometry ? (geo as InstancedGeometry).instanceCount : 1, r.instanceCulling);
            }
            if (entry.valid && sameKey(entry.key, key)) continue;
            entry.bundle = this._recordBundle(sets[set], stack, cameraBindGroup, targets);
            this._bundleKeyScratch = entry.key;
            entry.key = key;
            entry.valid = true;
        }
    }

    /**
     * Records the draws of the bundled renderables (`isBundled`) among `renderables` into a
     * GPURenderBundle, each at its scene slot; `null` when there is none to draw.
     */
    private _recordBundle(
        renderables: readonly Renderable[],
        stack: Scene,
        cameraBindGroup: GPUBindGroup,
        targets: PassTargets,
    ): GPURenderBundle | null {
        const encoder = this.device!.createRenderBundleEncoder({
            colorFormats: targets.colorFormats,
            depthStencilFormat: targets.depthFormat,
            sampleCount: targets.sampleCount,
        });

        // Camera and shadow bind groups are the same for every object — set once.
        encoder.setBindGroup(BindGroupSlot.Camera, cameraBindGroup);
        if (this._shadowBG) {
            encoder.setBindGroup(BindGroupSlot.Shadow, this._shadowBG);
        }

        const state: DrawState = { pipeline: null, materialBindGroup: null, indexBuffer: null, vertexBuffer: null };
        let draws = 0;
        for (const renderable of renderables) {
            if (isBundled(renderable) && this._encodeDraw(encoder, renderable, stack.slotOf(renderable), targets, state)) draws++;
        }
        this._bundleRecords++;
        return draws > 0 ? encoder.finish() : null;
    }

    /**
     * Draws one of a pass's draw sets: its cached bundle, then its visible dynamic and indirect
     * renderables live, so each frame's draw reads that frame's matrices and draw counts.
     */
    private _drawSet(
        pass: GPURenderPassEncoder,
        cache: PassBundles,
        set: DrawSet,
        renderables: readonly Renderable[],
        stack: Scene,
        cameraBindGroup: GPUBindGroup,
        targets: PassTargets,
    ) {
        const bundle = cache.sets[set].bundle;
        if (bundle) pass.executeBundles([bundle]);

        // Executing a bundle clears the pass's state: bind the shared groups again.
        let state: DrawState | null = null;
        for (const renderable of renderables) {
            if (isBundled(renderable)) continue;
            if (!state) {
                pass.setBindGroup(BindGroupSlot.Camera, cameraBindGroup);
                if (this._shadowBG) pass.setBindGroup(BindGroupSlot.Shadow, this._shadowBG);
                state = { pipeline: null, materialBindGroup: null, indexBuffer: null, vertexBuffer: null };
            }
            this._encodeDraw(pass, renderable, stack.slotOf(renderable), targets, state);
        }
    }

    /**
     * Encodes the draw of `renderable` with its matrices at `slot`, setting only the state that
     * differs from `state`: with its material's pipeline for `targets`, or with `pipeline`.
     * Returns false when it cannot be drawn yet.
     */
    private _encodeDraw(
        encoder: GPURenderPassEncoder | GPURenderBundleEncoder,
        renderable: Renderable,
        slot: number,
        targets: PassTargets | null,
        state: DrawState,
        pipeline?: GPURenderPipeline,
    ): boolean {
        const geometry = renderable.geometry;
        if (!geometry.initialized) return false;
        pipeline ??= this._pipelineFor(renderable, targets!);

        if (pipeline !== state.pipeline) {
            encoder.setPipeline(pipeline);
            state.pipeline = pipeline;
            state.materialBindGroup = null;
        }

        if (geometry.indexBuffer !== state.indexBuffer) {
            encoder.setIndexBuffer(geometry.indexBuffer!, geometry.indexFormat!);
            state.indexBuffer = geometry.indexBuffer!;
        }

        if (geometry.vertexBuffer !== state.vertexBuffer) {
            encoder.setVertexBuffer(0, geometry.vertexBuffer!);
            state.vertexBuffer = geometry.vertexBuffer!;
        }

        // Phase 1 updated it this frame (`_updateRenderables`).
        const materialBindGroup = renderable.material.currentBindGroup!;
        if (materialBindGroup !== state.materialBindGroup) {
            encoder.setBindGroup(BindGroupSlot.Material, materialBindGroup);
            state.materialBindGroup = materialBindGroup;
        }

        // Both bindings (normalMatrix, world + previous world) live in separate
        // buffers but share the same stride.
        const offset = slot * this._matrixAlignment;
        encoder.setBindGroup(BindGroupSlot.Mesh, this._sharedMeshBG!, [offset, offset]);

        // The instances culled for the camera, if culled (indirect: the count changes, the bundle not).
        drawGeometry(encoder, geometry, renderable.instanceCulling?.view(MAIN_VIEW) ?? null);
        return true;
    }

    /**
     * Executes a compute shader with the specified workgroup configuration.
     * @async
     * @param {Compute} compute - The compute shader to execute
     * @param {number} [workgroupsX=64] - Number of workgroups in X dimension
     * @param {number} [workgroupsY=1] - Number of workgroups in Y dimension
     * @param {number} [workgroupsZ=1] - Number of workgroups in Z dimension
     * @returns {Promise<void>}
     */
    public async compute(compute: Compute, workgroupsX: number = 64, workgroupsY: number = 1, workgroupsZ: number = 1): Promise<void> {
        const commandEncoder = this.device!.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass({ label: 'ComputeBatch/Pass', timestampWrites: gpuPass('ComputeBatch/Pass') });
        if (!compute.initialized) {
            compute.initialize(this.device!);
        }
        const bindGroup = compute.getBindGroup(this.device!);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.setPipeline(compute.pipeline!);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
        passEncoder.end();
        const commands = commandEncoder.finish();

        this.device!.queue.submit([commands]);

        // Wait for the compute work to complete
        return this.device!.queue.onSubmittedWorkDone();
    }

    /**
     * Executes multiple compute shaders in a single command buffer submission.
     * Storage buffer writes in pass N are visible to pass N+1 (WebGPU guarantee).
     * @param passes Array of compute passes with workgroup dimensions
     * @returns Promise that resolves when all passes complete
     */
    public async computeBatch(passes: { compute: Compute, workgroupsX: number, workgroupsY?: number, workgroupsZ?: number }[]): Promise<void> {
        const commandEncoder = this.device!.createCommandEncoder();
        for (const pass of passes) {
            if (!pass.compute.initialized) {
                pass.compute.initialize(this.device!);
            }
            const passEncoder = commandEncoder.beginComputePass({ label: 'ComputeBatch/Pass', timestampWrites: gpuPass('ComputeBatch/Pass') });
            passEncoder.setBindGroup(0, pass.compute.getBindGroup(this.device!));
            passEncoder.setPipeline(pass.compute.pipeline!);
            passEncoder.dispatchWorkgroups(pass.workgroupsX, pass.workgroupsY ?? 1, pass.workgroupsZ ?? 1);
            passEncoder.end();
        }
        this.device!.queue.submit([commandEncoder.finish()]);
        return this.device!.queue.onSubmittedWorkDone();
    }

    /**
     * Reads data back from a compute buffer.
     * @async
     * @template T
     * @param {ComputeBuffer} buffer - The buffer to read from
     * @param {new (buffer: ArrayBuffer) => T} ArrayType - The type of array to create
     * @returns {Promise<T>} The buffer data
     */
    public async readBackBuffer<T extends Float32Array | Uint32Array | Int32Array>(
        buffer: ComputeBuffer,
        ArrayType: new (buffer: ArrayBuffer) => T
    ): Promise<T> {
        const stagingBuffer = this.device!.createBuffer({
            size: buffer.resource.buffer!.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const commandEncoder = this.device!.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            buffer.resource.buffer,
            0, // Source offset
            stagingBuffer,
            0, // Destination offset
            buffer.resource.buffer!.size
        );

        this.device!.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(
            GPUMapMode.READ,
            0, // Offset
            buffer.resource.buffer!.size // Length
        );
        const copyArrayBuffer = stagingBuffer.getMappedRange(0, buffer.resource.buffer!.size);
        const data = new ArrayType(copyArrayBuffer.slice(0));
        stagingBuffer.unmap();

        return data;
    }
}

export { Renderer };
