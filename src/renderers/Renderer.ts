import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Camera } from "../cameras/Camera";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { Vector4 } from "../main";
import { Compute } from "../materials/Compute";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { GBuffer } from "../postprocessing/GBuffer";
import { ShadowMap } from "../shadows/ShadowMap";
import { CubeMapShadowMap } from "../shadows/CubeMapShadowMap";
import { BindGroupSlot, MESH_TRANSFORMS_BYTES, meshBindGroupLayoutEntries, meshSlotStride } from "./SharedLayouts";
import { FrameProfile, cpuScope, endProfiledFrame, gpuPass, setProfilingEnabled, takeProfile } from "../profiling/Profiler";

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

    // ── Public read-only accessors ──────────────────────────────────────────
    /** The initialised GPU device. Undefined before initialize() resolves. */
    public get gpuDevice(): GPUDevice { return this.device!; }
    /** Canvas colour format negotiated with the platform. */
    public get presentationFormat(): GPUTextureFormat { return this._presentationFormat!; }
    /** Render width in physical pixels (includes devicePixelRatio). */
    public get renderWidth(): number { return this.width; }
    /** Render height in physical pixels (includes devicePixelRatio). */
    public get renderHeight(): number { return this.height; }
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

    // ── Shadow resources ─────────────────────────────────────────────────────
    public shadowsEnabled: boolean = false;
    private _shadowMap: ShadowMap | null = null;
    private _shadowBGL: GPUBindGroupLayout | null = null;
    private _shadowBG: GPUBindGroup | null = null;
    private _shadowUniformBuf: GPUBuffer | null = null;
    private _shadowComparisonSampler: GPUSampler | null = null;
    private _dummyShadowDepthTex: GPUTexture | null = null;
    private _dummyCubeShadowTex: GPUTexture | null = null;
    private _cubeShadowSampler: GPUSampler | null = null;
    private _cubeMapShadowMap: CubeMapShadowMap | null = null;
    private _shadowBGDirty: boolean = true;

    public get shadowMap(): ShadowMap | null { return this._shadowMap; }
    public set shadowMap(value: ShadowMap | null) {
        this._shadowMap = value;
        this._shadowBGDirty = true;
    }

    public get cubeMapShadowMap(): CubeMapShadowMap | null { return this._cubeMapShadowMap; }
    public set cubeMapShadowMap(value: CubeMapShadowMap | null) {
        this._cubeMapShadowMap = value;
        this._shadowBGDirty = true;
    }

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
     * Creates the shadow GPU resources (dummy depth texture, comparison sampler,
     * uniform buffer, bind group layout) on first use.
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
            size: 96,
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

        this._shadowBGL = this.device!.createBindGroupLayout({
            label: 'Shadow BindGroupLayout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
            ],
        });

        this._shadowBGDirty = true;
    }

    /**
     * Creates or recreates the shadow bind group when the shadow map texture changes.
     */
    private _updateShadowBindGroup(): void {
        if (!this._shadowBGDirty) return;
        this._ensureShadowResources();

        const depthTex = this._shadowMap
            ? this._shadowMap.depthTexture
            : this._dummyShadowDepthTex!;

        const cubeTex = this._cubeMapShadowMap
            ? this._cubeMapShadowMap.distanceTexture
            : this._dummyCubeShadowTex!;

        this._shadowBG = this.device!.createBindGroup({
            label: 'Shadow BindGroup',
            layout: this._shadowBGL!,
            entries: [
                { binding: 0, resource: depthTex.createView() },
                { binding: 1, resource: this._shadowComparisonSampler! },
                { binding: 2, resource: { buffer: this._shadowUniformBuf! } },
                { binding: 3, resource: cubeTex.createView({ dimension: '2d-array' }) },
                { binding: 4, resource: this._cubeShadowSampler! },
            ],
        });

        this._shadowBGDirty = false;
    }

    /**
     * Renders a stack using the specified camera.
     *
     * Each frame has three phases:
     *  1. Update — compute the visible renderables' matrices on the CPU, copy them into
     *     their scene slots of the staging arrays, then upload via exactly 2 writeBuffer calls.
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

        const cameraBindGroup = camera.getBindGroup(this.device!);
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

        // Upload shadow uniforms.
        this._uploadShadowUniforms();
        this._updateShadowBindGroup();

        // Phase 2 — (re-)record the bundles whose draws changed.
        const sets = [stack.opaque, stack.transmissive, stack.transparent];
        this._syncBundles(this._canvasBundles, sets, stack, cameraBindGroup, targets);

        // Phase 3 — execute the bundles and the live draws in a fresh render pass.
        const commandRenderEncoder = this.device!.createCommandEncoder();
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
        endProfiledFrame();
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
     * It performs the same three-phase matrix-upload / bundle-record / execute loop but
     * targets the GBuffer's rgba16float colour texture and depth32float depth texture at
     * sampleCount=1 (no MSAA — post-processing handles aliasing via FXAA etc.).
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

        const cameraBindGroup = camera.getBindGroup(this.device!);

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
            // Mark initialized to skip initialize() which would build a
            // canvas-format pipeline that fails for shaders with @location(1).
            renderable.material.initialized = true;
        });

        // Upload shadow uniforms.
        this._uploadShadowUniforms();
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
        const commandEncoder = this.device!.createCommandEncoder();

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
        t?.end();

        t = cpuScope('scene/submit');
        this.device!.queue.submit([commandEncoder.finish()]);
        t?.end();
        sceneScope?.end();
    }

    /**
     * Uploads shadow uniform data (light VP matrix, bias values, enabled flag)
     * to the GPU buffer every frame.
     */
    private _uploadShadowUniforms(): void {
        this._ensureShadowResources();
        // 24 floats: mat4(16) + bias(1) + normalBias(1) + shadowEnabled(1) + pointShadowEnabled(1)
        //          + pointLightPos(3) + pointShadowFar(1)
        const staging = new Float32Array(24);
        if (this._shadowMap && this.shadowsEnabled) {
            staging.set(this._shadowMap.lightViewProjMatrix, 0);
            staging[16] = 0.001;  // bias
            staging[17] = 0.02;   // normalBias
            staging[18] = 1.0;    // shadowEnabled
        } else {
            staging[18] = 0.0;    // shadowEnabled = off
        }
        // Point shadow: [enabled, posX, posY, posZ, shadowFar] — must be set each frame
        staging.set(this._pointShadowParams, 19);
        this.device!.queue.writeBuffer(this._shadowUniformBuf!, 0, staging);
        // Reset — caller must call setPointShadowParams every frame it wants point shadow
        this._pointShadowParams.fill(0);
    }

    /** Point shadow params: [enabled, posX, posY, posZ, shadowFar]. Reset after each upload. */
    private _pointShadowParams = new Float32Array(5);

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
                    geo.isInstancedGeometry ? (geo as InstancedGeometry).instanceCount : 1);
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
     * differs from `state`. Returns false when it cannot be drawn yet.
     */
    private _encodeDraw(
        encoder: GPURenderPassEncoder | GPURenderBundleEncoder,
        renderable: Renderable,
        slot: number,
        targets: PassTargets,
        state: DrawState,
    ): boolean {
        const geometry = renderable.geometry;
        if (!geometry.initialized) return false;
        const pipeline = this._pipelineFor(renderable, targets);

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

        if (geometry.isInstancedGeometry) {
            const geo = geometry as InstancedGeometry;
            let idx = 1;
            for (const extraBuffer of geo.extraBuffers) {
                encoder.setVertexBuffer(idx++, extraBuffer.resource.buffer);
            }
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

        if (geometry.isInstancedGeometry) {
            const geo = geometry as InstancedGeometry;
            encoder.drawIndexed(geo.vertexCount, geo.instanceCount, 0, 0, 0);
        } else if (geometry.indirectArgsBuffer) {
            encoder.drawIndexedIndirect(geometry.indirectArgsBuffer, 0);
        } else {
            encoder.drawIndexed(geometry.vertexCount);
        }
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
