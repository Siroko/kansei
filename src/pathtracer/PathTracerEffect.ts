import { Camera } from '../cameras/Camera';
import { Scene } from '../objects/Scene';
import { GBuffer } from '../postprocessing/GBuffer';
import { PostProcessingEffect } from '../postprocessing/PostProcessingEffect';
import { BVHBuilder } from './BVHBuilder';
import { generateBlueNoise } from './blue-noise';
import { intersectionShader } from './shaders/intersection.wgsl';
import { traversalShader } from './shaders/traversal.wgsl';
import { traceShader } from './shaders/trace.wgsl';
import { denoiseTemporalShader } from './shaders/denoise-temporal.wgsl';
import { denoiseSpatialShader } from './shaders/denoise-spatial.wgsl';
import { compositeShader } from './shaders/composite.wgsl';
import { restirGenerateShader, restirSpatialShader } from './shaders/restir-di.wgsl';
import { mat4 } from 'gl-matrix';

export interface PathTracerOptions {
    scene: Scene;
    /** Temporal blend factor (0 = full history, 1 = current frame only). Default: 0.1 */
    temporalBlend?: number;
    /** Number of a-trous spatial denoise iterations. Default: 3 */
    spatialPasses?: number;
    /** Samples per pixel per frame. Default: 1 */
    spp?: number;
    /** Use blue noise sampling (vs white noise). Default: true */
    useBlueNoise?: boolean;
    /** Use fixed seed (no temporal animation). Default: false */
    fixedSeed?: boolean;
    /** GI trace resolution scale (0.25 = quarter res, 0.5 = half res, 1.0 = full). Default: 0.5 */
    traceScale?: number;
    /** Maximum path bounces for indirect illumination. Default: 3 */
    maxBounces?: number;
    /** Ambient/sky color added when bounce rays miss geometry. Default: [0, 0, 0] */
    ambientColor?: [number, number, number];
    /** Enable SVGF variance-guided spatial denoising. Default: false */
    useSVGF?: boolean;
    /** Enable ReSTIR DI for direct illumination. Default: true */
    useReSTIR?: boolean;
    /** ReSTIR temporal history cap. Default: 20 */
    restirMaxHistory?: number;
}

/** GPU-packed light data — 64 bytes, matches LightData WGSL struct. */
const LIGHT_STRIDE_FLOATS = 16; // 64 bytes / 4
const MAX_LIGHTS = 16;

export class PathTracerEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _scene: Scene;
    private _bvhBuilder: BVHBuilder | null = null;
    private _blasBuilt: boolean = false;
    private _frameIndex: number = 0;

    // Options
    private _temporalBlend: number;
    private _spatialPasses: number;
    private _spp: number;
    private _useBlueNoise: boolean;
    private _fixedSeed: boolean;
    private _traceScale: number;
    private _maxBounces: number;
    private _ambientColor: [number, number, number];
    private _useSVGF: boolean;
    private _useReSTIR: boolean;
    private _restirMaxHistory: number;

    // Trace pipeline
    private _tracePipeline: GPUComputePipeline | null = null;
    private _traceBGL: GPUBindGroupLayout | null = null;
    private _traceParamsBuf: GPUBuffer | null = null;
    private _lightsBuffer: GPUBuffer | null = null;
    private _blueNoiseBuffer: GPUBuffer | null = null;

    // Temporal denoise pipeline
    private _temporalPipeline: GPUComputePipeline | null = null;
    private _temporalBGL: GPUBindGroupLayout | null = null;
    private _temporalParamsBuf: GPUBuffer | null = null;
    private _historySampler: GPUSampler | null = null;

    // Spatial denoise pipeline
    private _spatialPipeline: GPUComputePipeline | null = null;
    private _spatialBGL: GPUBindGroupLayout | null = null;
    private _spatialParamsBuf: GPUBuffer | null = null;
    private _spatialIterParamsBufs: GPUBuffer[] = [];

    // Composite pipeline
    private _compositePipeline: GPUComputePipeline | null = null;
    private _compositeBGL: GPUBindGroupLayout | null = null;
    private _compositeParamsBuf: GPUBuffer | null = null;

    // GI textures (rgba16float)
    private _giTexture: GPUTexture | null = null;          // raw trace output
    private _historyTexA: GPUTexture | null = null;        // temporal ping
    private _historyTexB: GPUTexture | null = null;        // temporal pong
    private _spatialScratchTex: GPUTexture | null = null;  // spatial denoise scratch
    private _historyPing: boolean = true;                  // which history buffer is "current"

    // SVGF moments textures (rgba16float): R=μ₁, G=μ₂, B=historyLen, A=variance
    private _momentsTexA: GPUTexture | null = null;
    private _momentsTexB: GPUTexture | null = null;

    // ReSTIR DI
    private _restirGenPipeline: GPUComputePipeline | null = null;
    private _restirGenBGL: GPUBindGroupLayout | null = null;
    private _restirSpatialPipeline: GPUComputePipeline | null = null;
    private _restirSpatialBGL: GPUBindGroupLayout | null = null;
    private _restirParamsBuf: GPUBuffer | null = null;
    private _reservoirBufA: GPUBuffer | null = null;  // ping
    private _reservoirBufB: GPUBuffer | null = null;  // pong
    private _reservoirPing: boolean = true;
    private _restirDirectTex: GPUTexture | null = null;
    private _restirDummyTex: GPUTexture | null = null; // 1x1 dummy when ReSTIR disabled

    // Cached GBuffer textures for bind group rebuild detection
    private _gbuffer: GBuffer | null = null;
    private _width: number = 0;
    private _height: number = 0;

    // Transient reference to the last denoised texture (set each frame in spatial pass)
    private _lastDenoisedTex: GPUTexture | null = null;

    // View-projection matrices
    private _prevViewProj = mat4.create();
    private _invViewProj = mat4.create();

    constructor(options: PathTracerOptions) {
        super();
        this._scene = options.scene;
        this._temporalBlend = options.temporalBlend ?? 0.1;
        this._spatialPasses = options.spatialPasses ?? 3;
        this._spp = options.spp ?? 1;
        this._useBlueNoise = options.useBlueNoise ?? true;
        this._fixedSeed = options.fixedSeed ?? false;
        this._traceScale = options.traceScale ?? 0.5;
        this._maxBounces = options.maxBounces ?? 3;
        this._ambientColor = options.ambientColor ?? [0, 0, 0];
        this._useSVGF = options.useSVGF ?? false;
        this._useReSTIR = options.useReSTIR ?? true;
        this._restirMaxHistory = options.restirMaxHistory ?? 20;
    }

    /** Access the BVH builder for external configuration. */
    get bvhBuilder(): BVHBuilder | null { return this._bvhBuilder; }

    /** Temporal blend factor (0 = full history, 1 = current frame only). */
    get temporalBlend(): number { return this._temporalBlend; }
    set temporalBlend(v: number) { this._temporalBlend = v; }

    /** Number of à-trous spatial denoise iterations. */
    get spatialPasses(): number { return this._spatialPasses; }
    set spatialPasses(v: number) { this._spatialPasses = Math.max(0, Math.round(v)); }

    /** Samples per pixel per frame. */
    get spp(): number { return this._spp; }
    set spp(v: number) { this._spp = Math.max(1, Math.round(v)); }

    /** Use blue noise sampling (vs white noise). */
    get useBlueNoise(): boolean { return this._useBlueNoise; }
    set useBlueNoise(v: boolean) { this._useBlueNoise = v; }

    /** Use fixed seed (no temporal animation). */
    get fixedSeed(): boolean { return this._fixedSeed; }
    set fixedSeed(v: boolean) { this._fixedSeed = v; }

    /** GI trace resolution scale. */
    get traceScale(): number { return this._traceScale; }
    set traceScale(v: number) {
        v = Math.max(0.1, Math.min(1.0, v));
        if (v !== this._traceScale) {
            this._traceScale = v;
            // Force texture recreation on next render
            if (this._device) {
                this._destroyTextures();
                this._createTextures(this._device, this._width, this._height);
                this._frameIndex = 0;
            }
        }
    }

    /** Maximum path bounces for indirect illumination. */
    get maxBounces(): number { return this._maxBounces; }
    set maxBounces(v: number) { this._maxBounces = Math.max(0, Math.round(v)); }

    /** Ambient/sky color added when bounce rays miss geometry. */
    get ambientColor(): [number, number, number] { return this._ambientColor; }
    set ambientColor(v: [number, number, number]) { this._ambientColor = v; }

    /** Enable SVGF variance-guided spatial denoising. */
    get useSVGF(): boolean { return this._useSVGF; }
    set useSVGF(v: boolean) { this._useSVGF = v; }

    /** Enable ReSTIR DI for direct illumination. */
    get useReSTIR(): boolean { return this._useReSTIR; }
    set useReSTIR(v: boolean) { this._useReSTIR = v; }

    /** ReSTIR temporal history cap. */
    get restirMaxHistory(): number { return this._restirMaxHistory; }
    set restirMaxHistory(v: number) { this._restirMaxHistory = Math.max(1, Math.round(v)); }

    // ── PostProcessingEffect interface ────────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        this._gbuffer = gbuffer;
        this._width = gbuffer.width;
        this._height = gbuffer.height;

        this._bvhBuilder = new BVHBuilder(device);

        this._historySampler = device.createSampler({
            label: 'PathTracer/HistorySampler',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        this._createBlueNoiseBuffer(device);
        this._createPipelines(device);
        this._createTextures(device, gbuffer.width, gbuffer.height);
        this._createParamBuffers(device);

        this.initialized = true;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
    ): void {
        if (!this._device || !this._bvhBuilder || !this._gbuffer) return;

        // Clean up scratch buffers from previous frame (safe — prior commands submitted)
        this._bvhBuilder.cleanupScratchBuffers();

        // Build BLAS on first frame (or when geometry changes)
        if (!this._blasBuilt) {
            this._bvhBuilder.buildBLAS(this._scene);
            this._bvhBuilder.buildBLASTree(commandEncoder);
            this._blasBuilt = true;
            console.log(`[PathTracer] triangles: ${this._bvhBuilder.totalTriangleCount}, BLAS nodes: ${this._bvhBuilder.totalBLASNodes}, instances: ${this._bvhBuilder.totalInstances}, materials: ${this._bvhBuilder.materialCount}`);
        }

        // Update materials every frame (runtime slider changes)
        this._bvhBuilder.updateMaterials(this._scene);

        // Build TLAS every frame (transforms may have changed)
        this._bvhBuilder.buildTLAS(commandEncoder, this._scene);

        // Guard against missing BVH buffers
        if (!this._bvhBuilder.triangleBuffer || !this._bvhBuilder.bvh4NodeBuffer ||
            !this._bvhBuilder.tlasBvh4Buffer || !this._bvhBuilder.instanceBuffer ||
            !this._bvhBuilder.materialBuffer) {
            return;
        }

        // Recreate textures if size changed
        if (width !== this._width || height !== this._height) {
            this._width = width;
            this._height = height;
            this._destroyTextures();
            this._createTextures(this._device, width, height);
        }

        // Compute view-projection and inverse
        const vp = mat4.create();
        mat4.multiply(vp, camera.projectionMatrix.internalMat4, camera.viewMatrix.internalMat4);
        mat4.invert(this._invViewProj, vp);

        const iv = camera.inverseViewMatrix.internalMat4;
        const tw = this._traceWidth(width);
        const th = this._traceHeight(height);

        // Pack lights (shared by trace + ReSTIR)
        const lightCount = this._packLights(this._device);

        // ── Pass 0a/0b: ReSTIR DI (at reduced resolution) ──
        if (this._useReSTIR) {
            this._dispatchReSTIR(commandEncoder, depth, tw, th, lightCount, iv);
        }

        // ── Pass 1: Path trace (at reduced resolution) ──
        this._dispatchTrace(commandEncoder, depth, camera, tw, th, iv, lightCount);

        // ── Pass 2: Temporal denoise (at reduced resolution) ──
        this._dispatchTemporalDenoise(commandEncoder, depth, tw, th, vp);

        // ── Pass 3: Spatial denoise (at reduced resolution) ──
        this._dispatchSpatialDenoise(commandEncoder, depth, tw, th);

        // ── Pass 4: Composite GI onto scene (full resolution output) ──
        this._dispatchComposite(commandEncoder, input, output, width, height);

        // Save current VP as history for next frame
        mat4.copy(this._prevViewProj, vp);
        this._frameIndex++;
        this._reservoirPing = !this._reservoirPing;
    }

    resize(width: number, height: number, gbuffer: GBuffer): void {
        this._gbuffer = gbuffer;
        if (this._device && (width !== this._width || height !== this._height)) {
            this._width = width;
            this._height = height;
            this._destroyTextures();
            this._createTextures(this._device, width, height);
            this._frameIndex = 0; // reset temporal accumulation
        }
    }

    destroy(): void {
        this._bvhBuilder?.destroy();
        this._destroyTextures();
        this._traceParamsBuf?.destroy();
        this._lightsBuffer?.destroy();
        this._blueNoiseBuffer?.destroy();
        this._temporalParamsBuf?.destroy();
        this._spatialParamsBuf?.destroy();
        this._compositeParamsBuf?.destroy();
        this._restirParamsBuf?.destroy();
        this._restirDummyTex?.destroy();
        for (const buf of this._spatialIterParamsBufs) buf.destroy();
        this._spatialIterParamsBufs = [];
        this._bvhBuilder = null;
        this._tracePipeline = null;
        this._temporalPipeline = null;
        this._spatialPipeline = null;
        this._compositePipeline = null;
        this._restirGenPipeline = null;
        this._restirSpatialPipeline = null;
        this._device = null;
    }

    // ── Blue noise ─────────────────────────────────────────────────────────

    private _createBlueNoiseBuffer(device: GPUDevice): void {
        const data = generateBlueNoise(128);
        this._blueNoiseBuffer = device.createBuffer({
            label: 'PathTracer/BlueNoise',
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this._blueNoiseBuffer, 0, data);
    }

    // ── Pipeline creation ─────────────────────────────────────────────────

    private _createPipelines(device: GPUDevice): void {
        // Trace pipeline
        this._traceBGL = device.createBindGroupLayout({
            label: 'PathTracer/TraceBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },                  // depth
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // normal
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // albedo
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // giOutput
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },                       // traceParams
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },             // triangles
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },             // blasNodes
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },             // tlasNodes
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },             // instances
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },             // materials
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },            // sceneLights
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },            // blueNoise
                { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // emissive
                { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // restirDirect
            ],
        });

        const traceCode = intersectionShader + traversalShader + traceShader;
        const traceModule = device.createShaderModule({
            label: 'PathTracer/TraceShader',
            code: traceCode,
        });
        this._tracePipeline = device.createComputePipeline({
            label: 'PathTracer/TracePipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._traceBGL] }),
            compute: { module: traceModule, entryPoint: 'main' },
        });

        // Temporal denoise pipeline (SVGF with moments tracking)
        this._temporalBGL = device.createBindGroupLayout({
            label: 'PathTracer/TemporalBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // currentGI
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // historyGI
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // outputGI
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },                  // depth
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // normal
                { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },                    // historySamp
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },                       // params
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // momentsHistory
                { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // momentsOutput
            ],
        });

        const temporalModule = device.createShaderModule({
            label: 'PathTracer/TemporalShader',
            code: denoiseTemporalShader,
        });
        this._temporalPipeline = device.createComputePipeline({
            label: 'PathTracer/TemporalPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._temporalBGL] }),
            compute: { module: temporalModule, entryPoint: 'main' },
        });

        // Spatial denoise pipeline (SVGF variance-guided)
        this._spatialBGL = device.createBindGroupLayout({
            label: 'PathTracer/SpatialBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // inputGI
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // outputGI
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },                  // depth
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // normal
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },                       // params
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // momentsTex
            ],
        });

        const spatialModule = device.createShaderModule({
            label: 'PathTracer/SpatialShader',
            code: denoiseSpatialShader,
        });
        this._spatialPipeline = device.createComputePipeline({
            label: 'PathTracer/SpatialPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._spatialBGL] }),
            compute: { module: spatialModule, entryPoint: 'main' },
        });

        // Composite pipeline
        this._compositeBGL = device.createBindGroupLayout({
            label: 'PathTracer/CompositeBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // inputTex (scene)
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // denoisedGI
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // albedoTex
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // outputTex
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },                       // params
                { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },                    // giSampler
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // emissiveTex
            ],
        });

        const compositeModule = device.createShaderModule({
            label: 'PathTracer/CompositeShader',
            code: compositeShader,
        });
        this._compositePipeline = device.createComputePipeline({
            label: 'PathTracer/CompositePipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._compositeBGL] }),
            compute: { module: compositeModule, entryPoint: 'main' },
        });

        // ReSTIR DI — Generate + Temporal pipeline
        const V = GPUShaderStage.COMPUTE;
        this._restirGenBGL = device.createBindGroupLayout({
            label: 'PathTracer/ReSTIRGenBGL',
            entries: [
                { binding: 0,  visibility: V, texture: { sampleType: 'depth' } },            // depthTex
                { binding: 1,  visibility: V, texture: { sampleType: 'float' } },            // normalTex
                { binding: 2,  visibility: V, buffer: { type: 'uniform' } },                 // params
                { binding: 3,  visibility: V, buffer: { type: 'read-only-storage' } },       // sceneLights
                { binding: 4,  visibility: V, buffer: { type: 'read-only-storage' } },       // triangles
                { binding: 5,  visibility: V, buffer: { type: 'read-only-storage' } },       // bvh4Nodes
                { binding: 6,  visibility: V, buffer: { type: 'read-only-storage' } },       // tlasBvh4Nodes
                { binding: 7,  visibility: V, buffer: { type: 'read-only-storage' } },       // instances
                { binding: 8,  visibility: V, buffer: { type: 'read-only-storage' } },       // materials
                { binding: 9,  visibility: V, buffer: { type: 'read-only-storage' } },       // reservoirPrev
                { binding: 10, visibility: V, buffer: { type: 'storage' } },                 // reservoirCur
            ],
        });

        const restirGenCode = intersectionShader + traversalShader + restirGenerateShader;
        const restirGenModule = device.createShaderModule({
            label: 'PathTracer/ReSTIRGenShader',
            code: restirGenCode,
        });
        this._restirGenPipeline = device.createComputePipeline({
            label: 'PathTracer/ReSTIRGenPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._restirGenBGL] }),
            compute: { module: restirGenModule, entryPoint: 'main' },
        });

        // ReSTIR DI — Spatial + Shade pipeline
        this._restirSpatialBGL = device.createBindGroupLayout({
            label: 'PathTracer/ReSTIRSpatialBGL',
            entries: [
                { binding: 0,  visibility: V, texture: { sampleType: 'depth' } },            // depthTex
                { binding: 1,  visibility: V, texture: { sampleType: 'float' } },            // normalTex
                { binding: 2,  visibility: V, buffer: { type: 'uniform' } },                 // params
                { binding: 3,  visibility: V, buffer: { type: 'read-only-storage' } },       // sceneLights
                { binding: 4,  visibility: V, buffer: { type: 'read-only-storage' } },       // triangles
                { binding: 5,  visibility: V, buffer: { type: 'read-only-storage' } },       // bvh4Nodes
                { binding: 6,  visibility: V, buffer: { type: 'read-only-storage' } },       // tlasBvh4Nodes
                { binding: 7,  visibility: V, buffer: { type: 'read-only-storage' } },       // instances
                { binding: 8,  visibility: V, buffer: { type: 'read-only-storage' } },       // materials
                { binding: 9,  visibility: V, buffer: { type: 'read-only-storage' } },       // reservoirIn
                { binding: 10, visibility: V, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // directLightOut
            ],
        });

        const restirSpatialCode = intersectionShader + traversalShader + restirSpatialShader;
        const restirSpatialModule = device.createShaderModule({
            label: 'PathTracer/ReSTIRSpatialShader',
            code: restirSpatialCode,
        });
        this._restirSpatialPipeline = device.createComputePipeline({
            label: 'PathTracer/ReSTIRSpatialPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._restirSpatialBGL] }),
            compute: { module: restirSpatialModule, entryPoint: 'main' },
        });
    }

    private _createParamBuffers(device: GPUDevice): void {
        // TraceParams: mat4x4f(64) + vec3f+u32(16) + 8*u32(32) + vec3f+u32(16) = 128
        this._traceParamsBuf = device.createBuffer({
            label: 'PathTracer/TraceParams',
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Lights buffer: MAX_LIGHTS * 64 bytes
        this._lightsBuffer = device.createBuffer({
            label: 'PathTracer/Lights',
            size: MAX_LIGHTS * LIGHT_STRIDE_FLOATS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // TemporalParams: mat4x4f(64) + mat4x4f(64) + f32(4) + f32(4) + f32(4) + u32(4) = 144, pad to 160
        this._temporalParamsBuf = device.createBuffer({
            label: 'PathTracer/TemporalParams',
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // SpatialParams: u32(4) + f32(4) + f32(4) + f32(4) + u32(4) + u32(4) + vec2u(8) = 32
        this._spatialParamsBuf = device.createBuffer({
            label: 'PathTracer/SpatialParams',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // CompositeParams: u32(4) + u32(4) + vec2u(8) = 16
        this._compositeParamsBuf = device.createBuffer({
            label: 'PathTracer/CompositeParams',
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pre-allocate spatial denoise per-iteration param buffers (reused each frame)
        for (let i = 0; i < 5; i++) {
            this._spatialIterParamsBufs.push(device.createBuffer({
                label: `PathTracer/SpatialIterParams/${i}`,
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }

        // ReSTIR params: mat4x4f(64) + mat4x4f(64) + vec3f(12) + u32(4) + 4*u32(16) = 160
        this._restirParamsBuf = device.createBuffer({
            label: 'PathTracer/ReSTIRParams',
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // 1x1 dummy texture for when ReSTIR is disabled
        this._restirDummyTex = device.createTexture({
            label: 'PathTracer/ReSTIRDummy',
            size: [1, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
    }

    // ── Texture management ────────────────────────────────────────────────

    private _traceWidth(w: number): number { return Math.max(1, Math.floor(w * this._traceScale)); }
    private _traceHeight(h: number): number { return Math.max(1, Math.floor(h * this._traceScale)); }

    private _createTextures(device: GPUDevice, w: number, h: number): void {
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const tw = this._traceWidth(w);
        const th = this._traceHeight(h);

        this._giTexture = device.createTexture({
            label: 'PathTracer/GI',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        this._historyTexA = device.createTexture({
            label: 'PathTracer/HistoryA',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        this._historyTexB = device.createTexture({
            label: 'PathTracer/HistoryB',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        this._spatialScratchTex = device.createTexture({
            label: 'PathTracer/SpatialScratch',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        this._momentsTexA = device.createTexture({
            label: 'PathTracer/MomentsA',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        this._momentsTexB = device.createTexture({
            label: 'PathTracer/MomentsB',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        // ReSTIR direct light output texture
        this._restirDirectTex = device.createTexture({
            label: 'PathTracer/ReSTIRDirect',
            size: [tw, th],
            format: 'rgba16float',
            usage,
        });

        // Reservoir buffers: 3 vec4f (48 bytes) per pixel
        const reservoirSize = tw * th * 3 * 16;
        const bufUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this._reservoirBufA = device.createBuffer({
            label: 'PathTracer/ReservoirA',
            size: reservoirSize,
            usage: bufUsage,
        });
        this._reservoirBufB = device.createBuffer({
            label: 'PathTracer/ReservoirB',
            size: reservoirSize,
            usage: bufUsage,
        });
    }

    private _destroyTextures(): void {
        this._giTexture?.destroy();
        this._historyTexA?.destroy();
        this._historyTexB?.destroy();
        this._spatialScratchTex?.destroy();
        this._momentsTexA?.destroy();
        this._momentsTexB?.destroy();
        this._giTexture = null;
        this._historyTexA = null;
        this._historyTexB = null;
        this._spatialScratchTex = null;
        this._momentsTexA = null;
        this._momentsTexB = null;
        this._restirDirectTex?.destroy();
        this._reservoirBufA?.destroy();
        this._reservoirBufB?.destroy();
        this._restirDirectTex = null;
        this._reservoirBufA = null;
        this._reservoirBufB = null;
    }

    // ── Light packing ─────────────────────────────────────────────────────

    /**
     * Pack all scene lights into the lights storage buffer.
     * LightData layout (64 bytes = 16 floats):
     *   [0-2]  position.xyz   [3]  lightType (as float, bitcast to u32)
     *   [4-6]  color.rgb      [7]  intensity
     *   [8-10] normal.xyz     [11] _pad
     *   [12-15] extra.xyzw
     */
    private _packLights(device: GPUDevice): number {
        const scene = this._scene;
        const lightData = new Float32Array(MAX_LIGHTS * LIGHT_STRIDE_FLOATS);
        const lightU32 = new Uint32Array(lightData.buffer);
        let idx = 0;

        // Directional lights (type = 1)
        for (const dl of scene.directionalLights) {
            if (idx >= MAX_LIGHTS) break;
            const off = idx * LIGHT_STRIDE_FLOATS;
            lightData[off + 0] = dl.direction[0];
            lightData[off + 1] = dl.direction[1];
            lightData[off + 2] = dl.direction[2];
            lightU32[off + 3] = 1; // LIGHT_DIRECTIONAL
            lightData[off + 4] = dl.color[0];
            lightData[off + 5] = dl.color[1];
            lightData[off + 6] = dl.color[2];
            lightData[off + 7] = dl.intensity;
            // normal, extra unused for directional
            idx++;
        }

        // Area lights (type = 2)
        for (const al of scene.areaLights) {
            if (idx >= MAX_LIGHTS) break;
            const off = idx * LIGHT_STRIDE_FLOATS;
            lightData[off + 0] = al.position.x;
            lightData[off + 1] = al.position.y;
            lightData[off + 2] = al.position.z;
            lightU32[off + 3] = 2; // LIGHT_AREA
            lightData[off + 4] = al.color[0];
            lightData[off + 5] = al.color[1];
            lightData[off + 6] = al.color[2];
            lightData[off + 7] = al.intensity;
            // Normal: direction the light faces (from position toward target)
            const dir = al.direction;
            lightData[off + 8]  = dir[0];
            lightData[off + 9]  = dir[1];
            lightData[off + 10] = dir[2];
            lightData[off + 11] = 0; // _pad
            // Extra: sizeX, sizeZ
            lightData[off + 12] = al.size[0];
            lightData[off + 13] = al.size[1];
            lightData[off + 14] = 0;
            lightData[off + 15] = 0;
            idx++;
        }

        // Point lights (type = 3)
        for (const pl of scene.pointLights) {
            if (idx >= MAX_LIGHTS) break;
            const off = idx * LIGHT_STRIDE_FLOATS;
            lightData[off + 0] = pl.position.x;
            lightData[off + 1] = pl.position.y;
            lightData[off + 2] = pl.position.z;
            lightU32[off + 3] = 3; // LIGHT_POINT
            lightData[off + 4] = pl.color[0];
            lightData[off + 5] = pl.color[1];
            lightData[off + 6] = pl.color[2];
            lightData[off + 7] = pl.intensity;
            // extra.x = radius (if applicable)
            idx++;
        }

        device.queue.writeBuffer(this._lightsBuffer!, 0, lightData);
        return idx;
    }

    // ── Dispatch helpers ──────────────────────────────────────────────────

    private _dispatchTrace(
        commandEncoder: GPUCommandEncoder,
        depth: GPUTexture,
        _camera: Camera,
        width: number,
        height: number,
        invView: mat4,
        lightCount: number,
    ): void {
        const device = this._device!;
        const builder = this._bvhBuilder!;

        // TraceParams layout (128 bytes = 32 floats):
        const params = new Float32Array(32); // 128 bytes
        const paramsU32 = new Uint32Array(params.buffer);

        params.set(this._invViewProj as unknown as Float32Array, 0);
        params[16] = invView[12];
        params[17] = invView[13];
        params[18] = invView[14];
        paramsU32[19] = this._frameIndex;
        paramsU32[20] = width;
        paramsU32[21] = height;
        paramsU32[22] = lightCount;
        paramsU32[23] = this._spp;
        paramsU32[24] = this._useBlueNoise ? 1 : 0;
        paramsU32[25] = this._fixedSeed ? 1 : 0;
        paramsU32[26] = this._maxBounces;
        paramsU32[27] = this._useReSTIR ? 1 : 0; // useReSTIR flag
        params[28] = this._ambientColor[0];
        params[29] = this._ambientColor[1];
        params[30] = this._ambientColor[2];

        device.queue.writeBuffer(this._traceParamsBuf!, 0, params);

        // Use ReSTIR direct light texture or dummy when disabled
        const restirTex = (this._useReSTIR && this._restirDirectTex)
            ? this._restirDirectTex : this._restirDummyTex!;

        const traceBG = device.createBindGroup({
            layout: this._traceBGL!,
            entries: [
                { binding: 0, resource: depth.createView() },
                { binding: 1, resource: this._gbuffer!.normalTexture.createView() },
                { binding: 2, resource: this._gbuffer!.albedoTexture.createView() },
                { binding: 3, resource: this._giTexture!.createView() },
                { binding: 4, resource: { buffer: this._traceParamsBuf! } },
                { binding: 5, resource: { buffer: builder.triangleBuffer! } },
                { binding: 6, resource: { buffer: builder.bvh4NodeBuffer! } },
                { binding: 7, resource: { buffer: builder.tlasBvh4Buffer! } },
                { binding: 8, resource: { buffer: builder.instanceBuffer! } },
                { binding: 9, resource: { buffer: builder.materialBuffer! } },
                { binding: 10, resource: { buffer: this._lightsBuffer! } },
                { binding: 11, resource: { buffer: this._blueNoiseBuffer! } },
                { binding: 12, resource: this._gbuffer!.emissiveTexture.createView() },
                { binding: 13, resource: restirTex.createView() },
            ],
        });

        const pass = commandEncoder.beginComputePass({ label: 'PathTracer/Trace' });
        pass.setPipeline(this._tracePipeline!);
        pass.setBindGroup(0, traceBG);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    private _dispatchTemporalDenoise(
        commandEncoder: GPUCommandEncoder,
        depth: GPUTexture,
        width: number,
        height: number,
        _currentVP: mat4,
    ): void {
        const device = this._device!;

        const historyRead = this._historyPing ? this._historyTexA! : this._historyTexB!;
        const historyWrite = this._historyPing ? this._historyTexB! : this._historyTexA!;
        const momentsRead = this._historyPing ? this._momentsTexA! : this._momentsTexB!;
        const momentsWrite = this._historyPing ? this._momentsTexB! : this._momentsTexA!;

        const params = new Float32Array(36);
        const paramsU32 = new Uint32Array(params.buffer);

        params.set(this._invViewProj as unknown as Float32Array, 0);
        params.set(this._prevViewProj as unknown as Float32Array, 16);
        params[32] = this._temporalBlend;
        params[33] = width;
        params[34] = height;
        paramsU32[35] = this._frameIndex;

        device.queue.writeBuffer(this._temporalParamsBuf!, 0, params);

        const temporalBG = device.createBindGroup({
            layout: this._temporalBGL!,
            entries: [
                { binding: 0, resource: this._giTexture!.createView() },
                { binding: 1, resource: historyRead.createView() },
                { binding: 2, resource: historyWrite.createView() },
                { binding: 3, resource: depth.createView() },
                { binding: 4, resource: this._gbuffer!.normalTexture.createView() },
                { binding: 5, resource: this._historySampler! },
                { binding: 6, resource: { buffer: this._temporalParamsBuf! } },
                { binding: 7, resource: momentsRead.createView() },
                { binding: 8, resource: momentsWrite.createView() },
            ],
        });

        const pass = commandEncoder.beginComputePass({ label: 'PathTracer/TemporalDenoise' });
        pass.setPipeline(this._temporalPipeline!);
        pass.setBindGroup(0, temporalBG);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();

        this._historyPing = !this._historyPing;
    }

    private _dispatchSpatialDenoise(
        commandEncoder: GPUCommandEncoder,
        depth: GPUTexture,
        width: number,
        height: number,
    ): void {
        const device = this._device!;

        let currentInput = this._historyPing ? this._historyTexA! : this._historyTexB!;
        let currentOutput = this._spatialScratchTex!;

        for (let i = 0; i < this._spatialPasses; i++) {
            const stepSize = 1 << i;

            const iterParamsBuf = this._spatialIterParamsBufs[i];

            const params = new Float32Array(8);
            const paramsU32 = new Uint32Array(params.buffer);
            paramsU32[0] = stepSize;
            params[1] = 0.1;   // sigmaDepth — higher = more tolerant of depth variation
            params[2] = 32.0;  // sigmaNormal — lower = softer edge stopping on curved surfaces
            params[3] = 2.0;   // sigmaLum — lower = tighter luminance edge stopping
            paramsU32[4] = width;
            paramsU32[5] = height;
            paramsU32[6] = this._useSVGF ? 1 : 0;
            paramsU32[7] = 0;

            device.queue.writeBuffer(iterParamsBuf, 0, params);

            // Moments texture for SVGF variance — read from whichever was written by temporal pass
            const momentsTex = this._historyPing ? this._momentsTexA! : this._momentsTexB!;

            const spatialBG = device.createBindGroup({
                layout: this._spatialBGL!,
                entries: [
                    { binding: 0, resource: currentInput.createView() },
                    { binding: 1, resource: currentOutput.createView() },
                    { binding: 2, resource: depth.createView() },
                    { binding: 3, resource: this._gbuffer!.normalTexture.createView() },
                    { binding: 4, resource: { buffer: iterParamsBuf } },
                    { binding: 5, resource: momentsTex.createView() },
                ],
            });

            const pass = commandEncoder.beginComputePass({ label: `PathTracer/SpatialDenoise/${i}` });
            pass.setPipeline(this._spatialPipeline!);
            pass.setBindGroup(0, spatialBG);
            pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
            pass.end();

            const tmp = currentInput;
            currentInput = currentOutput;
            currentOutput = tmp;
        }

        this._lastDenoisedTex = currentInput;
    }

    private _dispatchComposite(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        output: GPUTexture,
        width: number,
        height: number,
    ): void {
        const device = this._device!;
        const denoisedGI = this._lastDenoisedTex ?? this._giTexture!;

        const params = new Uint32Array([width, height, 0, 0]);
        device.queue.writeBuffer(this._compositeParamsBuf!, 0, params);

        const compositeBG = device.createBindGroup({
            layout: this._compositeBGL!,
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: denoisedGI.createView() },
                { binding: 2, resource: this._gbuffer!.albedoTexture.createView() },
                { binding: 3, resource: output.createView() },
                { binding: 4, resource: { buffer: this._compositeParamsBuf! } },
                { binding: 5, resource: this._historySampler! },
                { binding: 6, resource: this._gbuffer!.emissiveTexture.createView() },
            ],
        });

        const pass = commandEncoder.beginComputePass({ label: 'PathTracer/Composite' });
        pass.setPipeline(this._compositePipeline!);
        pass.setBindGroup(0, compositeBG);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    // ── ReSTIR DI dispatch ──────────────────────────────────────────────

    private _dispatchReSTIR(
        commandEncoder: GPUCommandEncoder,
        depth: GPUTexture,
        width: number,
        height: number,
        lightCount: number,
        invView: mat4,
    ): void {
        const device = this._device!;
        const builder = this._bvhBuilder!;

        // ReSTIRParams: invViewProj(64) + prevViewProj(64) + cameraPos(12) + u32(4) + 4*u32(16) = 160
        const params = new Float32Array(40); // 160 bytes
        const paramsU32 = new Uint32Array(params.buffer);

        params.set(this._invViewProj as unknown as Float32Array, 0);
        params.set(this._prevViewProj as unknown as Float32Array, 16);
        params[32] = invView[12]; // cameraPos.x
        params[33] = invView[13]; // cameraPos.y
        params[34] = invView[14]; // cameraPos.z
        paramsU32[35] = this._frameIndex;
        paramsU32[36] = width;
        paramsU32[37] = height;
        paramsU32[38] = lightCount;
        paramsU32[39] = this._restirMaxHistory;

        device.queue.writeBuffer(this._restirParamsBuf!, 0, params);

        const reservoirRead = this._reservoirPing ? this._reservoirBufA! : this._reservoirBufB!;
        const reservoirWrite = this._reservoirPing ? this._reservoirBufB! : this._reservoirBufA!;

        // Pass 0a: Generate + Temporal reuse
        const genBG = device.createBindGroup({
            layout: this._restirGenBGL!,
            entries: [
                { binding: 0,  resource: depth.createView() },
                { binding: 1,  resource: this._gbuffer!.normalTexture.createView() },
                { binding: 2,  resource: { buffer: this._restirParamsBuf! } },
                { binding: 3,  resource: { buffer: this._lightsBuffer! } },
                { binding: 4,  resource: { buffer: builder.triangleBuffer! } },
                { binding: 5,  resource: { buffer: builder.bvh4NodeBuffer! } },
                { binding: 6,  resource: { buffer: builder.tlasBvh4Buffer! } },
                { binding: 7,  resource: { buffer: builder.instanceBuffer! } },
                { binding: 8,  resource: { buffer: builder.materialBuffer! } },
                { binding: 9,  resource: { buffer: reservoirRead } },
                { binding: 10, resource: { buffer: reservoirWrite } },
            ],
        });

        const genPass = commandEncoder.beginComputePass({ label: 'PathTracer/ReSTIR-Generate' });
        genPass.setPipeline(this._restirGenPipeline!);
        genPass.setBindGroup(0, genBG);
        genPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        genPass.end();

        // Pass 0b: Spatial reuse + shade → directLightOut
        const spatialBG = device.createBindGroup({
            layout: this._restirSpatialBGL!,
            entries: [
                { binding: 0,  resource: depth.createView() },
                { binding: 1,  resource: this._gbuffer!.normalTexture.createView() },
                { binding: 2,  resource: { buffer: this._restirParamsBuf! } },
                { binding: 3,  resource: { buffer: this._lightsBuffer! } },
                { binding: 4,  resource: { buffer: builder.triangleBuffer! } },
                { binding: 5,  resource: { buffer: builder.bvh4NodeBuffer! } },
                { binding: 6,  resource: { buffer: builder.tlasBvh4Buffer! } },
                { binding: 7,  resource: { buffer: builder.instanceBuffer! } },
                { binding: 8,  resource: { buffer: builder.materialBuffer! } },
                { binding: 9,  resource: { buffer: reservoirWrite } },
                { binding: 10, resource: this._restirDirectTex!.createView() },
            ],
        });

        const spatialPass = commandEncoder.beginComputePass({ label: 'PathTracer/ReSTIR-Spatial' });
        spatialPass.setPipeline(this._restirSpatialPipeline!);
        spatialPass.setBindGroup(0, spatialBG);
        spatialPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        spatialPass.end();
    }
}
