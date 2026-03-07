import { Camera } from '../cameras/Camera';
import { Scene } from '../objects/Scene';
import { GBuffer } from '../postprocessing/GBuffer';
import { PostProcessingEffect } from '../postprocessing/PostProcessingEffect';
import { BVHBuilder } from './BVHBuilder';
import { intersectionShader } from './shaders/intersection.wgsl';
import { traversalShader } from './shaders/traversal.wgsl';
import { traceShader } from './shaders/trace.wgsl';
import { denoiseTemporalShader } from './shaders/denoise-temporal.wgsl';
import { denoiseSpatialShader } from './shaders/denoise-spatial.wgsl';
import { compositeShader } from './shaders/composite.wgsl';
import { mat4 } from 'gl-matrix';

export interface PathTracerOptions {
    scene: Scene;
    /** Temporal blend factor (0 = full history, 1 = current frame only). Default: 0.1 */
    temporalBlend?: number;
    /** Number of a-trous spatial denoise iterations. Default: 3 */
    spatialPasses?: number;
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

    // Trace pipeline
    private _tracePipeline: GPUComputePipeline | null = null;
    private _traceBGL: GPUBindGroupLayout | null = null;
    private _traceParamsBuf: GPUBuffer | null = null;
    private _lightsBuffer: GPUBuffer | null = null;

    // Temporal denoise pipeline
    private _temporalPipeline: GPUComputePipeline | null = null;
    private _temporalBGL: GPUBindGroupLayout | null = null;
    private _temporalParamsBuf: GPUBuffer | null = null;
    private _historySampler: GPUSampler | null = null;

    // Spatial denoise pipeline
    private _spatialPipeline: GPUComputePipeline | null = null;
    private _spatialBGL: GPUBindGroupLayout | null = null;
    private _spatialParamsBuf: GPUBuffer | null = null;

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

    // Cached GBuffer textures for bind group rebuild detection
    private _gbuffer: GBuffer | null = null;
    private _width: number = 0;
    private _height: number = 0;

    // Transient reference to the last denoised texture (set each frame in spatial pass)
    private _lastDenoisedTex: GPUTexture | null = null;

    // Previous frame's view-projection matrix for temporal reprojection
    private _prevViewProj = mat4.create();
    private _invViewProj = mat4.create();

    constructor(options: PathTracerOptions) {
        super();
        this._scene = options.scene;
        this._temporalBlend = options.temporalBlend ?? 0.1;
        this._spatialPasses = options.spatialPasses ?? 3;
    }

    /** Access the BVH builder for external configuration. */
    get bvhBuilder(): BVHBuilder | null { return this._bvhBuilder; }

    /** Temporal blend factor (0 = full history, 1 = current frame only). */
    get temporalBlend(): number { return this._temporalBlend; }
    set temporalBlend(v: number) { this._temporalBlend = v; }

    /** Number of à-trous spatial denoise iterations. */
    get spatialPasses(): number { return this._spatialPasses; }
    set spatialPasses(v: number) { this._spatialPasses = Math.max(0, Math.round(v)); }

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
        }

        // Build TLAS every frame (transforms may have changed)
        this._bvhBuilder.buildTLAS(commandEncoder, this._scene);

        // Guard against missing BVH buffers
        if (!this._bvhBuilder.triangleBuffer || !this._bvhBuilder.blasNodeBuffer ||
            !this._bvhBuilder.tlasNodeBuffer || !this._bvhBuilder.instanceBuffer ||
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

        // ── Pass 1: Path trace ──
        this._dispatchTrace(commandEncoder, depth, camera, width, height, iv);

        // ── Pass 2: Temporal denoise ──
        this._dispatchTemporalDenoise(commandEncoder, depth, width, height, vp);

        // ── Pass 3: Spatial denoise (a-trous wavelet) ──
        this._dispatchSpatialDenoise(commandEncoder, depth, width, height);

        // ── Pass 4: Composite GI onto scene ──
        this._dispatchComposite(commandEncoder, input, output, width, height);

        // Save current VP as history for next frame
        mat4.copy(this._prevViewProj, vp);
        this._frameIndex++;
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
        this._temporalParamsBuf?.destroy();
        this._spatialParamsBuf?.destroy();
        this._compositeParamsBuf?.destroy();
        this._bvhBuilder = null;
        this._tracePipeline = null;
        this._temporalPipeline = null;
        this._spatialPipeline = null;
        this._compositePipeline = null;
        this._device = null;
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

        // Temporal denoise pipeline
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

        // Spatial denoise pipeline
        this._spatialBGL = device.createBindGroupLayout({
            label: 'PathTracer/SpatialBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // inputGI
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } }, // outputGI
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },                  // depth
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },                  // normal
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },                       // params
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
    }

    private _createParamBuffers(device: GPUDevice): void {
        // TraceParams: mat4x4f(64) + vec3f+u32(16) + 4*u32(16) = 96
        this._traceParamsBuf = device.createBuffer({
            label: 'PathTracer/TraceParams',
            size: 96,
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
    }

    // ── Texture management ────────────────────────────────────────────────

    private _createTextures(device: GPUDevice, w: number, h: number): void {
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

        this._giTexture = device.createTexture({
            label: 'PathTracer/GI',
            size: [w, h],
            format: 'rgba16float',
            usage,
        });

        this._historyTexA = device.createTexture({
            label: 'PathTracer/HistoryA',
            size: [w, h],
            format: 'rgba16float',
            usage,
        });

        this._historyTexB = device.createTexture({
            label: 'PathTracer/HistoryB',
            size: [w, h],
            format: 'rgba16float',
            usage,
        });

        this._spatialScratchTex = device.createTexture({
            label: 'PathTracer/SpatialScratch',
            size: [w, h],
            format: 'rgba16float',
            usage,
        });
    }

    private _destroyTextures(): void {
        this._giTexture?.destroy();
        this._historyTexA?.destroy();
        this._historyTexB?.destroy();
        this._spatialScratchTex?.destroy();
        this._giTexture = null;
        this._historyTexA = null;
        this._historyTexB = null;
        this._spatialScratchTex = null;
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
    ): void {
        const device = this._device!;
        const builder = this._bvhBuilder!;

        // Pack lights into storage buffer
        const lightCount = this._packLights(device);

        // TraceParams layout (96 bytes = 24 floats):
        // invViewProj : mat4x4f  offset 0   (float[0..15])
        // cameraPos   : vec3f    offset 64  (float[16..18])
        // frameIndex  : u32      offset 76  (float[19])
        // width       : u32      offset 80  (float[20])
        // height      : u32      offset 84  (float[21])
        // lightCount  : u32      offset 88  (float[22])
        // _pad        : u32      offset 92  (float[23])
        const params = new Float32Array(24); // 96 bytes
        const paramsU32 = new Uint32Array(params.buffer);

        params.set(this._invViewProj as unknown as Float32Array, 0);
        params[16] = invView[12];
        params[17] = invView[13];
        params[18] = invView[14];
        paramsU32[19] = this._frameIndex;
        paramsU32[20] = width;
        paramsU32[21] = height;
        paramsU32[22] = lightCount;
        paramsU32[23] = 0; // _pad

        device.queue.writeBuffer(this._traceParamsBuf!, 0, params);

        const traceBG = device.createBindGroup({
            layout: this._traceBGL!,
            entries: [
                { binding: 0, resource: depth.createView() },
                { binding: 1, resource: this._gbuffer!.normalTexture.createView() },
                { binding: 2, resource: this._gbuffer!.albedoTexture.createView() },
                { binding: 3, resource: this._giTexture!.createView() },
                { binding: 4, resource: { buffer: this._traceParamsBuf! } },
                { binding: 5, resource: { buffer: builder.triangleBuffer! } },
                { binding: 6, resource: { buffer: builder.blasNodeBuffer! } },
                { binding: 7, resource: { buffer: builder.tlasNodeBuffer! } },
                { binding: 8, resource: { buffer: builder.instanceBuffer! } },
                { binding: 9, resource: { buffer: builder.materialBuffer! } },
                { binding: 10, resource: { buffer: this._lightsBuffer! } },
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

        let currentInput = this._historyPing ? this._historyTexB! : this._historyTexA!;
        let currentOutput = this._spatialScratchTex!;

        for (let i = 0; i < this._spatialPasses; i++) {
            const stepSize = 1 << i;

            const iterParamsBuf = device.createBuffer({
                label: `PathTracer/SpatialParams/${i}`,
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            const params = new Float32Array(8);
            const paramsU32 = new Uint32Array(params.buffer);
            paramsU32[0] = stepSize;
            params[1] = 0.01;
            params[2] = 128.0;
            params[3] = 4.0;
            paramsU32[4] = width;
            paramsU32[5] = height;
            paramsU32[6] = 0;
            paramsU32[7] = 0;

            device.queue.writeBuffer(iterParamsBuf, 0, params);

            const spatialBG = device.createBindGroup({
                layout: this._spatialBGL!,
                entries: [
                    { binding: 0, resource: currentInput.createView() },
                    { binding: 1, resource: currentOutput.createView() },
                    { binding: 2, resource: depth.createView() },
                    { binding: 3, resource: this._gbuffer!.normalTexture.createView() },
                    { binding: 4, resource: { buffer: iterParamsBuf } },
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
            ],
        });

        const pass = commandEncoder.beginComputePass({ label: 'PathTracer/Composite' });
        pass.setPipeline(this._compositePipeline!);
        pass.setBindGroup(0, compositeBG);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }
}
