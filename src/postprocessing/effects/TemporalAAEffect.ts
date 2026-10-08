import { mat4 } from 'gl-matrix';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { TAA_RESOLVE_WGSL } from '../../materials/shaders/SharedWGSL';
import { gpuPass } from '../../profiling/Profiler';

const HISTORY_FORMAT: GPUTextureFormat = 'rgba16float';
/** `TaaParams` in `taa_resolve.wgsl`: three mat4, then 12 scalars (Rust `TaaParamsGpu`). */
const PARAMS_BYTES = 240;

export interface TemporalAAOptions {
    /** History weight when the image is still (0.95: about 20 frames of accumulation). */
    feedbackMax: number;
    /** History weight at 8+ pixels of motion per frame. */
    feedbackMin: number;
    /**
     * Scene multiplier of the resolve's working space; set it to the tonemapper's exposure
     * (`ToneMapEffect.totalExposure()`) so its luma weighting matches what is displayed.
     */
    exposure: number;
    /**
     * Half-size of the history clip box in standard deviations of the neighbourhood (lower:
     * less ghosting, more flicker).
     */
    varianceGamma: number;
}

export function defaultTemporalAAOptions(): TemporalAAOptions {
    return { feedbackMax: 0.95, feedbackMin: 0.85, exposure: 1, varianceGamma: 1 };
}

/**
 * Temporal anti-aliasing (Rust `TemporalAAEffect`, `postprocessing/effects/taa.rs`): the
 * `PostProcessingVolume` jitters the camera's projection by a sub-pixel Halton offset every
 * frame while this effect is in the chain, and the resolve accumulates the jittered frames into a
 * history reprojected by motion vectors (materials with `outputsVelocity`, drawn in the
 * renderer's velocity pass) or by depth (everything else, camera motion only), clipped to the
 * current neighbourhood so moving things don't ghost.
 *
 * Put it after the volumetric fog and before depth of field, bloom and the tonemapper (it works
 * on linear light). On camera cuts call `resetHistory` here and `Camera.resetMotion`, so
 * nothing is reprojected across the cut.
 *
 * History is also dropped per pixel where it saw another surface: its view depth is kept in the
 * history's alpha and compared with where this pixel's surface was last frame, which keeps
 * swaying foliage (which uncovers and covers itself every frame) from smearing.
 *
 * It is also the chain's temporal upscaler: with `Renderer.setRenderScale` below 1 it reads the
 * GBuffer-size frame and writes (and keeps its history at) the canvas size, placing each
 * jittered sample where it fell among the canvas's pixels, so the history gathers detail finer
 * than a rendered pixel over the frames. At scale 1 it is the plain 1:1 resolve.
 *
 * ```ts
 * const tonemap = new ToneMapEffect({ ...toneMapOptionsForSurface(renderer.presentationFormat), exposure });
 * const taa = new TemporalAAEffect({ exposure: tonemap.totalExposure() });
 * const volume = new PostProcessingVolume(renderer, [taa, tonemap]);
 * ```
 */
class TemporalAAEffect extends PostProcessingEffect {
    public options: TemporalAAOptions;

    private _hasHistory = false;
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _bgl: GPUBindGroupLayout | null = null;
    private _params: GPUBuffer | null = null;
    private _sampler: GPUSampler | null = null;
    private readonly _paramsData = new ArrayBuffer(PARAMS_BYTES);
    private readonly _scratch = mat4.create();
    /** Ping-pong history at the output size; `_read` is last frame's. */
    private _history: GPUTexture[] = [];
    private _read = 0;
    private _size: [number, number] = [0, 0];
    /** A bind group per history read index, and what it binds. */
    private _bindGroups: ({ group: GPUBindGroup; key: GPUTexture[] } | null)[] = [null, null];

    constructor(options: Partial<TemporalAAOptions> = {}) {
        super();
        this.options = { ...defaultTemporalAAOptions(), ...options };
    }

    /** Start over from the current frame (camera cuts, teleports). */
    public resetHistory(): void {
        this._hasHistory = false;
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const visibility = GPUShaderStage.COMPUTE;
        const storage = { access: 'write-only', format: HISTORY_FORMAT } as const;
        this._bgl = device.createBindGroupLayout({
            label: 'TAA/BGL',
            entries: [
                { binding: 0, visibility, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility, texture: { sampleType: 'depth' } },
                { binding: 2, visibility, texture: { sampleType: 'unfilterable-float' } },
                { binding: 3, visibility, texture: { sampleType: 'float' } },
                { binding: 4, visibility, sampler: { type: 'filtering' } },
                { binding: 5, visibility, storageTexture: storage },
                { binding: 6, visibility, storageTexture: storage },
                { binding: 7, visibility, buffer: { type: 'uniform' } },
            ],
        });
        this._pipeline = device.createComputePipeline({
            label: 'TAA/Resolve',
            layout: device.createPipelineLayout({ label: 'TAA/Layout', bindGroupLayouts: [this._bgl] }),
            compute: { module: device.createShaderModule({ label: 'TAA/Shader', code: TAA_RESOLVE_WGSL }), entryPoint: 'main' },
        });
        this._params = device.createBuffer({
            label: 'TAA/Params',
            size: PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._sampler = device.createSampler({
            label: 'TAA/Sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });
        this._createHistory(1, 1);
        this.initialized = true;
    }

    private _createHistory(width: number, height: number): void {
        for (const t of this._history) t.destroy();
        const texture = () => this._device!.createTexture({
            label: 'TAA/History',
            size: [width, height],
            format: HISTORY_FORMAT,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this._history = [texture(), texture()];
        this._size = [width, height];
        this._read = 0;
        this._bindGroups = [null, null];
        this._hasHistory = false;
    }

    /** Writes the resolve's parameters for an output of `width` x `height` from a frame rendered at `input`. */
    private _writeParams(camera: Camera, input: [number, number], width: number, height: number, hasVelocity: boolean): void {
        const f32 = new Float32Array(this._paramsData);
        const u32 = new Uint32Array(this._paramsData);
        const m = this._scratch;
        // the inverse of this frame's jittered view-projection (the depth's)
        camera.jitteredProjection(m);
        mat4.multiply(m, m, camera.viewMatrix.internalMat4);
        mat4.invert(m, m);
        f32.set(m, 0);
        const viewProj = camera.viewProjection(f32.subarray(16, 32));
        f32.set(camera.previousViewProjection() ?? viewProj, 32);
        const o = this.options;
        f32[48] = camera.jitter[0] * input[0] * 0.5;
        f32[49] = -camera.jitter[1] * input[1] * 0.5;
        f32[50] = width;
        f32[51] = height;
        f32[52] = Math.min(Math.max(o.feedbackMin, 0), 0.99);
        f32[53] = Math.min(Math.max(o.feedbackMax, 0), 0.99);
        f32[54] = Math.max(o.exposure, 1e-8);
        f32[55] = Math.max(o.varianceGamma, 0.1);
        u32[56] = this._hasHistory ? 1 : 0;
        u32[57] = hasVelocity ? 1 : 0;
        f32[58] = input[0];
        f32[59] = input[1];
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
        _emissive?: GPUTexture,
        gbuffer?: GBuffer,
    ): void {
        if (!this._pipeline || !gbuffer) return;
        const device = this._device!;
        // the history is at the output size, and starts over when that changes, so a new render
        // scale keeps it
        if (this._size[0] !== width || this._size[1] !== height) this._createHistory(width, height);

        // the input is at the GBuffer's size (the effects before the upscaler run at it)
        this._writeParams(camera, [gbuffer.width, gbuffer.height], width, height, true);
        device.queue.writeBuffer(this._params!, 0, this._paramsData);

        const read = this._read;
        const write = 1 - read;
        const velocity = gbuffer.velocityTexture;
        const key = [input, depth, velocity, output, this._history[read]];
        let cached = this._bindGroups[read];
        if (!cached || cached.key.some((t, i) => t !== key[i])) {
            cached = {
                key,
                group: device.createBindGroup({
                    label: 'TAA/BG',
                    layout: this._bgl!,
                    entries: [
                        { binding: 0, resource: input.createView() },
                        { binding: 1, resource: depth.createView() },
                        { binding: 2, resource: velocity.createView() },
                        { binding: 3, resource: this._history[read].createView() },
                        { binding: 4, resource: this._sampler! },
                        { binding: 5, resource: output.createView() },
                        { binding: 6, resource: this._history[write].createView() },
                        { binding: 7, resource: { buffer: this._params! } },
                    ],
                }),
            };
            this._bindGroups[read] = cached;
        }

        const pass = commandEncoder.beginComputePass({ label: 'TAA/Resolve', timestampWrites: gpuPass('TAA/Resolve') });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, cached.group);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        this._read = write;
        this._hasHistory = true;
    }

    // The bind groups follow their textures (see `render`); the history keeps its output size.
    resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    wantsJitter(): boolean {
        return true;
    }

    upscalesToDisplay(): boolean {
        return true;
    }

    destroy(): void {
        for (const t of this._history) t.destroy();
        this._params?.destroy();
        this._history = [];
        this._params = null;
        this._pipeline = null;
        this._bindGroups = [null, null];
        this._size = [0, 0];
        this.initialized = false;
    }
}

export { TemporalAAEffect };
