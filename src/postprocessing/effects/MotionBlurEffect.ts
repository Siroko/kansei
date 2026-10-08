import { mat4 } from 'gl-matrix';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { MOTION_BLUR_GATHER_WGSL, MOTION_BLUR_NEIGHBOURS_WGSL, MOTION_BLUR_PREPARE_WGSL } from '../../materials/shaders/SharedWGSL';
import { gpuPass } from '../../profiling/Profiler';

/** Pixels per tile side (the shaders' TILE, the prepare pass' workgroup size). */
const TILE = 16;
/** `MotionBlurParams` in `motion_blur_common.wgsl`: three mat4, then 8 scalars (Rust `MotionBlurParamsGpu`). */
const PARAMS_BYTES = 224;
const TARGET_FORMAT: GPUTextureFormat = 'rgba16float';

export interface MotionBlurOptions {
    /**
     * Fraction of the frame interval the shutter is open, as Unreal's Motion Blur Amount: 0.5
     * is a 180-degree shutter, 0 turns the blur off.
     */
    amount: number;
    /**
     * Largest blur, as a fraction of the image width, measured from the pixel to the end of its
     * streak (Unreal's Motion Blur Max / 100; its default is 5 %). Faster motion is clamped to
     * it, so the picture looks the same at any resolution.
     */
    max: number;
    /** Gather samples per pixel, in mirrored pairs (rounded up to even). */
    sampleCount: number;
    /**
     * Unreal's Motion Blur Target FPS: blur as if the frame rate were this, so the streaks don't
     * depend on the rate the browser runs at. Needs the frame's duration each frame
     * (`setFrameTime`); null blurs by each rendered frame's motion.
     */
    targetFps: number | null;
}

export function defaultMotionBlurOptions(): MotionBlurOptions {
    return { amount: 0.5, max: 0.05, sampleCount: 16, targetFps: null };
}

/** Per-pixel blur vectors and the two tile textures, at one output size. */
interface Targets {
    width: number;
    height: number;
    /** Per pixel: blur vector (px) and view depth. */
    motion: GPUTexture;
    /** Per tile: its longest blur vector, then the longest that reaches it. */
    tiles: GPUTexture;
    neighbours: GPUTexture;
}

/**
 * Camera and object motion blur, as Unreal's (McGuire et al. 2012, Jimenez 2014; Rust
 * `MotionBlurEffect`, `postprocessing/effects/motion_blur.rs`, whose WGSL it runs): each pixel's
 * motion is the GBuffer's velocity where a material writes one (`outputsVelocity`), else the
 * camera's reprojection of its depth. The longest motion per 16x16 tile, spread to the tiles it
 * reaches, gives each pixel a dominant direction; the pixel gathers jittered samples along it
 * (and along its own motion), weighted by depth and by how far each streak reaches, so moving
 * objects smear over what is behind them, the background shows through their smeared edges,
 * and nothing static bleeds over a sharp foreground. Tiles without motion copy the input and
 * tiles where everything moves alike take a plain average, so a still frame costs three light
 * passes.
 *
 * It works at the size it is given (after `TemporalAAEffect` under a render scale, the display
 * size): each output pixel reads the depth and velocity texel under it, blur lengths and the cap
 * are in output pixels, and the tiles cover the output.
 *
 * A cut is not motion: the effect copies the input for a frame after `Camera.resetMotion` (as
 * the TAA reprojects nothing across it) or after `reset`.
 *
 * Put it after `TemporalAAEffect` and before depth of field, bloom and the tonemapper:
 * - after the TAA, whose history must stay sharp (reprojecting blurred frames smears them again
 *   and breaks its neighbourhood clamp), and whose resolve gives the blur an unjittered,
 *   anti-aliased image;
 * - before the depth of field, while the colour still lines up with the depth and velocity the
 *   weights classify it by (the DoF spreads colour past the depth silhouettes);
 * - before bloom and the tonemapper, on scene-linear light, so highlights streak with their full
 *   energy as they do on film.
 *
 * ```ts
 * const blur = new MotionBlurEffect({ amount: 0.5, targetFps: 30 });
 * const volume = new PostProcessingVolume(renderer, [taa, blur, tonemap]);
 * // every frame, with targetFps:
 * blur.setFrameTime(frame.dt);
 * ```
 */
class MotionBlurEffect extends PostProcessingEffect {
    public options: MotionBlurOptions;

    private _frameTime = 0;
    private _skipNext = false;
    private _frame = 0;
    private _device: GPUDevice | null = null;
    private _params: GPUBuffer | null = null;
    private _prepare: GPUComputePipeline | null = null;
    private _neighbours: GPUComputePipeline | null = null;
    private _gather: GPUComputePipeline | null = null;
    private _prepareBgl: GPUBindGroupLayout | null = null;
    private _neighboursBgl: GPUBindGroupLayout | null = null;
    private _gatherBgl: GPUBindGroupLayout | null = null;
    private _targets: Targets | null = null;
    private readonly _paramsData = new ArrayBuffer(PARAMS_BYTES);
    private readonly _scratch = mat4.create();
    /** The bind groups, and the textures each was made for. */
    private _bindGroups: { prepare: GPUBindGroup; neighbours: GPUBindGroup; gather: GPUBindGroup; key: GPUTexture[] } | null = null;

    constructor(options: Partial<MotionBlurOptions> = {}) {
        super();
        this.options = { ...defaultMotionBlurOptions(), ...options };
    }

    /** No blur next frame (camera cuts, teleports). `Camera.resetMotion` does this too. */
    public reset(): void {
        this._skipNext = true;
    }

    /** The duration of the frame about to be rendered, seconds (for `targetFps`). */
    public setFrameTime(seconds: number): void {
        this._frameTime = seconds;
    }

    /**
     * Per-frame motion in pixels -> blur radius in pixels: half the shutter's share of the motion
     * (the streak runs both ways from the pixel), rescaled to `targetFps` if set.
     */
    public blurScale(): number {
        const fps = this.options.targetFps;
        const timeScale = fps !== null && fps > 0 && this._frameTime > 1e-4 ? 1 / (fps * this._frameTime) : 1;
        return 0.5 * Math.max(this.options.amount, 0) * timeScale;
    }

    /** Largest blur radius in pixels for an image `widthPx` wide. */
    public maxRadiusPx(widthPx: number): number {
        return Math.max(this.options.max, 0) * widthPx;
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const visibility = GPUShaderStage.COMPUTE;
        const tex = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'unfilterable-float' } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth' } });
        const storage = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, storageTexture: { access: 'write-only', format: TARGET_FORMAT } });
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });
        this._prepareBgl = bgl('MotionBlur/PrepareBGL', [depth(0), tex(1), storage(2), storage(3), uniform(4)]);
        this._neighboursBgl = bgl('MotionBlur/NeighboursBGL', [tex(0), storage(1), uniform(2)]);
        this._gatherBgl = bgl('MotionBlur/GatherBGL', [tex(0), tex(1), tex(2), storage(3), uniform(4)]);
        const pipeline = (label: string, code: string, layout: GPUBindGroupLayout) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
        });
        this._prepare = pipeline('MotionBlur/Prepare', MOTION_BLUR_PREPARE_WGSL, this._prepareBgl);
        this._neighbours = pipeline('MotionBlur/Neighbours', MOTION_BLUR_NEIGHBOURS_WGSL, this._neighboursBgl);
        this._gather = pipeline('MotionBlur/Gather', MOTION_BLUR_GATHER_WGSL, this._gatherBgl);
        this._params = device.createBuffer({
            label: 'MotionBlur/Params',
            size: PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.initialized = true;
    }

    private _ensureTargets(width: number, height: number): Targets {
        const t = this._targets;
        if (t && t.width === width && t.height === height) return t;
        this._destroyTargets();
        const target = (label: string, w: number, h: number) => this._device!.createTexture({
            label,
            size: [Math.max(w, 1), Math.max(h, 1)],
            format: TARGET_FORMAT,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        const tw = Math.ceil(width / TILE), th = Math.ceil(height / TILE);
        this._targets = {
            width,
            height,
            motion: target('MotionBlur/Motion', width, height),
            tiles: target('MotionBlur/Tiles', tw, th),
            neighbours: target('MotionBlur/Neighbours', tw, th),
        };
        return this._targets;
    }

    private _destroyTargets(): void {
        const t = this._targets;
        if (!t) return;
        t.motion.destroy();
        t.tiles.destroy();
        t.neighbours.destroy();
        this._targets = null;
        this._bindGroups = null;
    }

    private _writeParams(camera: Camera, width: number, height: number, enabled: boolean): void {
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
        f32[48] = width;
        f32[49] = height;
        f32[50] = this.blurScale();
        f32[51] = this.maxRadiusPx(width);
        u32[52] = Math.max(Math.ceil(this.options.sampleCount / 2), 1);
        u32[53] = this._frame;
        u32[54] = enabled ? 1 : 0;
        u32[55] = 0;
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
        if (!this._gather || !gbuffer) return;
        const device = this._device!;
        const t = this._ensureTargets(width, height);
        // a cut (no previous view) or an explicit reset: nothing moved as far as the shutter saw
        const skip = this._skipNext;
        this._skipNext = false;
        const enabled = !skip
            && camera.previousViewProjection() !== null
            && this.blurScale() > 0
            && this.maxRadiusPx(width) >= 0.5;
        this._writeParams(camera, width, height, enabled);
        this._frame = (this._frame + 1) >>> 0;
        device.queue.writeBuffer(this._params!, 0, this._paramsData);

        const velocity = gbuffer.velocityTexture;
        const key = [input, depth, velocity, output, t.motion];
        let groups = this._bindGroups;
        if (!groups || groups.key.some((k, i) => k !== key[i])) {
            const params = { buffer: this._params! };
            const group = (layout: GPUBindGroupLayout, resources: GPUBindingResource[]) => device.createBindGroup({
                label: 'MotionBlur/BG',
                layout,
                entries: resources.map((resource, binding) => ({ binding, resource })),
            });
            groups = {
                key,
                prepare: group(this._prepareBgl!, [depth.createView(), velocity.createView(), t.motion.createView(), t.tiles.createView(), params]),
                neighbours: group(this._neighboursBgl!, [t.tiles.createView(), t.neighbours.createView(), params]),
                gather: group(this._gatherBgl!, [input.createView(), t.motion.createView(), t.neighbours.createView(), output.createView(), params]),
            };
            this._bindGroups = groups;
        }
        const tw = Math.ceil(width / TILE), th = Math.ceil(height / TILE);

        const pass = commandEncoder.beginComputePass({ label: 'MotionBlur', timestampWrites: gpuPass('MotionBlur') });
        if (enabled) {
            pass.setPipeline(this._prepare!);
            pass.setBindGroup(0, groups.prepare);
            pass.dispatchWorkgroups(tw, th);
            pass.setPipeline(this._neighbours!);
            pass.setBindGroup(0, groups.neighbours);
            pass.dispatchWorkgroups(Math.ceil(tw / 8), Math.ceil(th / 8));
        }
        pass.setPipeline(this._gather);
        pass.setBindGroup(0, groups.gather);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    // The targets follow the size `render` is given; the bind groups follow their textures.
    resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    destroy(): void {
        this._destroyTargets();
        this._params?.destroy();
        this._params = null;
        this._prepare = this._neighbours = this._gather = null;
        this.initialized = false;
    }
}

export { MotionBlurEffect };
