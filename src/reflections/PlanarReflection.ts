import { mat4, vec4 } from "gl-matrix";
import { Camera } from "../cameras/Camera";
import { Texture } from "../buffers/Texture";
import { Vector3 } from "../math/Vector3";
import { GBuffer } from "../postprocessing/GBuffer";
import { gpuPass } from "../profiling/Profiler";
import type { Renderer } from "../renderers/Renderer";
import { assemble } from "../materials/shaders/ShaderUtils";
import { SCREEN_SPACE_PARAMS_BYTES, ScreenSpaceProjection } from "./ScreenSpaceProjection";
import froxelCommon from "../../rust/kansei-core/src/shaders/froxel_common.wgsl?raw";
import resolveWgsl from "../../rust/kansei-core/src/shaders/planar_reflection_resolve.wgsl?raw";
import downsampleWgsl from "../../rust/kansei-core/src/shaders/planar_reflection_downsample.wgsl?raw";
import sampleWgsl from "../../rust/kansei-core/src/shaders/planar_reflection_sample.wgsl?raw";

/**
 * WGSL for sampling a `PlanarReflection` in a material: `kansei_screen_uv`,
 * `kansei_reflection_offset` (ripples) and `kansei_planar_reflection` (roughness picks the mip).
 * Rust: `reflections::PLANAR_REFLECTION_WGSL`.
 */
export const PLANAR_REFLECTION_WGSL: string = sampleWgsl;

const RESOLVE_WGSL = assemble([froxelCommon, resolveWgsl]);
const DOWNSAMPLE_WGSL = downsampleWgsl;

/** Bytes of the WGSL `ResolveParams` and `ReflectionFogParams`. */
const RESOLVE_PARAMS_BYTES = 96;
export const REFLECTION_FOG_PARAMS_BYTES = 80;

type Vec3 = [number, number, number];

/** glam's `signum`: 1 for +0 and above, -1 for -0 and below. */
function signum(x: number): number {
    return x < 0 || Object.is(x, -0) ? -1 : 1;
}

/** Reflection about the plane `n·p + d = 0` (`n` unit length). */
export function reflectionMatrix(n: ArrayLike<number>, d: number): mat4 {
    const [x, y, z] = [n[0], n[1], n[2]];
    return mat4.fromValues(
        1 - 2 * x * x, -2 * x * y, -2 * x * z, 0,
        -2 * x * y, 1 - 2 * y * y, -2 * y * z, 0,
        -2 * x * z, -2 * y * z, 1 - 2 * z * z, 0,
        -2 * x * d, -2 * y * d, -2 * z * d, 1,
    );
}

/**
 * Replace the near plane of a `[0, 1]`-depth perspective projection by `clipPlane` (view space;
 * points with `plane · (p, 1) >= 0` are kept), after Lengyel, "Oblique View Frustum Depth
 * Projection and Clipping" (2005). Clipping happens in the rasterizer, so no shader needs a clip
 * distance.
 */
export function obliqueNearPlane(projection: ArrayLike<number>, clipPlane: ArrayLike<number>): mat4 {
    // the frustum corner opposite the plane, which must stay on the far plane (z_ndc = 1)
    const inverse = mat4.invert(mat4.create(), projection as mat4);
    const q = vec4.transformMat4(vec4.create(), [signum(clipPlane[0]), signum(clipPlane[1]), 1, 1], inverse);
    const row3 = [projection[3], projection[7], projection[11], projection[15]];
    const scale = vec4.dot(row3 as vec4, q) / vec4.dot(clipPlane as vec4, q);
    const m = mat4.clone(projection as mat4);
    m[2] = clipPlane[0] * scale;
    m[6] = clipPlane[1] * scale;
    m[10] = clipPlane[2] * scale;
    m[14] = clipPlane[3] * scale;
    return m;
}

/** The mirrored view of `view` across the plane `n·p + d = 0`, without the oblique near plane. */
export function mirroredView(view: ArrayLike<number>, n: ArrayLike<number>, d: number): mat4 {
    return mat4.multiply(mat4.create(), view as mat4, reflectionMatrix(n, d));
}

/**
 * Where a box is on screen: wholly outside the view, within a rectangle in screen uv
 * (x0, y0, x1, y1; y down), or unknown because it reaches behind the eye (the whole screen).
 */
export type ScreenRect = { kind: 'offscreen' } | { kind: 'rect'; rect: [number, number, number, number] } | { kind: 'unbounded' };

/**
 * The screen rectangle of the box `lo`..`hi` seen through `viewProj`, widened by `margin` (uv)
 * and clamped to the screen.
 */
export function screenRect(viewProj: ArrayLike<number>, lo: Vec3, hi: Vec3, margin: number): ScreenRect {
    let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
    let behind = 0;
    const clip = vec4.create();
    for (let k = 0; k < 8; k++) {
        vec4.transformMat4(clip, [k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2], 1], viewProj as mat4);
        if (clip[3] <= 1e-4) {
            behind++;
            continue;
        }
        const x = clip[0] / clip[3], y = clip[1] / clip[3];
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }
    if (behind === 8) return { kind: 'offscreen' };
    if (behind > 0) return { kind: 'unbounded' };
    // ndc -> uv (y down), widened, clamped
    const u0 = minX * 0.5 + 0.5 - margin, u1 = maxX * 0.5 + 0.5 + margin;
    const v0 = 0.5 - maxY * 0.5 - margin, v1 = 0.5 - minY * 0.5 + margin;
    if (u1 <= 0 || u0 >= 1 || v1 <= 0 || v0 >= 1) return { kind: 'offscreen' };
    return { kind: 'rect', rect: [Math.max(u0, 0), Math.max(v0, 0), Math.min(u1, 1), Math.min(v1, 1)] };
}

/**
 * Clip-space crop mapping the ndc rectangle x in [x0, x1], y in [y0, y1] to the whole of it:
 * frustum planes of `crop * viewProj` bound that part of the view.
 */
export function crop(x0: number, x1: number, y0: number, y1: number): mat4 {
    const sx = 2 / Math.max(x1 - x0, 1e-6), sy = 2 / Math.max(y1 - y0, 1e-6);
    return mat4.fromValues(
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, 1, 0,
        -(x0 + x1) * sx * 0.5, -(y0 + y1) * sy * 0.5, 0, 1,
    );
}

/** x negated in clip space: flips the mirrored view's winding back. */
export function flipX(): mat4 {
    return mat4.fromScaling(mat4.create(), [-1, 1, 1]);
}

/**
 * The volumetric fog as a planar reflection sees it (`VolumetricFogEffect.reflectionFog`): a
 * froxel volume built from the mirrored camera, holding only the fog above the mirror (the main
 * fog already covers the camera's path to the water). Give it to the reflection with
 * `PlanarReflection.setFog`; its resolve composites the fog over what the mirror saw, so a lake
 * mirrors the glow of beams and lamps in the mist. The volume is the one the fog built in the
 * previous frame, looked up by world position, so it stays in place as the camera moves.
 * Rust: `reflections::ReflectionFog`.
 */
export interface ReflectionFog {
    /** The fog's accumulated froxel volume (rgba16float, 3D). */
    readonly volume: GPUTexture;
    /** The WGSL `ReflectionFogParams` the resolve looks the volume up with. */
    readonly params: GPUBuffer;
    /**
     * Whether the reflection was drawn this frame (it is drawn before the post chain), so the fog
     * builds no volume for a reflection that is disabled or whose plane the camera is under.
     */
    readonly drawn: { value: boolean };
}

export interface PlanarReflectionOptions {
    /** Render-target size; half the canvas is typical. Default 960 x 540. */
    width?: number;
    height?: number;
    /**
     * Draw only renderables whose `layers` intersect this mask (leave out the water itself,
     * grass, small props). Default: every layer.
     */
    layerMask?: number;
    /**
     * Metres the clip plane sits below the reflecting plane, so geometry meeting the water
     * (shores, posts) reflects without a gap. Default 0.02.
     */
    clipBias?: number;
    /** Mip levels of the reflection texture for rough surfaces (1 = mirror only). Default 6. */
    mipLevels?: number;
}

/**
 * A mirror view of the scene across a plane (a lake, a wet floor), for materials to sample
 * (Rust `reflections::PlanarReflection`). The renderer draws every registered reflection each
 * frame (`Renderer.addPlanarReflection`), after its shadow maps and before the main pass, from the
 * camera mirrored in the plane, with an oblique near plane at the surface and only the renderables
 * on `layerMask`, through their materials' GBuffer pipelines. The result goes into a mip-mapped
 * texture: rgb radiance, a = the reflected path length for fog. Sample it in a material with
 * `PLANAR_REFLECTION_WGSL`.
 *
 * ```ts
 * const reflection = new PlanarReflection(renderer, new Vector3(0, 3, 0), new Vector3(0, 1, 0), {
 *     width: w / 2, height: h / 2, layerMask: ~WATER_LAYER,
 * });
 * // the water material binds reflection.materialTexture() and a linear sampler
 * const lake = renderer.addPlanarReflection(reflection);
 * ```
 */
export class PlanarReflection {
    /** A point on the reflecting plane. */
    public planePoint: Vector3;
    /** The plane's normal, pointing to the side that is reflected. */
    public planeNormal: Vector3;
    public layerMask: number;
    public clipBias: number;
    /** Skip rendering (the texture keeps its last contents). */
    public enabled: boolean = true;
    /**
     * Scales the distances by which instanced renderables (`InstanceCulling`) choose their LOD in
     * the mirrored view: below 1 it picks finer LODs than the camera. A mirror sees objects from
     * below, where coarse LODs built to read from the side (flat cards, dropped detail) show; 1
     * (the default) picks the camera's.
     */
    public lodDistanceScale: number = 1;
    /**
     * World-space bounds of the reflecting surface (its min and max corners), if known. Materials
     * sample the reflection by screen position, so it is then drawn only where the surface is on
     * screen: its pass is scissored to the surface's rectangle (plus `screenMargin`), its
     * instances are culled to that part of the view, and while the surface is off screen it is not
     * drawn at all.
     */
    public surfaceBounds: [Vec3, Vec3] | null = null;
    /**
     * Margin round the surface's screen rectangle, in screen uv: room for lookups displaced by
     * ripples and widened by roughness (0.05 by default).
     */
    public screenMargin: number = 0.05;
    /**
     * Reflect what the camera saw instead of drawing the mirrored view (off by default): last
     * frame's GBuffer, each pixel above the plane mirrored across it into this frame's view
     * (pixel-projected reflections), in one compute pass whatever the scene's geometry. What the
     * screen did not see (above its top edge, behind the camera, hidden from it) reads as sky,
     * which materials fill from their environment; after a camera cut (`Camera.resetMotion`) the
     * whole reflection does, for a frame. It needs a single-sampled GBuffer
     * (`PostProcessingVolume`, the default `msaaSampleCount`).
     */
    public screenSpace: boolean = false;

    public readonly width: number;
    public readonly height: number;

    private readonly _device: GPUDevice;
    private _active = false;
    // this frame's screen rectangle of the surface, in uv (x0, y0, x1, y1; y down), when bounded
    private _screenRect: [number, number, number, number] | null = null;
    private readonly _camera: Camera;
    // MRT targets matching the GBuffer, so the materials' GBuffer pipelines draw into them
    private readonly _targets: GPUTexture[];
    private readonly _depth: GPUTexture;
    // the sampled result, with its mip chain
    private readonly _texture: GPUTexture;
    private readonly _mip0View: GPUTextureView;
    private readonly _resolvePipeline: GPUComputePipeline;
    private readonly _resolveBGL: GPUBindGroupLayout;
    private _resolveBG: GPUBindGroup;
    private readonly _resolveParams: GPUBuffer;
    private readonly _resolveData = new ArrayBuffer(RESOLVE_PARAMS_BYTES);
    private readonly _fogSampler: GPUSampler;
    // no fog: an empty volume and parameters that say so
    private readonly _noFog: ReflectionFog;
    // the attached fog (the screen-space path binds it itself)
    private _fog: ReflectionFog | null = null;
    private readonly _downsamplePipeline: GPUComputePipeline;
    private readonly _downsampleBGs: GPUBindGroup[];
    private _screenSpaceProjection: ScreenSpaceProjection | null = null;
    private readonly _screenSpaceData = new ArrayBuffer(SCREEN_SPACE_PARAMS_BYTES);
    private readonly _viewProj = mat4.create();

    constructor(renderer: Renderer, planePoint: Vector3, planeNormal: Vector3, options: PlanarReflectionOptions = {}) {
        const device = renderer.gpuDevice;
        this._device = device;
        this.planePoint = planePoint;
        this.planeNormal = planeNormal;
        this.layerMask = options.layerMask ?? 0xffffffff;
        this.clipBias = options.clipBias ?? 0.02;
        const width = Math.max(Math.floor(options.width ?? 960), 1);
        const height = Math.max(Math.floor(options.height ?? 540), 1);
        this.width = width;
        this.height = height;
        const maxMips = Math.floor(Math.log2(Math.max(width, height))) + 1;
        const mipLevels = Math.min(Math.max(options.mipLevels ?? 6, 1), maxMips);

        const target = (label: string, format: GPUTextureFormat, usage: GPUTextureUsageFlags) =>
            device.createTexture({ label, size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | usage });
        const formats = GBuffer.MRT_FORMATS;
        this._targets = [
            target('PlanarReflection/Color', formats[0], GPUTextureUsage.TEXTURE_BINDING),
            target('PlanarReflection/Emissive', formats[1], 0),
            target('PlanarReflection/Normal', formats[2], 0),
            target('PlanarReflection/Albedo', formats[3], 0),
        ];
        this._depth = target('PlanarReflection/Depth', GBuffer.DEPTH_FORMAT, GPUTextureUsage.TEXTURE_BINDING);

        this._texture = device.createTexture({
            label: 'PlanarReflection/Texture',
            size: [width, height],
            mipLevelCount: mipLevels,
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        const mipViews = Array.from({ length: mipLevels }, (_, level) =>
            this._texture.createView({ label: 'PlanarReflection/Mip', baseMipLevel: level, mipLevelCount: 1 }));
        this._mip0View = mipViews[0];

        const compute = GPUShaderStage.COMPUTE;
        const storage: GPUStorageTextureBindingLayout = { access: 'write-only', format: 'rgba16float' };
        this._resolveBGL = device.createBindGroupLayout({
            label: 'PlanarReflection/ResolveBGL',
            entries: [
                { binding: 0, visibility: compute, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: compute, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: compute, storageTexture: storage },
                { binding: 3, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 4, visibility: compute, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 5, visibility: compute, sampler: { type: 'filtering' } },
                { binding: 6, visibility: compute, buffer: { type: 'uniform' } },
            ],
        });
        const downsampleBGL = device.createBindGroupLayout({
            label: 'PlanarReflection/DownsampleBGL',
            entries: [
                { binding: 0, visibility: compute, texture: { sampleType: 'float' } },
                { binding: 1, visibility: compute, sampler: { type: 'filtering' } },
                { binding: 2, visibility: compute, storageTexture: storage },
            ],
        });
        const pipeline = (label: string, code: string, layout: GPUBindGroupLayout) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
        });
        this._resolvePipeline = pipeline('PlanarReflection/Resolve', RESOLVE_WGSL, this._resolveBGL);
        this._downsamplePipeline = pipeline('PlanarReflection/Downsample', DOWNSAMPLE_WGSL, downsampleBGL);
        this._resolveParams = device.createBuffer({
            label: 'PlanarReflection/ResolveParams',
            size: RESOLVE_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._fogSampler = device.createSampler({ label: 'PlanarReflection/FogSampler', magFilter: 'linear', minFilter: 'linear' });
        this._noFog = {
            volume: device.createTexture({
                label: 'PlanarReflection/NoFog',
                size: [1, 1, 1],
                dimension: '3d',
                format: 'rgba16float',
                usage: GPUTextureUsage.TEXTURE_BINDING,
            }),
            // zeroed: enabled = 0
            params: device.createBuffer({ label: 'PlanarReflection/NoFogParams', size: REFLECTION_FOG_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM }),
            drawn: { value: false },
        };
        this._resolveBG = this._resolveBindGroup(this._noFog);
        const sampler = device.createSampler({ label: 'PlanarReflection/Sampler', magFilter: 'linear', minFilter: 'linear' });
        this._downsampleBGs = mipViews.slice(1).map((dst, i) => device.createBindGroup({
            label: 'PlanarReflection/DownsampleBG',
            layout: downsampleBGL,
            entries: [
                { binding: 0, resource: mipViews[i] },
                { binding: 1, resource: sampler },
                { binding: 2, resource: dst },
            ],
        }));

        this._camera = new Camera(60, 0.1, 1000, width / height);
        this._camera.useLightUniforms(renderer.lightUniforms);
    }

    private _resolveBindGroup(fog: ReflectionFog): GPUBindGroup {
        return this._device.createBindGroup({
            label: 'PlanarReflection/ResolveBG',
            layout: this._resolveBGL,
            entries: [
                { binding: 0, resource: this._targets[0].createView() },
                { binding: 1, resource: this._depth.createView() },
                { binding: 2, resource: this._mip0View },
                { binding: 3, resource: { buffer: this._resolveParams } },
                { binding: 4, resource: fog.volume.createView() },
                { binding: 5, resource: this._fogSampler },
                { binding: 6, resource: { buffer: fog.params } },
            ],
        });
    }

    /**
     * Composite a volumetric fog over the reflection (`VolumetricFogEffect.reflectionFog`), or
     * none. Materials then fog only what lies beyond the fog's volume along the reflected path.
     */
    public setFog(fog: ReflectionFog | null): void {
        this._fog = fog;
        this._resolveBG = this._resolveBindGroup(fog ?? this._noFog);
    }

    /** The plane as (unit normal, d) with `n·p + d = 0`. */
    public plane(): { n: Vec3; d: number } {
        const { x, y, z } = this.planeNormal;
        const length = Math.hypot(x, y, z);
        const n: Vec3 = length > 0 && Number.isFinite(length) ? [x / length, y / length, z / length] : [0, 1, 0];
        const p = this.planePoint;
        return { n, d: -(n[0] * p.x + n[1] * p.y + n[2] * p.z) };
    }

    /** The reflection (all mips) as a material bindable: `texture_2d<f32>` in WGSL. */
    public materialTexture(): Texture {
        return Texture.fromView('PlanarReflection', this._texture);
    }

    public setPlane(point: Vector3, normal: Vector3): void {
        this.planePoint = point;
        this.planeNormal = normal;
    }

    /** Whether the last frame rendered the reflection (the camera was on the reflected side). */
    public get isActive(): boolean {
        return this._active;
    }

    /**
     * Point the mirrored camera for this frame (the renderer calls it before culling). Returns
     * false when the main camera is not on the reflected side of the plane (nothing to render).
     */
    public updateCamera(main: Camera): boolean {
        const { n, d } = this.plane();
        const view = main.viewMatrix.internalMat4;
        const eye = main.inverseViewMatrix.internalMat4;
        this._active = this.enabled && n[0] * eye[12] + n[1] * eye[13] + n[2] * eye[14] + d > 0;
        // only the surface's part of the screen, and nothing while it is off screen
        this._screenRect = null;
        if (this._active && this.surfaceBounds) {
            const rect = screenRect(main.viewProjection(this._viewProj), this.surfaceBounds[0], this.surfaceBounds[1], this.screenMargin);
            if (rect.kind === 'offscreen') this._active = false;
            else if (rect.kind === 'rect') this._screenRect = rect.rect;
        }
        if (this._fog) this._fog.drawn.value = this._active;
        if (!this._active) return false;
        const mirrored = mirroredView(view, n, d);
        // clip plane (lowered by the bias) in the mirrored view space: planes transform by the
        // inverse transpose
        const inverse = mat4.invert(mat4.create(), mirrored);
        const planeView = vec4.transformMat4(vec4.create(), [n[0], n[1], n[2], d + this.clipBias], mat4.transpose(mat4.create(), inverse));
        const projection = obliqueNearPlane(main.projectionMatrix.internalMat4, planeView);
        // the mirror flips triangle winding; flipping x in clip space flips it back, so the
        // materials' back-face culling still works (samplers flip u back, see the WGSL helper)
        const camera = this._camera;
        mat4.copy(camera.viewMatrix.internalMat4, mirrored);
        camera.viewMatrix.syncBuffer();
        mat4.copy(camera.inverseViewMatrix.internalMat4, inverse);
        camera.inverseViewMatrix.syncBuffer();
        mat4.multiply(camera.projectionMatrix.internalMat4, flipX(), projection);
        camera.projectionMatrix.syncBuffer();
        camera.uploadTemporal();
        return true;
    }

    /** The mirrored camera, as `updateCamera` last placed it. */
    public get camera(): Camera {
        return this._camera;
    }

    /**
     * The view-projection to cull this frame's instances with: the mirrored camera's, cropped to
     * the surface's part of the screen.
     */
    public cullViewProj(out: mat4 = mat4.create()): mat4 {
        const viewProj = this._camera.viewProjection(out);
        const rect = this._screenRect;
        if (!rect) return viewProj;
        // screen uv -> the mirrored view's ndc: x flipped (see `flipX`), y up
        const [u0, v0, u1, v1] = rect;
        return mat4.multiply(out, crop(-(2 * u1 - 1), -(2 * u0 - 1), 1 - 2 * v1, 1 - 2 * v0), viewProj);
    }

    /** This frame's scissor rectangle in the render target (x, y, width, height), when bounded. */
    public scissor(): [number, number, number, number] | null {
        const rect = this._screenRect;
        if (!rect) return null;
        const [u0, v0, u1, v1] = rect;
        const w = this.width, h = this.height;
        const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
        // the target is mirrored left-right
        const x0 = clamp(Math.floor((1 - u1) * w), 0, w - 1);
        const x1 = clamp(Math.ceil((1 - u0) * w), x0 + 1, w);
        const y0 = clamp(Math.floor(v0 * h), 0, h - 1);
        const y1 = clamp(Math.ceil(v1 * h), y0 + 1, h);
        return [x0, y0, x1 - x0, y1 - y0];
    }

    /** The GBuffer-format targets the mirrored view is drawn into (colour, emissive, normal, albedo). */
    public get colorTargets(): readonly GPUTexture[] {
        return this._targets;
    }

    /** The mirrored view's depth target (`GBuffer.DEPTH_FORMAT`). */
    public get depthTarget(): GPUTexture {
        return this._depth;
    }

    /** Resolve the render into mip 0 (with the path length in alpha), then build the mips. */
    public resolve(encoder: GPUCommandEncoder): void {
        const camera = this._camera;
        const inverse = mat4.invert(mat4.create(), camera.viewProjection(this._viewProj));
        const eye = camera.inverseViewMatrix.internalMat4;
        const f32 = new Float32Array(this._resolveData);
        f32.set(inverse, 0);
        f32.set([eye[12], eye[13], eye[14], 0], 16);
        new Uint32Array(this._resolveData).set([this.width, this.height, 0, 0], 20);
        this._device.queue.writeBuffer(this._resolveParams, 0, this._resolveData);
        const pass = encoder.beginComputePass({ label: 'PlanarReflection/Resolve', timestampWrites: gpuPass('PlanarReflection/Resolve') });
        pass.setPipeline(this._resolvePipeline);
        pass.setBindGroup(0, this._resolveBG);
        pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        this._buildMips(pass);
        pass.end();
    }

    private _buildMips(pass: GPUComputePassEncoder): void {
        pass.setPipeline(this._downsamplePipeline);
        this._downsampleBGs.forEach((bg, level) => {
            const w = Math.max(this.width >> (level + 1), 1), h = Math.max(this.height >> (level + 1), 1);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        });
    }

    /**
     * The screen-space path (`screenSpace`): project `gbuffer`'s colour and depth, still last
     * frame's (`camera`'s previous view), into mip 0, then build the mips.
     */
    public projectScreenSpace(encoder: GPUCommandEncoder, camera: Camera, gbuffer: GBuffer): void {
        const { n, d } = this.plane();
        const viewProj = camera.viewProjection(mat4.create());
        const eye = camera.inverseViewMatrix.internalMat4;
        // no last frame to project (a camera cut, the first frame): an empty rectangle, all sky
        const prev = screenSpaceSource(camera);
        const rect = prev ? (this._screenRect ?? [0, 0, 1, 1]) : [1, 1, 0, 0];
        const data = this._screenSpaceData;
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32.set(mat4.invert(mat4.create(), prev ?? viewProj), 0);
        f32.set(viewProj, 16);
        f32.set([n[0], n[1], n[2], d], 32);
        f32.set([eye[12], eye[13], eye[14]], 36);
        f32[39] = Math.max(this.clipBias, 0);
        u32.set([gbuffer.width, gbuffer.height, this.width, this.height], 40);
        f32.set(rect, 44);
        this._screenSpaceProjection ??= new ScreenSpaceProjection(this._device, this.width, this.height);
        const fog = this._fog ?? this._noFog;
        this._screenSpaceProjection.run(encoder, gbuffer.colorTexture, gbuffer.depthTexture, this._mip0View, fog, this._fogSampler, data);
        const pass = encoder.beginComputePass({ label: 'PlanarReflection/Mips' });
        this._buildMips(pass);
        pass.end();
    }

    /** Free the reflection's targets, texture and buffers. */
    public destroy(): void {
        for (const t of [...this._targets, this._depth, this._texture, this._noFog.volume]) t.destroy();
        this._resolveParams.destroy();
        this._noFog.params.destroy();
        this._screenSpaceProjection?.destroy();
        this._screenSpaceProjection = null;
    }
}

/**
 * The view last frame's GBuffer was drawn with (jittered, with TAA), which the screen-space path
 * reconstructs its pixels' world positions with; null without a last frame.
 */
export function screenSpaceSource(camera: Camera): mat4 | null {
    return camera.previousJitteredViewProjection();
}
