import { mat4, vec3 } from 'gl-matrix';
import type { Camera } from '../cameras/Camera';
import type { Renderable } from '../objects/Renderable';
import type { DepthBias, Material } from '../materials/Material';
import type { CascadedShadowSource } from './ComputeShadows';
import { drawGeometry } from '../culling/InstanceCulling';
import { InstancedGeometry } from '../geometries/InstancedGeometry';
import { gpuPass } from '../profiling/Profiler';
import { BindGroupSlot, CAMERA_TEMPORAL_BYTES, CASCADES_BYTES, LIGHT_UNIFORM_BYTES, cameraBindGroupLayoutEntries } from '../renderers/SharedLayouts';

export const MAX_CASCADES = 4;

/** `Renderer.enableCascadedShadows`' options; the defaults are Rust's `CascadedShadowOptions::default()`. */
export interface CascadedShadowOptions {
    /** Cascades, 1 to 4. Default 4. */
    cascades?: number;
    /** Resolution of each cascade's square map. Default 2048. */
    resolution?: number;
    /** View distance the last cascade reaches; shadows fade out over its last tenth. Default 250. */
    maxDistance?: number;
    /** Split scheme between uniform (0) and logarithmic (1) cascade depths. Default 0.75. */
    splitLambda?: number;
    /**
     * Metres toward the light that casters are still drawn beyond each cascade's bounds, so tall
     * things outside the view keep their shadows. Default 200.
     */
    casterDistance?: number;
    /**
     * Apparent diameter of the light in radians, for contact-hardening (PCSS) penumbrae: the sun
     * is 0.0093 (the default). 0 gives a fixed small PCF kernel.
     */
    lightAngularDiameter?: number;
    /** Receiver offset along the normal, in texels of the cascade it is looked up in. Default 1.5. */
    normalBias?: number;
    /** Fraction of a cascade's half-width over which it dithers into the next one. Default 0.15. */
    blend?: number;
}

/** One cascade this frame: its light view and orthographic projection. */
export interface CascadeSlot {
    view: mat4;
    projection: mat4;
    radius: number;
    depthRange: number;
}

function resolveOptions(o: CascadedShadowOptions): Required<CascadedShadowOptions> {
    return {
        cascades: Math.min(Math.max(Math.floor(o.cascades ?? 4), 1), MAX_CASCADES),
        resolution: o.resolution ?? 2048,
        maxDistance: o.maxDistance ?? 250,
        splitLambda: o.splitLambda ?? 0.75,
        casterDistance: o.casterDistance ?? 200,
        lightAngularDiameter: o.lightAngularDiameter ?? 0.0093,
        normalBias: o.normalBias ?? 1.5,
        blend: o.blend ?? 0.15,
    };
}

/**
 * The view depths splitting `[near, maxDistance]` into `count` cascades: a blend of uniform and
 * logarithmic splits (the "practical" scheme, Zhang et al. 2006). Returns count + 1 depths.
 */
export function cascadeSplits(near: number, maxDistance: number, count: number, lambda: number): number[] {
    return Array.from({ length: count + 1 }, (_, i) => {
        const t = i / count;
        const log = near * Math.pow(maxDistance / near, t);
        const uniform = near + (maxDistance - near) * t;
        return lambda * log + (1 - lambda) * uniform;
    });
}

/**
 * The smallest sphere around the part of a view frustum between view depths `near` and `far`
 * (`tanDiag`: tangent of the half-angle to the frustum's corners): its centre's depth along the
 * view axis and its radius. It depends only on the depths and the lens, so it doesn't change as
 * the camera turns: the cascade keeps its size, and its texels stay put.
 */
export function frustumSliceSphere(near: number, far: number, tanDiag: number): [number, number] {
    const t2 = tanDiag * tanDiag;
    const center = Math.min((near + far) * (1 + t2) * 0.5, far);
    const radius = Math.max(
        Math.sqrt((center - near) ** 2 + near * near * t2),
        Math.sqrt((far - center) ** 2 + far * far * t2),
    );
    return [center, radius];
}

/**
 * The cascades for `camera` and a light travelling along `lightDir`: per cascade, the bounding
 * sphere of its frustum slice, its centre snapped to whole texels in light space, an orthographic
 * box around it reaching `casterDistance` further toward the light (Rust `fit_cascades`).
 */
export function fitCascades(options: CascadedShadowOptions, camera: Camera, lightDir: ArrayLike<number>): CascadeSlot[] {
    const o = resolveOptions(options);
    const dir = vec3.fromValues(lightDir[0], lightDir[1], lightDir[2]);
    if (vec3.length(dir) < 1e-6) vec3.set(dir, 0, -1, 0);
    vec3.normalize(dir, dir);
    const up: vec3 = Math.abs(dir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
    // look_to_rh(0, dir, up): world to a light space looking down -z along `dir`
    const rotation = mat4.lookAt(mat4.create(), [0, 0, 0], dir, up);
    const inv = camera.inverseViewMatrix.internalMat4;
    const eye = vec3.fromValues(inv[12], inv[13], inv[14]);
    const forward = vec3.fromValues(-inv[8], -inv[9], -inv[10]);
    if (vec3.length(forward) < 1e-6) vec3.set(forward, 0, 0, -1);
    vec3.normalize(forward, forward);
    const tanV = Math.tan(camera.fov * Math.PI / 180 * 0.5);
    const tanDiag = tanV * Math.sqrt(1 + camera.aspect * camera.aspect);
    const splits = cascadeSplits(camera.near, Math.max(o.maxDistance, camera.near * 2), o.cascades, o.splitLambda);
    const slots: CascadeSlot[] = [];
    for (let c = 0; c < o.cascades; c++) {
        const [depth, radius] = frustumSliceSphere(splits[c], splits[c + 1], tanDiag);
        const center = vec3.scaleAndAdd(vec3.create(), eye, forward, depth);
        const texel = 2 * radius / o.resolution;
        const centerLs = vec3.transformMat4(vec3.create(), center, rotation);
        centerLs[0] = Math.floor(centerLs[0] / texel) * texel;
        centerLs[1] = Math.floor(centerLs[1] / texel) * texel;
        // the eye sits `radius + casterDistance` toward the light from the centre
        const back = radius + o.casterDistance;
        const view = mat4.fromTranslation(mat4.create(), [-centerLs[0], -centerLs[1], -(centerLs[2] + back)]);
        mat4.multiply(view, view, rotation);
        const depthRange = back + radius;
        const projection = mat4.orthoZO(mat4.create(), -radius, radius, -radius, radius, 0, depthRange);
        slots.push({ view, projection, radius, depthRange });
    }
    return slots;
}

/**
 * Cascaded shadow maps for the scene's first directional light when it casts shadows (the sun,
 * or the moon), Rust's `shadows::CascadedShadowMap`: stable cascades (bounding spheres that keep
 * their size as the camera turns, light-space centres snapped to whole texels, so shadows don't
 * shimmer), rendered each frame through the casters' own vertex shaders
 * (`Material.getDepthPipeline`) and culled per cascade on the GPU, and looked up with
 * contact-hardening PCSS and dithered transitions (`CASCADED_SHADOWS_WGSL`). Create it with
 * `Renderer.enableCascadedShadows`.
 *
 * The widest cascade also stands in for the single directional shadow map: materials that read
 * group 3 binding 0 (`kansei_shadow_map`) and the volumetric fog (`setCascadedShadowMap`) use it.
 */
class CascadedShadowMap implements CascadedShadowSource {
    static readonly FORMAT: GPUTextureFormat = 'depth32float';
    /** Depth bias of the cascade pipelines (the shader adds a normal offset on top). */
    static readonly DEPTH_BIAS: DepthBias = { constant: 0, slopeScale: 2.0, clamp: 0 };

    readonly options: Readonly<Required<CascadedShadowOptions>>;
    readonly texture: GPUTexture;
    /** All cascades (`texture_depth_2d_array`). */
    readonly arrayView: GPUTextureView;
    /** The widest cascade alone, as a `texture_depth_2d`. */
    readonly farView: GPUTextureView;
    /** The widest cascade's view-projection (a `mat4x4f` uniform), rewritten every frame. */
    readonly farViewProj: GPUBuffer;
    /** `KanseiCascades` (group 3 binding 11). */
    readonly uniform: GPUBuffer;
    /** This frame's cascades (`fit`); empty while no light casts. */
    slots: CascadeSlot[] = [];

    private readonly _device: GPUDevice;
    private readonly _layerViews: GPUTextureView[];
    // One camera group per cascade (group 1 layout): its view and projection, then the scene
    // lights and temporal data, which depth pipelines do not read (shared, zeroed).
    private readonly _buffers: GPUBuffer[] = [];
    private readonly _viewBuffers: GPUBuffer[];
    private readonly _projectionBuffers: GPUBuffer[];
    private readonly _cameraBindGroups: GPUBindGroup[];
    private readonly _data = new ArrayBuffer(CASCADES_BYTES);
    private readonly _farViewProj = mat4.create();

    constructor(device: GPUDevice, options: CascadedShadowOptions = {}) {
        this._device = device;
        this.options = resolveOptions(options);
        const { cascades: count, resolution } = this.options;
        this.texture = device.createTexture({
            label: 'CascadedShadowMap',
            size: [resolution, resolution, count],
            format: CascadedShadowMap.FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.arrayView = this.texture.createView({ label: 'CascadedShadowMap/Array', dimension: '2d-array' });
        const layer = (l: number) => this.texture.createView({
            label: `CascadedShadowMap/Cascade${l}`,
            dimension: '2d',
            baseArrayLayer: l,
            arrayLayerCount: 1,
        });
        this._layerViews = Array.from({ length: count }, (_, l) => layer(l));
        this.farView = layer(count - 1);

        const buffer = (label: string, size: number) => {
            const b = device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            this._buffers.push(b);
            return b;
        };
        this.farViewProj = buffer('CascadedShadowMap/FarViewProj', 64);
        this.uniform = buffer('CascadedShadowMap/Cascades', CASCADES_BYTES);
        const lights = buffer('CascadedShadowMap/Lights', LIGHT_UNIFORM_BYTES);
        const temporal = buffer('CascadedShadowMap/Temporal', CAMERA_TEMPORAL_BYTES);
        const layout = device.createBindGroupLayout({ label: 'CascadedShadowMap/Camera BGL', entries: cameraBindGroupLayoutEntries() });
        this._viewBuffers = this._layerViews.map((_, c) => buffer(`CascadedShadowMap/View${c}`, 64));
        this._projectionBuffers = this._layerViews.map((_, c) => buffer(`CascadedShadowMap/Projection${c}`, 64));
        this._cameraBindGroups = this._layerViews.map((_, c) => device.createBindGroup({
            label: `CascadedShadowMap/Camera${c}`,
            layout,
            entries: [this._viewBuffers[c], this._projectionBuffers[c], lights, temporal]
                .map((b, binding) => ({ binding, resource: { buffer: b } })),
        }));
    }

    /** Fit the cascades to `camera` for a light travelling along `lightDir`. */
    fit(camera: Camera, lightDir: ArrayLike<number>): void {
        this.slots = fitCascades(this.options, camera, lightDir);
    }

    /** Upload this frame's cascades, the wide cascade's matrix and the cascade cameras. */
    upload(lightDir: ArrayLike<number>, lightColor: ArrayLike<number>, cameraPos: ArrayLike<number>): void {
        const o = this.options;
        const queue = this._device.queue;
        const f = new Float32Array(this._data);
        const u = new Uint32Array(this._data);
        f.fill(0);
        const viewProj = mat4.create();
        // KanseiCascade: viewProj, texelWorld, depthRange, radius, _pad (80 bytes)
        this.slots.forEach((slot, c) => {
            mat4.multiply(viewProj, slot.projection, slot.view);
            f.set(viewProj, c * 20);
            f[c * 20 + 16] = 2 * slot.radius / o.resolution;
            f[c * 20 + 17] = slot.depthRange;
            f[c * 20 + 18] = slot.radius;
        });
        const base = MAX_CASCADES * 20;
        const dir = vec3.fromValues(lightDir[0], lightDir[1], lightDir[2]);
        if (vec3.length(dir) < 1e-6) vec3.set(dir, 0, -1, 0);
        vec3.normalize(dir, dir);
        f.set(dir, base);
        u[base + 3] = this.slots.length;
        f[base + 4] = lightColor[0];
        f[base + 5] = lightColor[1];
        f[base + 6] = lightColor[2];
        f[base + 7] = Math.tan(Math.max(o.lightAngularDiameter, 0) * 0.5);
        f[base + 8] = o.normalBias;
        f[base + 9] = Math.min(Math.max(o.blend, 1e-3), 1);
        f[base + 10] = o.maxDistance;
        f[base + 12] = cameraPos[0];
        f[base + 13] = cameraPos[1];
        f[base + 14] = cameraPos[2];
        queue.writeBuffer(this.uniform, 0, this._data);
        const far = this.farViewProjection();
        if (far) queue.writeBuffer(this.farViewProj, 0, far as Float32Array);
        // each cascade has its own buffers, so these writes do not overwrite one another
        this.slots.forEach((slot, c) => {
            queue.writeBuffer(this._viewBuffers[c], 0, slot.view as Float32Array);
            queue.writeBuffer(this._projectionBuffers[c], 0, slot.projection as Float32Array);
        });
    }

    /** No shadowed directional light this frame: cascades off. */
    disable(): void {
        this.slots = [];
        new Uint8Array(this._data).fill(0);
        this._device.queue.writeBuffer(this.uniform, 0, this._data);
    }

    /** The widest cascade's view-projection this frame, or null with no cascades. */
    farViewProjection(): mat4 | null {
        const far = this.slots[this.slots.length - 1];
        return far ? mat4.multiply(this._farViewProj, far.projection, far.view) : null;
    }

    /**
     * Renders each cascade: clears it and draws the visible `castShadow` renderables among
     * `objects` from it, each with its matrices in `meshBindGroup` (group 2) at
     * `meshOffset(renderable)`. Renderables with `instanceCulling` draw the instances culled for
     * the renderer's cull view `cullView(cascade)`.
     */
    encode(
        encoder: GPUCommandEncoder,
        objects: readonly Renderable[],
        meshBindGroup: GPUBindGroup,
        meshOffset: (renderable: Renderable) => number,
        cullView: (cascade: number) => number,
    ): void {
        const device = this._device;
        for (let c = 0; c < this.slots.length; c++) {
            const pass = encoder.beginRenderPass({
                label: 'Renderer/CascadePass',
                timestampWrites: gpuPass('Renderer/CascadePass'),
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this._layerViews[c],
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            });
            pass.setBindGroup(BindGroupSlot.Camera, this._cameraBindGroups[c]);
            const view = cullView(c);

            let pipeline: GPURenderPipeline | null = null;
            let material: Material | null = null;
            let layouts: Iterable<GPUVertexBufferLayout | null> | null = null;
            let vertexBuffer: GPUBuffer | null = null;
            let indexBuffer: GPUBuffer | null = null;
            for (const obj of objects) {
                const geometry = obj.geometry;
                if (!obj.castShadow || !geometry.initialized) continue;
                if (obj.material !== material || geometry.vertexBuffersDescriptors !== layouts) {
                    if (obj.material !== material) {
                        // the renderer updated it this frame
                        pass.setBindGroup(BindGroupSlot.Material, obj.material.currentBindGroup ?? obj.material.getBindGroup(device));
                    }
                    material = obj.material;
                    layouts = geometry.vertexBuffersDescriptors;
                    const next = material.getDepthPipeline(device, layouts, CascadedShadowMap.FORMAT, CascadedShadowMap.DEPTH_BIAS);
                    if (next !== pipeline) {
                        pass.setPipeline(next);
                        pipeline = next;
                    }
                }
                const offset = meshOffset(obj);
                pass.setBindGroup(BindGroupSlot.Mesh, meshBindGroup, [offset, offset]);
                if (geometry.vertexBuffer !== vertexBuffer) {
                    pass.setVertexBuffer(0, geometry.vertexBuffer!);
                    vertexBuffer = geometry.vertexBuffer!;
                }
                if (geometry.indexBuffer !== indexBuffer) {
                    pass.setIndexBuffer(geometry.indexBuffer!, geometry.indexFormat!);
                    indexBuffer = geometry.indexBuffer!;
                }
                if (geometry.isInstancedGeometry) {
                    for (const extra of (geometry as InstancedGeometry).extraBuffers) {
                        if (!extra.initialized) extra.initialize(device);
                    }
                }
                // culled against this cascade's box, not the camera's frustum
                drawGeometry(pass, geometry, obj.instanceCulling?.view(view) ?? null);
            }
            pass.end();
        }
    }

    destroy(): void {
        this.texture.destroy();
        for (const buffer of this._buffers) buffer.destroy();
    }
}

export { CascadedShadowMap };
