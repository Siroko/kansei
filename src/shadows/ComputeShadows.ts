import { assemble } from '../materials/shaders/ShaderUtils';
import { DirectionalLight } from '../lights/DirectionalLight';
import { PointLight } from '../lights/PointLight';
import { AreaLight } from '../lights/AreaLight';
import { ShadowMap } from './ShadowMap';
import { CubeMapShadowMap } from './CubeMapShadowMap';
import spotLightTypes from '../../rust/kansei-core/src/shaders/spot_light_types.wgsl?raw';
import computeShadows from '../../rust/kansei-core/src/shaders/compute_shadows.wgsl?raw';

/**
 * The renderer's lights and shadow maps as compute shaders see them (group 0 bindings 1 and 3-9),
 * with the spot-light types it needs: `DirLightData`, `PointLightData`, `dirShadowLookup`,
 * `dirShadowCovers`, `pointShadowLookup` and the spot lights' buffer and atlas. Rust:
 * `shadows/compute_shadows.wgsl` after `spot_light_types.wgsl`.
 */
export const COMPUTE_SHADOWS_WGSL: string = assemble([spotLightTypes, computeShadows]);

/** `PointLightData.shadowLayer` of a light without a shadow (WGSL `NO_SHADOW`). */
export const NO_SHADOW = 0xffffffff;
/**
 * `PointLightData.shadowLayer` of a positional light shadowed by the directional map binding: an
 * area light that owns the 2D shadow map (a perspective map, TS only). `compute_shadows.wgsl` has
 * no such case; a pass that honours it says so (the volumetric fog's injection).
 */
export const SHADOW_MAP = 0xfffffffe;

/** Bytes of a `DirLightData` and a `PointLightData`. */
const LIGHT_BYTES = 32;
/** `KanseiSpotLights` with one zeroed light: the count (0) padded to 16 bytes, then 144. */
const NO_SPOT_LIGHTS_BYTES = 16 + 144;

/**
 * A cascaded shadow map's widest cascade, as compute passes read it: its depth and its
 * view-projection's uniform buffer, rewritten every frame (Rust `CascadedShadowMap::far_view`,
 * `far_view_proj`).
 */
export interface CascadedShadowSource {
    readonly farView: GPUTextureView;
    readonly farViewProj: GPUBuffer;
}

interface Gpu {
    dirLights: GPUBuffer;
    pointLights: GPUBuffer;
    dummyDepth: GPUTexture;
    dummyAtlas: GPUTexture;
    dummyVP: GPUBuffer;
    dummySpotLights: GPUBuffer;
    dummySpotAtlas: GPUTexture;
    dummyDepthView: GPUTextureView;
    dummyAtlasView: GPUTextureView;
    dummySpotAtlasView: GPUTextureView;
    spotSampler: GPUSampler;
}

/**
 * The renderer's lights and shadow maps as compute passes see them: the compute-visible copy of
 * group 3 (`COMPUTE_SHADOWS_WGSL`, group 0 bindings 1 and 3-9), shared by the volumetric fog and,
 * as in the Rust engine, voxel GI's light injection.
 *
 * - Directional lights: the light the directional map was rendered from casts its shadow through
 *   it (`setShadowMap`), or through the widest cascade of a cascaded map (`setCascadedShadowMap`).
 * - Point lights: each light the cube shadow map drew (`setPointShadows`) through its own faces
 *   (TS cube maps hold several lights; Rust's only the first). Area lights are point lights at
 *   their position, shadowed by the 2D map when they own it (`SHADOW_MAP`).
 * - Spot lights: a buffer and shadow atlas given as they are (`setSpotLights`).
 *
 * A pass lays its group out with `layoutEntries()` next to its own bindings, calls `prepare`
 * before recording (and rebuilds its bind group when that returns true), and binds `entries()`.
 * Rust: `shadows::compute_shadows::ComputeShadows`.
 */
class ComputeShadows {
    private _dir = new Float32Array(0);
    private _point = new Float32Array(0);
    private _shadowMap: ShadowMap | null = null;
    private _shadowSource: { view: GPUTextureView; viewProj: GPUBuffer } | null = null;
    private _pointShadows: CubeMapShadowMap | null = null;
    private _pointShadowView: GPUTextureView | null = null;
    private _spotLights: GPUBuffer | null = null;
    private _spotShadows: GPUTextureView | null = null;
    private _lightsDirty = true;
    private _bindingsDirty = true;
    private _gpu: Gpu | null = null;

    /** Number of directional lights uploaded (`FogParams.numDirLights`). */
    get dirCount(): number { return this._dir.length / 8; }
    /** Number of point and area lights uploaded (`FogParams.numPointLights`). */
    get pointCount(): number { return this._point.length / 8; }
    get hasShadowMap(): boolean { return this._shadowSource !== null; }
    get hasPointShadows(): boolean { return this._pointShadows !== null; }
    get hasSpotLights(): boolean { return this._spotLights !== null; }

    /**
     * Collect the directional, point and area lights (spot lights come as a buffer,
     * `setSpotLights`). The directional light the shadow map was rendered from (`ShadowMap.light`,
     * or the light whose own `shadowMap` it is) is shadowed by it; when that is unknown (a cascaded
     * map, or a map rendered from a bare direction), the first directional light with
     * `castShadow`, else the first. A point light reads its own cube faces if the cube map drew it
     * last frame; an area light that owns the 2D map reads it (`SHADOW_MAP`). `volumetricOnly`
     * keeps the fog's rule: directional lights that aren't `volumetric` are left out, and such
     * point and area lights scatter nothing. The lights are uploaded again only if they changed.
     */
    updateLights(
        dirLights: readonly DirectionalLight[],
        pointLights: readonly PointLight[],
        areaLights: readonly AreaLight[] = [],
        volumetricOnly = true,
    ): void {
        const sm = this._shadowSource ? this._shadowMap : null;
        const owns = (l: DirectionalLight | AreaLight) => sm !== null && (sm.light === l || l.shadowMap === sm);
        const ownerKnown = sm !== null && (dirLights.some(owns) || areaLights.some(owns));
        const fallback = dirLights.find((l) => l.castShadow) ?? dirLights[0];
        const dirShadowed = (l: DirectionalLight) => this._shadowSource !== null && (ownerKnown ? owns(l) : l === fallback);

        const dirs = dirLights.filter((l) => l.volumetric || !volumetricOnly);
        const dir = new Float32Array(dirs.length * 8);
        const dirU32 = new Uint32Array(dir.buffer);
        dirs.forEach((light, i) => {
            const c = light.effectiveColor;
            dir.set(light.direction, i * 8);
            dirU32[i * 8 + 3] = dirShadowed(light) ? 1 : 0;
            dir.set(c, i * 8 + 4);
        });

        const cubeLights = this._pointShadows?.lights ?? [];
        const positional: (PointLight | AreaLight)[] = [...pointLights, ...areaLights];
        const point = new Float32Array(positional.length * 8);
        const pointU32 = new Uint32Array(point.buffer);
        positional.forEach((light, i) => {
            light.updateModelMatrix();
            const wm = light.worldMatrix.internalMat4;
            point.set([wm[12], wm[13], wm[14], light.radius], i * 8);
            if (light.volumetric || !volumetricOnly) point.set(light.effectiveColor, i * 8 + 4);
            const cubeIndex = cubeLights.indexOf(light);
            pointU32[i * 8 + 7] = cubeIndex >= 0 ? cubeIndex * 6
                : (light instanceof AreaLight && owns(light)) ? SHADOW_MAP
                : NO_SHADOW;
        });

        const same = (a: Float32Array, b: Float32Array) => {
            const bits = new Uint32Array(b.buffer, b.byteOffset, b.length);
            return a.length === b.length && new Uint32Array(a.buffer, a.byteOffset, a.length).every((v, i) => v === bits[i]);
        };
        this._lightsDirty ||= !same(dir, this._dir) || !same(point, this._point);
        this._dir = dir;
        this._point = point;
    }

    /** The directional shadow from a `ShadowMap` (its view-projection read from its own buffer). */
    setShadowMap(shadowMap: ShadowMap | null): void {
        this._shadowMap = shadowMap;
        this._shadowSource = shadowMap
            ? { view: shadowMap.depthTexture.createView(), viewProj: shadowMap.lightViewProjBuffer }
            : null;
        this._bindingsDirty = true;
    }

    /** The directional shadow from a cascaded shadow map's widest cascade. */
    setCascadedShadowMap(csm: CascadedShadowSource | null): void {
        this._shadowMap = null;
        this._shadowSource = csm ? { view: csm.farView, viewProj: csm.farViewProj } : null;
        this._bindingsDirty = true;
    }

    /** The point lights' cube shadows (`Renderer.cubeMapShadowMap`). */
    setPointShadows(cube: CubeMapShadowMap | null): void {
        this._pointShadows = cube;
        this._pointShadowView = cube ? cube.distanceTexture.createView({ dimension: '2d-array' }) : null;
        this._bindingsDirty = true;
    }

    /**
     * Spot lights as the renderer uploads them (a `KanseiSpotLights` storage buffer) and their
     * depth atlas (a `2d-array` view), or null for none.
     */
    setSpotLights(lights: GPUBuffer | null, shadowAtlas: GPUTextureView | null): void {
        this._spotLights = lights;
        this._spotShadows = shadowAtlas;
        this._bindingsDirty = true;
    }

    /** The layout entries of bindings 1 and 3-9, visible to compute. */
    static layoutEntries(): GPUBindGroupLayoutEntry[] {
        const visibility = GPUShaderStage.COMPUTE;
        const storage: GPUBufferBindingLayout = { type: 'read-only-storage' };
        return [
            { binding: 1, visibility, texture: { sampleType: 'depth' } },
            { binding: 3, visibility, buffer: storage },
            { binding: 4, visibility, buffer: storage },
            { binding: 5, visibility, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
            { binding: 6, visibility, buffer: { type: 'uniform' } },
            { binding: 7, visibility, buffer: storage },
            { binding: 8, visibility, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
            { binding: 9, visibility, sampler: { type: 'comparison' } },
        ];
    }

    /**
     * Make the stand-ins and upload the lights (growing their buffers). True when the bind groups
     * made from `entries` must be rebuilt.
     */
    prepare(device: GPUDevice): boolean {
        if (!this._gpu) {
            this._gpu = ComputeShadows._initGpu(device);
            this._lightsDirty = true;
            this._bindingsDirty = true;
        }
        if (this._lightsDirty) {
            const gpu = this._gpu;
            const fit = (buffer: GPUBuffer, data: Float32Array, label: string): GPUBuffer => {
                const needed = Math.max(data.byteLength, LIGHT_BYTES);
                if (needed <= buffer.size) return buffer;
                buffer.destroy();
                this._bindingsDirty = true;
                return device.createBuffer({
                    label,
                    size: 2 ** Math.ceil(Math.log2(needed)),
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
            };
            gpu.dirLights = fit(gpu.dirLights, this._dir, 'ComputeShadows/DirLights');
            gpu.pointLights = fit(gpu.pointLights, this._point, 'ComputeShadows/PointLights');
            if (this._dir.length > 0) device.queue.writeBuffer(gpu.dirLights, 0, this._dir);
            if (this._point.length > 0) device.queue.writeBuffer(gpu.pointLights, 0, this._point);
            this._lightsDirty = false;
        }
        const dirty = this._bindingsDirty;
        this._bindingsDirty = false;
        return dirty;
    }

    /** Bindings 1 and 3-9 (after `prepare`). */
    entries(): GPUBindGroupEntry[] {
        const gpu = this._gpuOrThrow();
        return [
            { binding: 1, resource: this._shadowSource?.view ?? gpu.dummyDepthView },
            { binding: 3, resource: { buffer: gpu.dirLights } },
            { binding: 4, resource: { buffer: gpu.pointLights } },
            { binding: 5, resource: this._pointShadowView ?? gpu.dummyAtlasView },
            { binding: 6, resource: { buffer: this._shadowSource?.viewProj ?? gpu.dummyVP } },
            ...this.spotEntries(),
        ];
    }

    /** Bindings 7-9 alone: the spot lights, their atlas and its sampler. */
    spotEntries(): GPUBindGroupEntry[] {
        const gpu = this._gpuOrThrow();
        return [
            { binding: 7, resource: { buffer: this._spotLights ?? gpu.dummySpotLights } },
            { binding: 8, resource: this._spotShadows ?? gpu.dummySpotAtlasView },
            { binding: 9, resource: gpu.spotSampler },
        ];
    }

    destroy(): void {
        const gpu = this._gpu;
        if (!gpu) return;
        for (const b of [gpu.dirLights, gpu.pointLights, gpu.dummyVP, gpu.dummySpotLights]) b.destroy();
        for (const t of [gpu.dummyDepth, gpu.dummyAtlas, gpu.dummySpotAtlas]) t.destroy();
        this._gpu = null;
    }

    private _gpuOrThrow(): Gpu {
        if (!this._gpu) throw new Error('ComputeShadows.prepare first');
        return this._gpu;
    }

    private static _initGpu(device: GPUDevice): Gpu {
        const buffer = (label: string, size: number, usage: number) =>
            device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
        // Fallbacks bound when no shadow map is set: a 1x1 depth texture (never sampled, the
        // lookups are gated by flags), a 1x1x6 distance atlas, and an identity light VP.
        const dummyDepth = device.createTexture({
            label: 'ComputeShadows/DummyShadowDepth',
            size: [1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        const dummyAtlas = device.createTexture({
            label: 'ComputeShadows/DummyPointShadow',
            size: [1, 1, 6],
            format: 'r32float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        const dummyVP = buffer('ComputeShadows/DummyLightVP', 64, GPUBufferUsage.UNIFORM);
        device.queue.writeBuffer(dummyVP, 0, new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
        // no spot lights: a buffer whose count is 0 (zero-initialised), and a 1x1 atlas
        const dummySpotLights = buffer('ComputeShadows/DummySpotLights', NO_SPOT_LIGHTS_BYTES, GPUBufferUsage.STORAGE);
        const dummySpotAtlas = device.createTexture({
            label: 'ComputeShadows/DummySpotShadow',
            size: [1, 1, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        return {
            dirLights: buffer('ComputeShadows/DirLights', LIGHT_BYTES, GPUBufferUsage.STORAGE),
            pointLights: buffer('ComputeShadows/PointLights', LIGHT_BYTES, GPUBufferUsage.STORAGE),
            dummyDepth,
            dummyAtlas,
            dummyVP,
            dummySpotLights,
            dummySpotAtlas,
            dummyDepthView: dummyDepth.createView(),
            dummyAtlasView: dummyAtlas.createView({ dimension: '2d-array' }),
            dummySpotAtlasView: dummySpotAtlas.createView({ dimension: '2d-array' }),
            spotSampler: device.createSampler({
                label: 'ComputeShadows/SpotShadowSampler',
                compare: 'less-equal',
                magFilter: 'linear',
                minFilter: 'linear',
            }),
        };
    }
}

export { ComputeShadows };
