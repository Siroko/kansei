import { Camera } from '../../cameras/Camera';
import { DirectionalLight } from '../../lights/DirectionalLight';
import { PointLight } from '../../lights/PointLight';
import { AreaLight } from '../../lights/AreaLight';
import { FroxelGrid } from '../../froxels/FroxelGrid';
import { ShadowMap } from '../../shadows/ShadowMap';
import { CubeMapShadowMap } from '../../shadows/CubeMapShadowMap';
import type { SkyOcclusion } from '../../shadows/SkyOcclusion';
import { COMPUTE_SHADOWS_WGSL, ComputeShadows, SHADOW_MAP } from '../../shadows/ComputeShadows';
import type { CascadedShadowSource } from '../../shadows/ComputeShadows';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { assemble } from '../../materials/shaders/ShaderUtils';
import { mat4 } from 'gl-matrix';
import { gpuPass } from '../../profiling/Profiler';
import froxelCommon from '../../../rust/kansei-core/src/shaders/froxel_common.wgsl?raw';
import fogInject from '../../../rust/kansei-core/src/shaders/volumetric_fog_inject.wgsl?raw';
import skyLighting from '../../../rust/kansei-core/src/atmosphere/shaders/sky_lighting.wgsl?raw';
import skyOcclusion from '../../../rust/kansei-core/src/shaders/sky_occlusion.wgsl?raw';
import clipmapProbes from '../../../rust/kansei-core/src/gi/shaders/clipmap_probes.wgsl?raw';
import fogMedia from '../../../rust/kansei-core/src/shaders/volumetric_fog_media.wgsl?raw';
import fogSpot from '../../../rust/kansei-core/src/shaders/volumetric_fog_spot.wgsl?raw';
import fogComposite from '../../../rust/kansei-core/src/shaders/volumetric_fog_composite.wgsl?raw';

export interface VolumetricFogOptions {
    froxelGrid: FroxelGrid;
    /** The directional shadow map (`Renderer.shadowMap`) the fog's shafts come through. */
    shadowMap?: ShadowMap;
    /** A cascaded shadow map whose widest cascade shadows the fog, instead of `shadowMap`. */
    cascadedShadowMap?: CascadedShadowSource;
    /** The point lights' cube shadows (`Renderer.cubeMapShadowMap`). */
    cubeMapShadowMap?: CubeMapShadowMap;
    /** Scattering density at `fogHeight` (per metre). */
    baseDensity?: number;
    /** Exponential falloff of density with height above `fogHeight` (per metre). */
    heightFalloff?: number;
    /** Height below which density stays at `baseDensity`. */
    fogHeight?: number;
    /** Extinction = density * extinctionCoeff (1 = no absorption). */
    extinctionCoeff?: number;
    /** Henyey-Greenstein g: 0 isotropic, > 0 forward scattering. */
    anisotropy?: number;
    /** View distance before which there is no fog (UE's fog start distance). */
    startDistance?: number;
    /** View depth past which the froxels hold no fog, at most the grid's `far`; 0: the grid's `far`. */
    maxDistance?: number;
    /** Density field drift, in metres per second of `time`. */
    windDirection?: [number, number, number];
    /**
     * Radiance of a uniform sky around the fog (scatters as density * ambient): set it so fog stays
     * lit with no direct light (dusk, overcast). Far fog tends to it. Zero by default.
     */
    ambient?: [number, number, number];
    /**
     * Scattering albedo: scattering = density * albedo (extinction = density * extinctionCoeff).
     * Unreal's volumetric fog maps as albedo = its albedo * its extinction scale,
     * extinctionCoeff = its extinction scale. Default 1.
     */
    albedo?: [number, number, number];
    /** Scales the sky's light on the fog once a sky is bound (`setSkyLighting`); 1 (the default) is physical. */
    skyAmbientScale?: number;
}

/** The shape of a `LocalFogVolume`. */
export type LocalFogShape = 'ellipsoid' | 'box';

/**
 * A local fog volume: an ellipsoid or box of mist (over a lake, in a hollow) injected into the
 * fog's froxels, after Unreal's `LocalFogVolume` (Rust `LocalFogVolume`). In the volume's unit
 * shape `q`, with `r` = |q| (the largest |q_i| for a box) below 1, the extinction is
 * `radialExtinction * (1 - r^2) + heightExtinction * exp(-heightFalloff * max(q.y - heightOffset, 0))`,
 * faded to zero over the outer `edgeFade` of the radius. It scatters the fog's lights and sky
 * with its own albedo; wind and start distance leave it alone.
 *
 * Unreal's `RadialFogExtinction`, `HeightFogExtinction`, `HeightFogFalloff`, `HeightFogOffset`
 * and `FogAlbedo` carry over; the shapes of the two terms are kansei's own, so the look may need
 * a trim.
 */
export class LocalFogVolume {
    shape: LocalFogShape = 'ellipsoid';
    center: [number, number, number];
    /** Semi-axes of the ellipsoid, or half extents of the box, metres. */
    radii: [number, number, number];
    /** Rotation about +Y, radians (as `Object3D.rotation.y`). */
    yaw = 0;
    /** Extinction per metre at the centre, falling to zero at the surface. */
    radialExtinction = 1;
    /** Extinction per metre at and below `heightOffset`, falling off above it. */
    heightExtinction = 0;
    /** Exponential falloff per unit of the volume's half height. */
    heightFalloff = 1000;
    /** Height in the unit sphere (-1 bottom, 1 top) below which the height term is at full strength. */
    heightOffset = 0;
    albedo: [number, number, number] = [1, 1, 1];
    /** Fraction of the radius over which the fog fades out at the surface (0: a hard edge). */
    edgeFade = 0.25;

    /**
     * An axis-aligned ellipsoid of `radius` across and `halfHeight` up and down, with Unreal's
     * defaults otherwise: radial extinction 1, no height term, a soft edge.
     */
    constructor(center: [number, number, number], radius: number, halfHeight: number) {
        this.center = [...center];
        this.radii = [radius, halfHeight, radius];
    }

    /** A box of `halfExtents`, with radial extinction 1, no height term and a soft edge. */
    static box(center: [number, number, number], halfExtents: [number, number, number]): LocalFogVolume {
        const v = new LocalFogVolume(center, 1, 1);
        v.shape = 'box';
        v.radii = [...halfExtents];
        return v;
    }

    /** Extinction per metre at a world-space point, as the fog shader computes it. */
    extinctionAt(p: [number, number, number]): number {
        const inv = this.radii.map((r) => 1 / Math.max(r, 1e-3));
        const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
        const d = [p[0] - this.center[0], p[1] - this.center[1], p[2] - this.center[2]];
        const q = [(c * d[0] - s * d[2]) * inv[0], d[1] * inv[1], (s * d[0] + c * d[2]) * inv[2]];
        const r = this.shape === 'box' ? Math.max(Math.abs(q[0]), Math.abs(q[1]), Math.abs(q[2])) : Math.hypot(q[0], q[1], q[2]);
        if (r >= 1) return 0;
        const fade = Math.min(Math.max(this.edgeFade, 0), 1);
        let edge = 1;
        if (fade > 0) {
            const t = Math.min(Math.max((1 - r) / fade, 0), 1);
            edge = t * t * (3 - 2 * t);
        }
        const radial = Math.max(this.radialExtinction, 0) * (1 - r * r);
        const height = Math.max(this.heightExtinction, 0) * Math.exp(-this.heightFalloff * Math.max(q[1] - this.heightOffset, 0));
        return (radial + height) * edge;
    }

    /** The WGSL `LocalFogVolume` (volumetric_fog_media.wgsl) at float `offset` of `f32`/`u32`. */
    writeGpu(f32: Float32Array, u32: Uint32Array, offset: number): void {
        const inv = (r: number) => 1 / Math.max(r, 1e-3);
        f32.set(this.center, offset);
        f32[offset + 3] = Math.max(this.radialExtinction, 0);
        f32.set(this.radii.map(inv), offset + 4);
        f32[offset + 7] = Math.max(this.heightExtinction, 0);
        f32.set(this.albedo, offset + 8);
        f32[offset + 11] = this.heightFalloff;
        f32[offset + 12] = Math.cos(this.yaw);
        f32[offset + 13] = Math.sin(this.yaw);
        f32[offset + 14] = this.heightOffset;
        f32[offset + 15] = Math.min(Math.max(this.edgeFade, 0), 1);
        u32.set([this.shape === 'box' ? 1 : 0, 0, 0, 0], offset + 16);
    }
}

/** Bytes of the WGSL `FogParams`, `CompositeParams`, `FogMediaParams`. */
const FOG_PARAMS_BYTES = 208;
const COMPOSITE_PARAMS_BYTES = 32;
const MEDIA_PARAMS_BYTES = 32;
/** Bytes of a `LocalFogVolume`, the `SkyLighting` uniform and a `ClipProbeGrid`. */
const LOCAL_FOG_VOLUME_BYTES = 80;
const SKY_LIGHTING_BYTES = 240;
const CLIP_PROBE_GRID_BYTES = 128;

/** `text` with `from` replaced once; throws when the shared WGSL no longer has `from`. */
function patch(text: string, from: string, to: string): string {
    if (!text.includes(from)) throw new Error(`VolumetricFogEffect: the injection WGSL changed, no "${from}"`);
    return text.replace(from, to);
}

/**
 * The Rust fog's injection (`volumetric_fog.rs`'s `INJECT_WGSL`), plus the TS area light that
 * owns the 2D shadow map: its `PointLightData.shadowLayer` is `SHADOW_MAP`, read through the
 * directional map's binding.
 */
const INJECT_WGSL = patch(
    assemble([
        froxelCommon, fogInject, skyLighting, skyOcclusion, clipmapProbes, fogMedia,
        COMPUTE_SHADOWS_WGSL, fogSpot,
        `const SHADOW_MAP : u32 = ${SHADOW_MAP}u;`,
    ]),
    'if (pl.shadowLayer != NO_SHADOW && params.hasPointShadows != 0u) {',
    'if (pl.shadowLayer == SHADOW_MAP) {\n'
    + '            if (params.hasShadowMap != 0u) { visibility = dirShadowLookup(worldPos); }\n'
    + '        } else if (pl.shadowLayer != NO_SHADOW && params.hasPointShadows != 0u) {',
);
const COMPOSITE_WGSL = assemble([froxelCommon, fogComposite]);

/**
 * Froxel volumetric fog, a port of the Rust `VolumetricFogEffect` on its WGSL: per froxel, a
 * height fog (flat below `fogHeight`, from `startDistance` to the `reach()`) plus any
 * `localVolumes`, lit by the directional, point and area lights with their shadows
 * (`ComputeShadows`), by a uniform `ambient` sky and by a `SkyAtmosphere`'s sky lighting
 * (`setSkyLighting`); jittered per frame on a temporal grid; integrated front to back and
 * composited over the scene.
 *
 * Its spot lights and clipmap probes are bound to stand-ins (none) until those land in the TS
 * engine. The sky occlusion (`setSkyOcclusion`) dims the sky lighting only, so it shows once the
 * fog has a sky (`setSkyLighting`).
 */
class VolumetricFogEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _froxelGrid: FroxelGrid;
    private _shadows = new ComputeShadows();

    baseDensity: number;
    heightFalloff: number;
    fogHeight: number;
    extinctionCoeff: number;
    anisotropy: number;
    startDistance: number;
    /**
     * View depth past which the froxels hold no fog (Unreal's `VolumetricFogDistance`): at most
     * the grid's `far`, 0 for the grid's `far`. Change it any frame; the grid stays.
     */
    maxDistance: number;
    windDir: [number, number, number];
    ambient: [number, number, number];
    /** Scattering albedo of the height fog: scattering = density * albedo. */
    albedo: [number, number, number];
    /** Scales the sky's light on the fog once a sky is bound (`setSkyLighting`); 1 is physical. */
    skyAmbientScale: number;
    /**
     * Local fog volumes injected with the height fog, uploaded every frame (keep it to tens).
     * Each scatters the fog's lights and sky with its own albedo.
     */
    localVolumes: LocalFogVolume[] = [];
    /**
     * Seconds, drives the wind offset. Null: the effect's own clock, from its construction. Rust
     * has no clock and sets `time` per frame.
     */
    time: number | null = null;

    /** Temporal jitter index of the injection (1..=1024), advanced every frame. */
    private _frame = 1;
    private _shadowMap: ShadowMap | null = null;
    private _cubeMapShadowMap: CubeMapShadowMap | null = null;
    /** The sky occlusion's volume and parameters (`setSkyOcclusion`). */
    private _skyOcclusion: { volume: GPUTextureView; params: GPUBuffer } | null = null;
    private _skyLighting: GPUBuffer | null = null;
    /** The injection bind group is stale (a sky or sky occlusion bound, the volume buffer grown). */
    private _injectBGDirty = false;

    // Injection pass
    private _injectPipeline: GPUComputePipeline | null = null;
    private _injectBGL: GPUBindGroupLayout | null = null;
    private _injectBG: GPUBindGroup | null = null;
    private _injectTarget: GPUTexture | null = null;
    private _fogParamsBuffer: GPUBuffer | null = null;
    private _fogParams = new ArrayBuffer(FOG_PARAMS_BYTES);

    // The media (volumetric_fog_media.wgsl): its parameters, the local volumes, and stand-ins for
    // the sky lighting and the sky occlusion until they are bound, and the clipmap probes
    private _mediaParamsBuffer: GPUBuffer | null = null;
    private _mediaParams = new ArrayBuffer(MEDIA_PARAMS_BYTES);
    private _volumesBuffer: GPUBuffer | null = null;
    private _volumesData = new ArrayBuffer(LOCAL_FOG_VOLUME_BYTES);
    private _noSkyLighting: GPUBuffer | null = null;
    private _noOcclusionVolume: GPUTexture | null = null;
    private _noOcclusionParams: GPUBuffer | null = null;
    private _noClipProbeGrid: GPUBuffer | null = null;
    private _noClipProbes: GPUBuffer | null = null;

    // Composite pass
    private _compositePipeline: GPUComputePipeline | null = null;
    private _compositeBG: GPUBindGroup | null = null;
    private _compositeParamsBuffer: GPUBuffer | null = null;
    private _accumSampler: GPUSampler | null = null;
    private _noShafts: GPUTexture | null = null;
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;
    private _currentAccum: GPUTexture | null = null;

    private _startTime = performance.now();
    private _vp = mat4.create();
    private _invVP = mat4.create();

    constructor(options: VolumetricFogOptions) {
        super();
        this._froxelGrid     = options.froxelGrid;
        this.baseDensity     = options.baseDensity ?? 0.02;
        this.heightFalloff   = options.heightFalloff ?? 0.1;
        this.fogHeight       = options.fogHeight ?? 0;
        this.extinctionCoeff = options.extinctionCoeff ?? 1.0;
        this.anisotropy      = options.anisotropy ?? 0.6;
        this.startDistance   = options.startDistance ?? 0;
        this.maxDistance     = options.maxDistance ?? 0;
        this.windDir         = options.windDirection ?? [0, 0, 0];
        this.ambient         = options.ambient ?? [0, 0, 0];
        this.albedo          = options.albedo ?? [1, 1, 1];
        this.skyAmbientScale = options.skyAmbientScale ?? 1;
        if (options.cascadedShadowMap) this.setCascadedShadowMap(options.cascadedShadowMap);
        else this.setShadowMap(options.shadowMap ?? null);
        this.setPointShadows(options.cubeMapShadowMap ?? null);
    }

    /** The froxel grid the fog is injected into. */
    get froxelGrid(): FroxelGrid { return this._froxelGrid; }

    /**
     * The view depth the froxels hold fog to: `maxDistance`, or the grid's `far` when it is 0 or
     * beyond it.
     */
    reach(): number {
        const far = this._froxelGrid.far;
        return this.maxDistance > 0 ? Math.min(this.maxDistance, far) : far;
    }

    /** Drop the temporal history; call on camera cuts. No-op without a temporal grid. */
    resetHistory(): void {
        this._froxelGrid.resetHistory();
    }

    /** The directional shadow map the shafts come through (`setShadowMap`). */
    get shadowMap(): ShadowMap | null { return this._shadowMap; }
    set shadowMap(sm: ShadowMap | null) { this.setShadowMap(sm); }

    /**
     * Use a directional shadow map (`Renderer.shadowMap`) for light shafts. The light's
     * view-projection is read from the map's own uniform buffer, so it is always the one the map
     * was last rendered with.
     */
    setShadowMap(shadowMap: ShadowMap | null): void {
        this._shadowMap = shadowMap;
        this._shadows.setShadowMap(shadowMap);
    }

    /** Shafts from a cascaded shadow map's widest cascade. Use instead of `setShadowMap`. */
    setCascadedShadowMap(csm: CascadedShadowSource | null): void {
        this._shadowMap = null;
        this._shadows.setCascadedShadowMap(csm);
    }

    /** The point lights' cube shadows (`Renderer.cubeMapShadowMap`). */
    get cubeMapShadowMap(): CubeMapShadowMap | null { return this._cubeMapShadowMap; }
    setPointShadows(cube: CubeMapShadowMap | null): void {
        this._cubeMapShadowMap = cube;
        this._shadows.setPointShadows(cube);
    }

    /**
     * Dim the sky's light on the fog by how much of the sky each froxel sees
     * (`Renderer.skyOcclusion`, `SKY_OCCLUSION_WGSL`'s `skyVisibility`), as Unreal's Lumen occludes
     * the sky light its volumetric fog receives: under the canopy, and where the trees round a
     * clearing hide the horizon. The fog's other lights are not affected. Rust:
     * `set_sky_occlusion`.
     */
    setSkyOcclusion(skyOcclusion: SkyOcclusion | null): void {
        this._skyOcclusion = skyOcclusion ? { volume: skyOcclusion.volume, params: skyOcclusion.params } : null;
        this._injectBGDirty = true;
    }

    /**
     * Light the fog with a sky: `SkyAtmosphere.bindings.skyLighting`. The sky's radiance,
     * convolved with the fog's phase function, scatters in every froxel (times
     * `skyAmbientScale`), on top of `ambient`, so the fog takes the sky's colour and stays lit at
     * dusk. Put the `AtmosphereEffect` before the fog in the chain, so the fog lies in front of
     * the sky and its aerial perspective. Null unbinds it.
     */
    setSkyLighting(skyLighting: GPUBuffer | null): void {
        this._skyLighting = skyLighting;
        this._injectBGDirty = true;
    }

    /**
     * Collect the volumetric lights; call each frame before render(). The directional light the
     * shadow map was rendered from casts shafts through it; a point light the cube map drew reads
     * its faces; an area light scatters as a point light at its position, shadowed by the 2D map
     * when it owns it (`ComputeShadows.updateLights`).
     */
    updateLights(dirLights: readonly DirectionalLight[], pointLights: readonly PointLight[], areaLights: readonly AreaLight[] = []): void {
        this._shadows.updateLights(dirLights, pointLights, areaLights, true);
    }

    // ── PostProcessingEffect interface ────────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const buffer = (label: string, size: number, usage: number) =>
            device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
        const compute = GPUShaderStage.COMPUTE;

        this._fogParamsBuffer = buffer('VolumetricFog/Params', FOG_PARAMS_BYTES, GPUBufferUsage.UNIFORM);
        this._compositeParamsBuffer = buffer('VolumetricFog/CompositeParams', COMPOSITE_PARAMS_BYTES, GPUBufferUsage.UNIFORM);
        this._accumSampler = device.createSampler({
            label: 'VolumetricFog/AccumSampler',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // The media: written every frame (`_uploadMedia`); the sky lighting's stand-in until one is bound
        this._mediaParamsBuffer = buffer('VolumetricFog/MediaParams', MEDIA_PARAMS_BYTES, GPUBufferUsage.UNIFORM);
        this._volumesBuffer = buffer('VolumetricFog/LocalVolumes', LOCAL_FOG_VOLUME_BYTES, GPUBufferUsage.STORAGE);
        this._noSkyLighting = buffer('VolumetricFog/NoSkyLighting', SKY_LIGHTING_BYTES, GPUBufferUsage.UNIFORM);
        // no sky occlusion: its parameters zero (off, so skyVisibility is 1) and a 1-texel volume
        this._noOcclusionParams = buffer('VolumetricFog/NoSkyOcclusion', 32, GPUBufferUsage.UNIFORM);
        this._noOcclusionVolume = device.createTexture({
            label: 'VolumetricFog/NoSkyOcclusionVolume',
            size: [1, 1, 1],
            dimension: '3d',
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        this._noClipProbeGrid = buffer('VolumetricFog/NoClipProbeGrid', CLIP_PROBE_GRID_BYTES, GPUBufferUsage.UNIFORM);
        this._noClipProbes = buffer('VolumetricFog/NoClipProbes', 64, GPUBufferUsage.STORAGE);
        // no raymarched spot shafts
        this._noShafts = device.createTexture({
            label: 'VolumetricFog/NoShafts',
            size: [1, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });

        // ── Injection pipeline: its own bindings 0, 2, 10-12, 18-22, the lights at 1, 3-9 ──
        this._injectBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/InjectBGL',
            entries: [
                { binding: 0, visibility: compute, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 2, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 10, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 11, visibility: compute, buffer: { type: 'read-only-storage' } },
                { binding: 12, visibility: compute, buffer: { type: 'uniform' } },
                // the sky occlusion (volumetric_fog_media.wgsl)
                { binding: 18, visibility: compute, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 19, visibility: compute, sampler: { type: 'filtering' } },
                { binding: 20, visibility: compute, buffer: { type: 'uniform' } },
                // a clipmap's probes (volumetric_fog_media.wgsl)
                { binding: 21, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 22, visibility: compute, buffer: { type: 'read-only-storage' } },
                ...ComputeShadows.layoutEntries(),
            ],
        });
        this._injectPipeline = device.createComputePipeline({
            label: 'VolumetricFog/Inject',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._injectBGL] }),
            compute: { module: device.createShaderModule({ label: 'VolumetricFog/Inject', code: INJECT_WGSL }), entryPoint: 'main' },
        });

        // ── Composite pipeline ──
        const compositeBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/CompositeBGL',
            entries: [
                { binding: 0, visibility: compute, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: compute, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: compute, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: compute, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 4, visibility: compute, sampler: { type: 'filtering' } },
                { binding: 5, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 6, visibility: compute, texture: { sampleType: 'unfilterable-float' } },
            ],
        });
        this._compositePipeline = device.createComputePipeline({
            label: 'VolumetricFog/Composite',
            layout: device.createPipelineLayout({ bindGroupLayouts: [compositeBGL] }),
            compute: { module: device.createShaderModule({ label: 'VolumetricFog/Composite', code: COMPOSITE_WGSL }), entryPoint: 'main' },
        });

        this._buildCompositeBG(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _rebuildInjectBG(): void {
        const device = this._device!;
        const target = this._froxelGrid.scatterExtinctionTex;
        this._injectBG = device.createBindGroup({
            label: 'VolumetricFog/InjectBG',
            layout: this._injectBGL!,
            entries: [
                { binding: 0, resource: target.createView() },
                { binding: 2, resource: { buffer: this._fogParamsBuffer! } },
                { binding: 10, resource: { buffer: this._mediaParamsBuffer! } },
                { binding: 11, resource: { buffer: this._volumesBuffer! } },
                { binding: 12, resource: { buffer: this._skyLighting ?? this._noSkyLighting! } },
                { binding: 18, resource: this._skyOcclusion?.volume ?? this._noOcclusionVolume!.createView() },
                { binding: 19, resource: this._accumSampler! },
                { binding: 20, resource: { buffer: this._skyOcclusion?.params ?? this._noOcclusionParams! } },
                { binding: 21, resource: { buffer: this._noClipProbeGrid! } },
                { binding: 22, resource: { buffer: this._noClipProbes! } },
                ...this._shadows.entries(),
            ],
        });
        this._injectTarget = target;
        this._injectBGDirty = false;
    }

    /** Upload the media parameters and the local volumes, growing the volume buffer if needed. */
    private _uploadMedia(): void {
        const device = this._device!;
        const count = this.localVolumes.length;
        const needed = count * LOCAL_FOG_VOLUME_BYTES;
        if (needed > this._volumesBuffer!.size) {
            this._volumesBuffer!.destroy();
            const size = 2 ** Math.ceil(Math.log2(needed));
            this._volumesBuffer = device.createBuffer({
                label: 'VolumetricFog/LocalVolumes', size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._volumesData = new ArrayBuffer(size);
            this._injectBGDirty = true;
        }
        if (count > 0) {
            const f32 = new Float32Array(this._volumesData);
            const u32 = new Uint32Array(this._volumesData);
            this.localVolumes.forEach((v, i) => v.writeGpu(f32, u32, i * LOCAL_FOG_VOLUME_BYTES / 4));
            device.queue.writeBuffer(this._volumesBuffer!, 0, this._volumesData, 0, needed);
        }
        // FogMediaParams: albedo, skyAmbientScale, numVolumes, hasSkyLighting, hasClipmapProbes
        const f32 = new Float32Array(this._mediaParams);
        const u32 = new Uint32Array(this._mediaParams);
        f32.set(this.albedo, 0);
        f32[3] = this.skyAmbientScale;
        u32.set([count, this._skyLighting ? 1 : 0, 0, 0], 4);
        device.queue.writeBuffer(this._mediaParamsBuffer!, 0, this._mediaParams);
    }

    private _buildCompositeBG(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const accum = this._froxelGrid.accumTex;
        this._compositeBG = this._device!.createBindGroup({
            label: 'VolumetricFog/CompositeBG',
            layout: this._compositePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: output.createView() },
                { binding: 3, resource: accum.createView() },
                { binding: 4, resource: this._accumSampler! },
                { binding: 5, resource: { buffer: this._compositeParamsBuffer! } },
                { binding: 6, resource: this._noShafts!.createView() },
            ],
        });
        this._currentInput  = input;
        this._currentDepth  = depth;
        this._currentOutput = output;
        this._currentAccum  = accum;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number
    ): void {
        if (!this._injectPipeline || !this._compositePipeline) return;

        const device = this._device!;
        const grid = this._froxelGrid;
        const time = this.time ?? (performance.now() - this._startTime) / 1000;

        // the media, the lights, and the bind group when they, a shadow map, the sky or the grid changed
        this._uploadMedia();
        if (this._shadows.prepare(device) || this._injectBGDirty || grid.scatterExtinctionTex !== this._injectTarget) {
            this._rebuildInjectBG();
        }

        mat4.multiply(this._vp, camera.projectionMatrix.internalMat4, camera.viewMatrix.internalMat4);
        mat4.invert(this._invVP, this._vp);
        const iv = camera.inverseViewMatrix.internalMat4;

        // ── FogParams ──
        const f32 = new Float32Array(this._fogParams);
        const u32 = new Uint32Array(this._fogParams);
        f32.set(this._invVP, 0);
        f32.set([iv[12], iv[13], iv[14]], 16);
        f32[19] = this.baseDensity;
        f32.set(this.windDir.map((w) => w * time), 20);
        f32[23] = this.heightFalloff;
        f32.set(this.ambient, 24);
        f32[27] = this.fogHeight;
        f32[28] = grid.near;
        f32[29] = grid.far;
        f32[30] = camera.near;
        f32[31] = camera.far;
        u32[32] = grid.gridW;
        u32[33] = grid.gridH;
        u32[34] = grid.gridD;
        u32[35] = this._shadows.dirCount;
        u32[36] = this._shadows.pointCount;
        u32[37] = this._shadows.hasShadowMap ? 1 : 0;
        u32[38] = this._shadows.hasPointShadows ? 1 : 0;
        f32[39] = this.extinctionCoeff;
        f32[40] = this.anisotropy;
        f32[41] = this.startDistance;
        u32[42] = grid.isTemporal ? this._frame : 0;
        u32[43] = 0;                       // skipSpots: no raymarched spot shafts
        f32.set([0, 0, 0, 1], 44);         // clipPlane: keep all the fog
        f32[48] = this.reach();
        this._frame = this._frame % 1024 + 1;
        device.queue.writeBuffer(this._fogParamsBuffer!, 0, this._fogParams);

        // ── CompositeParams ──
        device.queue.writeBuffer(this._compositeParamsBuffer!, 0, new Float32Array([
            camera.near, camera.far, grid.near, grid.far, grid.gridD, width, height, 0,
        ]));

        // Rebuild composite bind group on texture change (ping-pong, or a resized grid)
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput
            || grid.accumTex !== this._currentAccum) {
            this._buildCompositeBG(input, depth, output);
        }

        // ── 1. inject density and lighting into the froxels ──
        const injectPass = commandEncoder.beginComputePass({ label: 'VolumetricFog/Inject', timestampWrites: gpuPass('VolumetricFog/Inject') });
        injectPass.setPipeline(this._injectPipeline);
        injectPass.setBindGroup(0, this._injectBG!);
        injectPass.dispatchWorkgroups(
            Math.ceil(grid.gridW / 4),
            Math.ceil(grid.gridH / 4),
            Math.ceil(grid.gridD / 4)
        );
        injectPass.end();

        // ── 2. temporal reprojection (no-op unless temporal), 3. front-to-back accumulation ──
        grid.temporalBlend(
            commandEncoder,
            this._invVP as Float32Array,
            this._vp as Float32Array,
            camera.near,
            camera.far
        );
        grid.accumulate(commandEncoder);

        // ── 4. composite over the scene ──
        const compositePass = commandEncoder.beginComputePass({ label: 'VolumetricFog/Composite', timestampWrites: gpuPass('VolumetricFog/Composite') });
        compositePass.setPipeline(this._compositePipeline);
        compositePass.setBindGroup(0, this._compositeBG!);
        compositePass.dispatchWorkgroups(
            Math.ceil(width / 8),
            Math.ceil(height / 8)
        );
        compositePass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // Composite params updated every frame. Bind group rebuilt on texture change in render().
    }

    destroy(): void {
        for (const b of [
            this._fogParamsBuffer, this._compositeParamsBuffer, this._mediaParamsBuffer, this._volumesBuffer,
            this._noSkyLighting, this._noOcclusionParams, this._noClipProbeGrid, this._noClipProbes,
        ]) b?.destroy();
        this._noOcclusionVolume?.destroy();
        this._noShafts?.destroy();
        this._shadows.destroy();
        this._fogParamsBuffer = null;
        this._compositeParamsBuffer = null;
        this._injectPipeline = null;
        this._compositePipeline = null;
        this._injectBG = null;
        this._injectTarget = null;
        this._compositeBG = null;
        this._currentInput = null;
        this._currentAccum = null;
    }
}

export { VolumetricFogEffect };
