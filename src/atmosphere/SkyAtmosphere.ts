import { mat4 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { gpuPass } from '../profiling/Profiler';
import type { HeightFogEffect, HeightFogLayer } from '../postprocessing/effects/HeightFogEffect';
import {
    ATMOSPHERE_BYTES, AtmosphereParams, CelestialLight, Vec3, atmosphereGpu, celestialAngularRadius, celestialDiskLuminance,
    earthAtmosphere, moonLight, sunLight, transmittanceToSpace,
} from './AtmosphereParams';
import {
    AERIAL_PERSPECTIVE_SOURCE, DISTANT_SKY_LIGHT_SOURCE, MULTI_SCATTERING_SOURCE, SKY_LIGHTING_SOURCE, SKY_VIEW_SOURCE,
    TRANSMITTANCE_SOURCE,
} from './AtmosphereWGSL';

// A physically based sky after Hillaire 2020, as the Rust engine's `atmosphere/sky_atmosphere.rs`
// on the same WGSL. The environment cubemap and the clouds' shadow are not part of it yet.

/** Format of every LUT and of the aerial-perspective volumes. */
export const LUT_FORMAT: GPUTextureFormat = 'rgba16float';
/** Limb darkening of the sun's disk (the moon's is flat); the composite shader uses the same. */
export const SUN_LIMB_DARKENING = 0.6;
/** The cloud map's size (cloud_map.wgsl): azimuth by zenith angle. */
export const CLOUD_MAP_SIZE: [number, number] = [128, 64];
/** Lowest camera altitude the LUTs are built for, km: below it the horizon maths loses precision. */
const MIN_CAMERA_ALTITUDE_KM = 0.005;

/** Bytes of the WGSL `SkyFrame` (frame.wgsl), `SkyLighting` (sky_lighting.wgsl) and `SkyCapture` (sky_capture.wgsl). */
export const SKY_FRAME_BYTES = 208;
export const SKY_LIGHTING_BYTES = 240;
export const SKY_CAPTURE_BYTES = 80;

/**
 * LUT resolutions. The defaults are Hillaire 2020's, except the sky-view LUT, which spans the
 * full world azimuth (so the sun and the moon can share it) and has more columns for that.
 */
export interface SkyAtmosphereOptions {
    transmittanceSize: [number, number];
    multiScatteringSize: number;
    skyViewSize: [number, number];
    /** Aerial-perspective volume: screen-aligned columns and depth slices. */
    aerialPerspectiveSize: [number, number, number];
    /** Distance covered by the aerial-perspective volume, km (farther surfaces use its last slice). */
    aerialPerspectiveDistanceKm: number;
}

export function defaultSkyAtmosphereOptions(): SkyAtmosphereOptions {
    return {
        transmittanceSize: [256, 64],
        multiScatteringSize: 32,
        skyViewSize: [256, 128],
        aerialPerspectiveSize: [32, 32, 32],
        aerialPerspectiveDistanceKm: 32,
    };
}

/**
 * The scene's exponential height fog as the sky lighting captures it (`SkyAtmosphere.captureFog`):
 * composited at infinite distance over the sky and the clouds, as seen from `captureHeightM`, as
 * Unreal's real-time sky-light capture does with its ExponentialHeightFog. Seen from low down it
 * covers the horizon and, opaque below it, the ground; the sky lighting still sees the lit ground
 * below the horizon unless `SkyAtmosphere.lightingSeesGround` is off.
 */
export interface SkyCaptureFog {
    /** The fog's layers (as `HeightFogEffect.layers`); the second is off while its density is 0. */
    layers: [HeightFogLayer, HeightFogLayer];
    /** Its colour at full opacity, cd/m² (as `HeightFogEffect.inscattering`). */
    inscattering: Vec3;
    /** At most this opaque (as `HeightFogEffect.maxOpacity`). */
    maxOpacity: number;
    /** World height of the point the sky is captured from, metres (Unreal: its SkyLight actor). */
    captureHeightM: number;
    /**
     * How much of the sky's distant light (`SkyLighting.distantSkyLight`) the fog adds to its
     * colour, as `HeightFogEffect.skyAmbientScale`.
     */
    skyAmbientScale: number;
}

/** The fog a `HeightFogEffect` draws (its layers, colour and opacity), captured from `captureHeightM`. */
export function skyCaptureFogFromHeightFog(fog: HeightFogEffect, captureHeightM: number): SkyCaptureFog {
    return {
        layers: [{ ...fog.layers[0] }, { ...fog.layers[1] }],
        inscattering: [...fog.inscattering],
        maxOpacity: fog.maxOpacity,
        captureHeightM,
        skyAmbientScale: fog.skyAmbientScale,
    };
}

/**
 * What the sky lighting sees below the horizon: `'ground'`, a Lambertian ground of
 * `skyLightGroundAlbedo` lit by the sky and the lights, behind the air (under a capture fog the
 * fog, when `lightingSeesGround` is off); or a radiance whatever the fog (Unreal's
 * `bLowerHemisphereIsBlack` with its `LowerHemisphereColor`; black is `{ color: [0, 0, 0] }`).
 */
export type SkyLowerHemisphere = 'ground' | { color: Vec3 };

/** The GPU handles an effect or a material needs to read the atmosphere (Rust `SkyAtmosphereBindings`). */
export interface SkyAtmosphereBindings {
    /** `Atmosphere` uniform (the WGSL struct in `ATMOSPHERE_WGSL`). */
    atmosphere: GPUBuffer;
    /** `SkyFrame` uniform, rewritten by every `SkyAtmosphere.update`. */
    frame: GPUBuffer;
    transmittance: GPUTextureView;
    multiScattering: GPUTextureView;
    skyView: GPUTextureView;
    /** Aerial-perspective volume (3D): in-scattered light toward the camera, and transmittance. */
    apScattering: GPUTextureView;
    apTransmittance: GPUTextureView;
    /**
     * `SkyLighting` (see `SKY_LIGHTING_WGSL`): the sky's radiance as order-2 SH and the sun and
     * the moon at the camera, rewritten by every update. Usable as a uniform or a storage buffer.
     */
    skyLighting: GPUBuffer;
    /** Linear, clamping: for the transmittance and multiple-scattering LUTs. */
    lutSampler: GPUSampler;
    /** Linear, repeating in u (the azimuth): for the sky-view LUT. */
    skyViewSampler: GPUSampler;
    /**
     * The clouds around the camera (rgba16float, azimuth by zenith angle; rgb their light, a their
     * opacity), read by the sky lighting. Empty (a clear sky) until clouds write it.
     */
    cloudMap: GPUTextureView;
    /**
     * The camera frame (plus one; 0 never) the clouds last wrote `cloudMap` in. The sky lighting
     * reads the map only while it is that fresh, so clouds taken out of the chain leave no trace.
     */
    cloudMapFrame: number;
}

/**
 * A physically based sky and atmosphere after Hillaire 2020, as the LUTs the sky, the aerial
 * perspective and the sky lighting are rendered from:
 * - **transmittance** (256x64): transmittance to space by altitude and zenith angle;
 * - **multiple scattering** (32x32): all scattering orders >= 2, per unit illuminance;
 * - **sky view** (256x128): the sky's luminance around the camera, every frame;
 * - **aerial perspective** (32x32x32 camera froxels, to 32 km): the light scattered toward the
 *   camera and the transmittance in front of every surface, every frame;
 * - **sky lighting**: the sky's radiance as order-2 spherical harmonics (with light bounced off
 *   the ground below the horizon), the sun and the moon at the camera, and the sky's distant
 *   light, every frame, for materials and media (`SKY_LIGHTING_WGSL`).
 *
 * The first two depend only on `params` and are rebuilt when they change. The sun can be
 * anywhere, including below the horizon at dusk, where the sky is lit only by the light scattered
 * high above the planet's shadow. A moon (off by default) scatters as a second light.
 *
 * World space is metres, Y up, with the planet's surface at `y = -originAltitudeM` and its centre
 * straight below the world origin.
 *
 * ```ts
 * const sky = new SkyAtmosphere(renderer.gpuDevice);
 * sky.sun.direction = directionFromElevationBearing(-2.5, 140);
 * sky.sun.illuminance = [100000, 100000, 100000]; // lux
 * const volume = new PostProcessingVolume(renderer, [new AtmosphereEffect(sky), tonemap]);
 * // each frame, once the camera is placed:
 * sky.update(camera);
 * volume.render(scene, camera);
 * ```
 */
export class SkyAtmosphere {
    public params: AtmosphereParams = earthAtmosphere();
    public sun: CelestialLight = sunLight();
    public moon: CelestialLight = moonLight();
    /** Altitude of the world origin above the planet's surface, metres. */
    public originAltitudeM = 0;
    /**
     * Albedo of the ground below the horizon in the sky lighting (the light it bounces up); null
     * uses `params.groundAlbedo`, zero leaves the lower hemisphere black.
     */
    public skyLightGroundAlbedo: Vec3 | null = null;
    /**
     * The scene's height fog as the sky lighting captures it, in front of the sky and the clouds
     * (Unreal's real-time sky-light capture); null captures the sky alone.
     */
    public captureFog: SkyCaptureFog | null = null;
    /** What the sky lighting sees below the horizon. */
    public lowerHemisphere: SkyLowerHemisphere = 'ground';
    /**
     * Under a capture fog, whether the sky lighting sees the lit ground below the horizon (the
     * default), as Unreal lights its scene with Lumen; when off, it sees the fog there, as
     * Unreal's SkyLight captures it.
     */
    public lightingSeesGround = true;

    public readonly bindings: SkyAtmosphereBindings;
    /** The transmittance, multiple-scattering and sky-view LUT textures. */
    public readonly lutTextures: [GPUTexture, GPUTexture, GPUTexture];
    /** The aerial-perspective volumes: scattering, transmittance (3D). */
    public readonly aerialPerspectiveTextures: [GPUTexture, GPUTexture];

    private readonly _device: GPUDevice;
    private readonly _capture: GPUBuffer;
    private readonly _distant: GPUBuffer;
    private readonly _ownedTextures: GPUTexture[];
    private readonly _pipelines: {
        transmittance: GPUComputePipeline; transmittanceBG: GPUBindGroup;
        multiScattering: GPUComputePipeline; multiScatteringBG: GPUBindGroup;
        skyView: GPUComputePipeline; skyViewBG: GPUBindGroup;
        aerialPerspective: GPUComputePipeline; aerialPerspectiveBG: GPUBindGroup;
        distant: GPUComputePipeline; distantBG: GPUBindGroup;
        skyLighting: GPUComputePipeline;
        // with the cloud map, and without it (no clouds wrote it last frame)
        skyLightingBGs: [GPUBindGroup, GPUBindGroup];
    };
    private readonly _transmittanceSize: [number, number];
    private readonly _multiScatteringSize: number;
    private readonly _skyViewSize: [number, number];
    private readonly _apSize: [number, number, number];
    private readonly _apDistanceKm: number;
    /** The `Atmosphere` the static LUTs were last built for. */
    private _builtFor: Float32Array | null = null;
    private readonly _frameData = new Float32Array(SKY_FRAME_BYTES / 4);
    private readonly _captureData = new ArrayBuffer(SKY_CAPTURE_BYTES);

    constructor(device: GPUDevice, options: Partial<SkyAtmosphereOptions> = {}) {
        const o = { ...defaultSkyAtmosphereOptions(), ...options };
        this._device = device;
        const uniform = (label: string, size: number) =>
            device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const sampler = (label: string, addressModeU: GPUAddressMode) => device.createSampler({
            label, addressModeU, addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear',
        });
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
        const lut2d = (label: string, [w, h]: [number, number]) =>
            device.createTexture({ label, size: [Math.max(w, 1), Math.max(h, 1)], format: LUT_FORMAT, usage });
        const volume3d = (label: string, [w, h, d]: [number, number, number]) =>
            device.createTexture({ label, size: [Math.max(w, 1), Math.max(h, 1), Math.max(d, 1)], dimension: '3d', format: LUT_FORMAT, usage });

        const ms = Math.max(o.multiScatteringSize, 2);
        this.lutTextures = [
            lut2d('SkyAtmosphere/TransmittanceLUT', o.transmittanceSize),
            lut2d('SkyAtmosphere/MultiScatteringLUT', [ms, ms]),
            lut2d('SkyAtmosphere/SkyViewLUT', o.skyViewSize),
        ];
        this.aerialPerspectiveTextures = [
            volume3d('SkyAtmosphere/AerialPerspectiveScattering', o.aerialPerspectiveSize),
            volume3d('SkyAtmosphere/AerialPerspectiveTransmittance', o.aerialPerspectiveSize),
        ];
        // zero-initialised: no clouds until the clouds write it
        const cloudMap = lut2d('SkyAtmosphere/CloudMap', CLOUD_MAP_SIZE);
        // bound in place of the cloud map while no clouds write it
        const noClouds = lut2d('SkyAtmosphere/NoClouds', [1, 1]);
        this._ownedTextures = [...this.lutTextures, ...this.aerialPerspectiveTextures, cloudMap, noClouds];

        const b: SkyAtmosphereBindings = this.bindings = {
            atmosphere: uniform('SkyAtmosphere/Atmosphere', ATMOSPHERE_BYTES),
            frame: uniform('SkyAtmosphere/Frame', SKY_FRAME_BYTES),
            transmittance: this.lutTextures[0].createView(),
            multiScattering: this.lutTextures[1].createView(),
            skyView: this.lutTextures[2].createView(),
            apScattering: this.aerialPerspectiveTextures[0].createView(),
            apTransmittance: this.aerialPerspectiveTextures[1].createView(),
            skyLighting: device.createBuffer({
                label: 'SkyAtmosphere/SkyLighting',
                size: SKY_LIGHTING_BYTES,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC,
            }),
            lutSampler: sampler('SkyAtmosphere/LutSampler', 'clamp-to-edge'),
            skyViewSampler: sampler('SkyAtmosphere/SkyViewSampler', 'repeat'),
            cloudMap: cloudMap.createView(),
            cloudMapFrame: 0,
        };
        this._capture = uniform('SkyAtmosphere/Capture', SKY_CAPTURE_BYTES);
        // the sky's distant light, before the sky lighting that reads it and passes it on
        this._distant = device.createBuffer({ label: 'SkyAtmosphere/DistantSkyLight', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM });

        const C = GPUShaderStage.COMPUTE;
        const uniformEntry = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'uniform' } });
        const storageBufferEntry = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'storage' } });
        const textureEntry = (binding: number, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: C, texture: { sampleType: 'float', viewDimension } });
        const samplerEntry = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, sampler: { type: 'filtering' } });
        const storageEntry = (binding: number, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: C, storageTexture: { access: 'write-only', format: LUT_FORMAT, viewDimension } });
        const pipeline = (label: string, code: string, entries: GPUBindGroupLayoutEntry[]) => {
            const layout = device.createBindGroupLayout({ label: `${label}BGL`, entries });
            const p = device.createComputePipeline({
                label,
                layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
            });
            return { pipeline: p, layout };
        };
        const group = (label: string, layout: GPUBindGroupLayout, resources: GPUBindingResource[]) => device.createBindGroup({
            label, layout, entries: resources.map((resource, binding) => ({ binding, resource })),
        });
        const buf = (buffer: GPUBuffer): GPUBindingResource => ({ buffer });

        const transmittance = pipeline('SkyAtmosphere/Transmittance', TRANSMITTANCE_SOURCE, [uniformEntry(0), storageEntry(1)]);
        const multiScattering = pipeline('SkyAtmosphere/MultiScattering', MULTI_SCATTERING_SOURCE,
            [uniformEntry(0), textureEntry(1), samplerEntry(2), storageEntry(3)]);
        const skyView = pipeline('SkyAtmosphere/SkyView', SKY_VIEW_SOURCE,
            [uniformEntry(0), uniformEntry(1), textureEntry(2), textureEntry(3), samplerEntry(4), storageEntry(5)]);
        const aerialPerspective = pipeline('SkyAtmosphere/AerialPerspective', AERIAL_PERSPECTIVE_SOURCE,
            [uniformEntry(0), uniformEntry(1), textureEntry(2), textureEntry(3), samplerEntry(4), storageEntry(5, '3d'), storageEntry(6, '3d')]);
        const distant = pipeline('SkyAtmosphere/DistantSkyLight', DISTANT_SKY_LIGHT_SOURCE,
            [uniformEntry(0), uniformEntry(1), textureEntry(2), textureEntry(3), samplerEntry(4), storageBufferEntry(5)]);
        const skyLighting = pipeline('SkyAtmosphere/SkyLighting', SKY_LIGHTING_SOURCE, [
            uniformEntry(0), uniformEntry(1), textureEntry(2), textureEntry(3), samplerEntry(4), samplerEntry(5),
            storageBufferEntry(6), textureEntry(7), uniformEntry(8), uniformEntry(9),
        ]);
        const noCloudsView = noClouds.createView();

        this._pipelines = {
            transmittance: transmittance.pipeline,
            transmittanceBG: group('SkyAtmosphere/TransmittanceBG', transmittance.layout, [buf(b.atmosphere), b.transmittance]),
            multiScattering: multiScattering.pipeline,
            multiScatteringBG: group('SkyAtmosphere/MultiScatteringBG', multiScattering.layout,
                [buf(b.atmosphere), b.transmittance, b.lutSampler, b.multiScattering]),
            skyView: skyView.pipeline,
            skyViewBG: group('SkyAtmosphere/SkyViewBG', skyView.layout,
                [buf(b.atmosphere), buf(b.frame), b.transmittance, b.multiScattering, b.lutSampler, b.skyView]),
            aerialPerspective: aerialPerspective.pipeline,
            aerialPerspectiveBG: group('SkyAtmosphere/AerialPerspectiveBG', aerialPerspective.layout,
                [buf(b.atmosphere), buf(b.frame), b.transmittance, b.multiScattering, b.lutSampler, b.apScattering, b.apTransmittance]),
            distant: distant.pipeline,
            distantBG: group('SkyAtmosphere/DistantSkyLightBG', distant.layout,
                [buf(b.atmosphere), buf(b.frame), b.transmittance, b.multiScattering, b.lutSampler, buf(this._distant)]),
            skyLighting: skyLighting.pipeline,
            skyLightingBGs: [b.cloudMap, noCloudsView].map((clouds) => group('SkyAtmosphere/SkyLightingBG', skyLighting.layout, [
                buf(b.atmosphere), buf(b.frame), b.transmittance, b.skyView, b.lutSampler, b.skyViewSampler,
                buf(b.skyLighting), clouds, buf(this._capture), buf(this._distant),
            ])) as [GPUBindGroup, GPUBindGroup],
        };

        this._transmittanceSize = o.transmittanceSize;
        this._multiScatteringSize = ms;
        this._skyViewSize = o.skyViewSize;
        this._apSize = o.aerialPerspectiveSize;
        this._apDistanceKm = Math.max(o.aerialPerspectiveDistanceKm, 1e-3);
    }

    /** Planet-frame position (km) of a world-space point (metres). */
    public toPlanetKm(world: Vec3): Vec3 {
        return [world[0] * 0.001, (world[1] + this.originAltitudeM) * 0.001 + this.params.bottomRadiusKm, world[2] * 0.001];
    }

    /**
     * Transmittance of the atmosphere from a world-space point toward the sun: 0 when the sun is
     * below that point's horizon. Multiply the sun's illuminance by it to light the scene with the
     * colour the sky is rendered with.
     */
    public sunTransmittance(world: Vec3): Vec3 {
        const p = this.toPlanetKm(world);
        const len = Math.hypot(p[0], p[1], p[2]);
        const r = Math.max(len, this.params.bottomRadiusKm);
        const s = normalizeOr(this.sun.direction, [0, 1, 0]);
        const mu = (p[0] * s[0] + p[1] * s[1] + p[2] * s[2]) / len;
        // the part of the disk above the horizon, as the shaders' horizonVisibility
        const bottom = this.params.bottomRadiusKm;
        const horizon = -Math.sqrt(Math.max((r - bottom) * (r + bottom), 0)) / r;
        const w = celestialAngularRadius(this.sun);
        const x = Math.min(Math.max((mu - horizon + w) / (2 * w), 0), 1);
        const visibility = x * x * (3 - 2 * x);
        const t = transmittanceToSpace(this.params, r - bottom, Math.max(mu, horizon + 1e-5));
        return [t[0] * visibility, t[1] * visibility, t[2] * visibility];
    }

    /**
     * The sun's illuminance after the atmosphere at a world-space point: the colour (times
     * intensity) of a directional light that agrees with the sky.
     */
    public sunIlluminanceAt(world: Vec3): Vec3 {
        const t = this.sunTransmittance(world);
        const e = this.sun.illuminance;
        return [e[0] * t[0], e[1] * t[1], e[2] * t[2]];
    }

    /** The frame uniform for `camera` into `_frameData`. */
    private _writeFrame(camera: Camera): void {
        const f = this._frameData;
        const viewProj = camera.viewProjection(mat4.create());
        mat4.invert(f.subarray(0, 16), viewProj);
        const inv = camera.inverseViewMatrix.internalMat4;
        const eye: Vec3 = [inv[12], inv[13], inv[14]];
        const bottom = this.params.bottomRadiusKm;
        const top = bottom + Math.max(this.params.atmosphereHeightKm, 1e-3);
        // keep the camera inside the atmosphere, above the ground by a few metres
        const p = this.toPlanetKm(eye);
        const r = Math.min(Math.max(Math.hypot(p[0], p[1], p[2]), bottom + MIN_CAMERA_ALTITUDE_KM), top - 1e-3);
        const n = normalizeOr(p, [0, 1, 0]);
        const origin = this.toPlanetKm([0, 0, 0]);
        const put = (offset: number, v: Vec3, w: number) => { f.set(v, offset); f[offset + 3] = w; };
        put(16, [n[0] * r, n[1] * r, n[2] * r], this._apDistanceKm);
        put(20, origin, Math.max(this.params.aerialPerspectiveStartDepthKm, 0));
        put(24, eye, Math.max(this.params.aerialPerspectiveViewDistanceScale, 0));
        put(28, normalizeOr(this.sun.direction, [0, 1, 0]), celestialAngularRadius(this.sun));
        put(32, this.sun.illuminance, celestialDiskLuminance(this.sun, SUN_LIMB_DARKENING));
        put(36, normalizeOr(this.moon.direction, [0, 1, 0]), celestialAngularRadius(this.moon));
        put(40, this.moon.illuminance, celestialDiskLuminance(this.moon, 0));
        put(44, this.params.skyLuminanceFactor, 0);
        put(48, this.skyLightGroundAlbedo ?? this.params.groundAlbedo, 0);
    }

    /** The capture's fog and lower hemisphere (WGSL `SkyCapture`) into `_captureData`. */
    private _writeCapture(): void {
        const f = new Float32Array(this._captureData);
        const u = new Uint32Array(this._captureData);
        f.fill(0);
        const lower = this.lowerHemisphere;
        if (lower !== 'ground') {
            f.set(lower.color, 12);
            u[15] = 1;
        }
        const fog = this.captureFog;
        if (!fog) return;
        const layer = (offset: number, l: HeightFogLayer) => f.set([Math.max(l.density, 0), l.heightFalloff, l.height, 0], offset);
        layer(0, fog.layers[0]);
        layer(4, fog.layers[1]);
        f.set(fog.inscattering, 8);
        u[11] = 1;
        f[16] = fog.captureHeightM;
        f[17] = Math.min(Math.max(fog.maxOpacity, 0), 1);
        f[18] = Math.max(fog.skyAmbientScale, 0);
        u[19] = this.lightingSeesGround ? 1 : 0;
    }

    /**
     * Record this frame's passes: the transmittance and multiple-scattering LUTs when the
     * atmosphere changed, then the sky-view LUT, the aerial perspective, the distant sky light and
     * the sky lighting for the camera. Uses the camera's current view and projection matrices, so
     * place the camera first.
     */
    public encode(encoder: GPUCommandEncoder, camera: Camera): void {
        const queue = this._device.queue;
        const b = this.bindings;
        const p = this._pipelines;
        // the cloud map (0) if the clouds wrote it this frame or the last, else none (1)
        const written = b.cloudMapFrame;
        const clouds = written !== 0 && ((camera.frame - (written - 1)) >>> 0) <= 1 ? 0 : 1;

        const atmosphere = atmosphereGpu(this.params);
        if (!this._builtFor || !atmosphere.every((v, i) => v === this._builtFor![i])) {
            queue.writeBuffer(b.atmosphere, 0, atmosphere);
            const pass = encoder.beginComputePass({ label: 'SkyAtmosphere/StaticLUTs', timestampWrites: gpuPass('SkyAtmosphere/StaticLUTs') });
            pass.setPipeline(p.transmittance);
            pass.setBindGroup(0, p.transmittanceBG);
            pass.dispatchWorkgroups(Math.ceil(this._transmittanceSize[0] / 8), Math.ceil(this._transmittanceSize[1] / 8));
            // one workgroup per texel
            pass.setPipeline(p.multiScattering);
            pass.setBindGroup(0, p.multiScatteringBG);
            pass.dispatchWorkgroups(this._multiScatteringSize, this._multiScatteringSize);
            pass.end();
            this._builtFor = atmosphere;
        }

        this._writeFrame(camera);
        this._writeCapture();
        queue.writeBuffer(b.frame, 0, this._frameData);
        queue.writeBuffer(this._capture, 0, this._captureData);
        const pass = encoder.beginComputePass({ label: 'SkyAtmosphere/SkyView', timestampWrites: gpuPass('SkyAtmosphere/SkyView') });
        pass.setPipeline(p.skyView);
        pass.setBindGroup(0, p.skyViewBG);
        pass.dispatchWorkgroups(Math.ceil(this._skyViewSize[0] / 8), Math.ceil(this._skyViewSize[1] / 8));
        pass.setPipeline(p.aerialPerspective);
        pass.setBindGroup(0, p.aerialPerspectiveBG);
        pass.dispatchWorkgroups(Math.ceil(this._apSize[0] / 8), Math.ceil(this._apSize[1] / 8));
        pass.setPipeline(p.distant);
        pass.setBindGroup(0, p.distantBG);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(p.skyLighting);
        pass.setBindGroup(0, p.skyLightingBGs[clouds]);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /**
     * `encode` on a fresh encoder, submitted at once. Call every frame after placing the camera
     * and before rendering: the scene's materials and the post effects read what it writes.
     */
    public update(camera: Camera): void {
        camera.updateViewMatrix();
        const encoder = this._device.createCommandEncoder({ label: 'SkyAtmosphere' });
        this.encode(encoder, camera);
        this._device.queue.submit([encoder.finish()]);
    }

    /** Force the transmittance and multiple-scattering LUTs to be rebuilt on the next update. */
    public invalidate(): void {
        this._builtFor = null;
    }

    /** Release the GPU resources (the bindings stop working). */
    public destroy(): void {
        for (const t of this._ownedTextures) t.destroy();
        for (const buffer of [this.bindings.atmosphere, this.bindings.frame, this.bindings.skyLighting, this._capture, this._distant]) buffer.destroy();
    }
}

function normalizeOr(v: Vec3, fallback: Vec3): Vec3 {
    const len = Math.hypot(v[0], v[1], v[2]);
    return len > 1e-12 ? [v[0] / len, v[1] / len, v[2] / len] : fallback;
}
