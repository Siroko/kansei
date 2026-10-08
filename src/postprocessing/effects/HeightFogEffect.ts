import { mat4 } from 'gl-matrix';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { HEIGHT_FOG_SOURCE } from '../../atmosphere/AtmosphereWGSL';
import { SKY_LIGHTING_BYTES } from '../../atmosphere/SkyAtmosphere';
import { gpuPass } from '../../profiling/Profiler';

// Unreal's ExponentialHeightFog, as the Rust engine's `postprocessing/effects/height_fog.rs` on the
// same WGSL (`shaders/height_fog.wgsl`).

type Vec3 = [number, number, number];

/**
 * One exponential layer: extinction `density` per metre at `height`, falling by e every
 * 1 / `heightFalloff` metres above it (and rising below it).
 */
export interface HeightFogLayer {
    density: number;
    heightFalloff: number;
    height: number;
}

/**
 * A layer from Unreal's `FogDensity`, `FogHeightFalloff` and the fog actor's height in metres, so
 * the fog is as opaque as Unreal draws it. Unreal takes both coefficients per 1000 cm and in
 * base 2, and its line integral is ln 2 times the base-2 integral; its transmittance is
 * `2^-integral`. Per metre in base e: the falloff is x 0.1 x ln 2, the density x 0.1 x (ln 2)².
 */
export function heightFogLayerFromUnreal(fogDensity: number, fogHeightFalloff: number, heightM: number): HeightFogLayer {
    const ln2 = Math.LN2;
    return { density: fogDensity * 0.1 * ln2 * ln2, heightFalloff: fogHeightFalloff * 0.1 * ln2, height: heightM };
}

/** Optical depth of `layer` along `origin + dir * t`, t in [t0, t1] (dir unit), as the shader computes it. */
export function heightFogOpticalDepth(layer: HeightFogLayer, origin: Vec3, dir: Vec3, t0: number, t1: number): number {
    if (layer.density <= 0 || t1 <= t0) return 0;
    const start = layer.density * Math.exp(Math.min(-layer.heightFalloff * (origin[1] + dir[1] * t0 - layer.height), 80));
    const k = layer.heightFalloff * dir[1];
    const len = t1 - t0;
    const x = k * len;
    const shape = Math.abs(x) > 1e-4 ? (1 - Math.exp(Math.min(-x, 80))) / k : len;
    return start * shape;
}

// the WGSL `HeightFogParams`
const PARAMS_BYTES = 192;

/**
 * Analytic exponential height fog, after Unreal's `ExponentialHeightFog`: per pixel, the line
 * integral of one or two exponential layers from the camera (from `startDistance`) to the
 * surface, or to `skyDistance` for the sky, fading the scene toward a fog colour. The colour is
 * `inscattering` plus the sky's distant light when a sky is bound (`setSkyLighting`, times
 * `skyAmbientScale`), with a lobe of `directionalInscattering` toward the sun.
 *
 * It is the far fog of a scene: start it where the volumetric fog's froxels end (their `far`)
 * and put it after the `AtmosphereEffect` and before the `VolumetricFogEffect` in the chain, so
 * near fog lies in front of far fog, which lies in front of the aerial perspective and the sky.
 */
export class HeightFogEffect extends PostProcessingEffect {
    /** The second layer is off while its density is 0. */
    public layers: [HeightFogLayer, HeightFogLayer];
    /** Fog luminance at full opacity (Unreal's fog inscattering luminance). */
    public inscattering: Vec3 = [0, 0, 0];
    /** Scales the sky's light on the fog once a sky is bound. */
    public skyAmbientScale = 1;
    /** Luminance of the lobe toward the light; zero switches it off. */
    public directionalInscattering: Vec3 = [0, 0, 0];
    public directionalExponent = 4;
    public directionalStartDistance = 0;
    /** Toward the light, for the lobe when no sky is bound (the sky's sun otherwise). */
    public lightDirection: Vec3 = [0, 1, 0];
    /** Metres before which there is no fog, along each ray (Unreal's StartDistance). */
    public startDistance = 0;
    /**
     * Where the volumetric fog's froxels end, as a depth along the view (its grid's `far`; 0: no
     * volumetric fog): the fog starts on that plane, so the two neither overlap nor leave a gap.
     */
    public volumetricFogDistance = 0;
    /** Nothing farther than this gets fog (0: no cutoff). */
    public cutoffDistance = 0;
    public maxOpacity = 1;
    /** Distance the sky is fogged as, metres. */
    public skyDistance = 100_000;

    private _skyLighting: GPUBuffer | null = null;
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _params: GPUBuffer | null = null;
    private _noSky: GPUBuffer | null = null;
    private readonly _paramsData = new ArrayBuffer(PARAMS_BYTES);
    private _bindGroup: GPUBindGroup | null = null;
    private _bound: [GPUTexture | null, GPUTexture | null, GPUTexture | null, GPUBuffer | null] = [null, null, null, null];

    constructor(layer: HeightFogLayer = { density: 0.002, heightFalloff: 0.01, height: 0 }) {
        super();
        this.layers = [{ ...layer }, { density: 0, heightFalloff: 0, height: 0 }];
    }

    /**
     * Colour the fog with a sky (`SkyAtmosphere.bindings.skyLighting`): its distant light (the
     * mean radiance all round from 6 km up, `SkyLighting.distantSkyLight`, as Unreal's fog adds),
     * and its sun for the directional lobe (off while the sun is below the horizon).
     */
    setSkyLighting(skyLighting: GPUBuffer | null): void {
        this._skyLighting = skyLighting;
    }

    /**
     * Transmittance of the fog between `origin` and `point` (world space), as the shader (without
     * `volumetricFogDistance`, which depends on the view).
     */
    transmittance(origin: Vec3, point: Vec3): number {
        const d: Vec3 = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
        const dist = Math.hypot(d[0], d[1], d[2]);
        if (dist <= this.startDistance || (this.cutoffDistance > 0 && dist > this.cutoffDistance)) return 1;
        const dir: Vec3 = [d[0] / dist, d[1] / dist, d[2] / dist];
        const depth = this.layers.reduce((sum, l) => sum + heightFogOpticalDepth(l, origin, dir, this.startDistance, dist), 0);
        return 1 - Math.min(1 - Math.exp(-depth), this.maxOpacity);
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const C = GPUShaderStage.COMPUTE;
        const layout = device.createBindGroupLayout({
            label: 'HeightFog/BGL',
            entries: [
                { binding: 0, visibility: C, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: C, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: C, buffer: { type: 'uniform' } },
                { binding: 4, visibility: C, buffer: { type: 'uniform' } },
            ],
        });
        this._pipeline = device.createComputePipeline({
            label: 'HeightFog',
            layout: device.createPipelineLayout({ label: 'HeightFog', bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label: 'HeightFog', code: HEIGHT_FOG_SOURCE }), entryPoint: 'main' },
        });
        const buffer = (label: string, size: number) => device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._params = buffer('HeightFog/Params', PARAMS_BYTES);
        this._noSky = buffer('HeightFog/NoSkyLighting', SKY_LIGHTING_BYTES);
        this.initialized = true;
    }

    private _writeParams(camera: Camera): void {
        const f = new Float32Array(this._paramsData);
        const u = new Uint32Array(this._paramsData);
        mat4.invert(f.subarray(0, 16), camera.viewProjection(mat4.create()));
        const inv = camera.inverseViewMatrix.internalMat4;
        const put = (offset: number, v: Vec3, w: number) => { f.set(v, offset); f[offset + 3] = w; };
        const layer = (offset: number, l: HeightFogLayer) => f.set([Math.max(l.density, 0), l.heightFalloff, l.height, 0], offset);
        put(16, [inv[12], inv[13], inv[14]], Math.max(this.startDistance, 0));
        put(20, this.inscattering, Math.max(this.cutoffDistance, 0));
        put(24, this.directionalInscattering, Math.max(this.directionalExponent, 0));
        put(28, this.lightDirection, Math.max(this.directionalStartDistance, 0));
        layer(32, this.layers[0]);
        layer(36, this.layers[1]);
        f[40] = Math.min(Math.max(this.maxOpacity, 0), 1);
        f[41] = this.skyAmbientScale;
        f[42] = Math.max(this.skyDistance, 0);
        u[43] = this._skyLighting ? 1 : 0;
        // the camera's forward axis: -Z of its world matrix
        const len = Math.hypot(inv[8], inv[9], inv[10]) || 1;
        put(44, [-inv[8] / len, -inv[9] / len, -inv[10] / len], Math.max(this.volumetricFogDistance, 0));
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
        if (!this._pipeline) return;
        const device = this._device!;
        this._writeParams(camera);
        device.queue.writeBuffer(this._params!, 0, this._paramsData);
        const sky = this._skyLighting ?? this._noSky!;
        const b = this._bound;
        if (!this._bindGroup || b[0] !== input || b[1] !== depth || b[2] !== output || b[3] !== sky) {
            this._bindGroup = device.createBindGroup({
                label: 'HeightFog/BG',
                layout: this._pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: input.createView() },
                    { binding: 1, resource: depth.createView() },
                    { binding: 2, resource: output.createView() },
                    { binding: 3, resource: { buffer: this._params! } },
                    { binding: 4, resource: { buffer: sky } },
                ],
            });
            this._bound = [input, depth, output, sky];
        }
        const pass = commandEncoder.beginComputePass({ label: 'HeightFog', timestampWrites: gpuPass('HeightFog') });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    resize(_width: number, _height: number, _gbuffer: GBuffer): void {
        // the GBuffer's textures are new: the bind group follows the next render's
        this._bindGroup = null;
    }

    destroy(): void {
        this._params?.destroy();
        this._noSky?.destroy();
        this._params = null;
        this._noSky = null;
        this._pipeline = null;
        this._bindGroup = null;
        this._bound = [null, null, null, null];
        this.initialized = false;
    }
}
