import { mat4 } from 'gl-matrix';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { SSGI_COMPOSITE_WGSL, SSGI_TEMPORAL_WGSL, SSGI_TRACE_WGSL } from '../../materials/shaders/SharedWGSL';
import { SKY_LIGHTING_BYTES } from '../../atmosphere/SkyAtmosphere';
import { gpuPass } from '../../profiling/Profiler';

/** Bytes of the WGSL `SsgiParams` (`ssgi_common.wgsl`; Rust `SsgiParamsGpu`): five mat4, then 16 scalars. */
export const SSGI_PARAMS_BYTES = 384;

/**
 * The search radius on screen at most, as a share of the image's height. A quarter held the
 * radius to about 0.3 times the depth in metres whatever `radiusM` asked, and a Cornell box got
 * 10-40 % of its bounce; the full height costs no more (the steps stay as many) and gets 70-95 %.
 */
const MAX_RADIUS_SCREEN = 1.0;

/**
 * How much work the global illumination does per frame (Rust `GiQuality`): `low` a quarter of
 * the resolution each way, 2 slices of 6 steps a side; `medium` half resolution, 2 slices of 8
 * steps; `high` half resolution, 4 slices of 12 steps; `ultra` full resolution, 4 slices of 16.
 */
export type GiQuality = 'low' | 'medium' | 'high' | 'ultra';

export const GiQuality = {
    /** (resolution scale, slices, steps per side) */
    settings(quality: GiQuality): [number, number, number] {
        switch (quality) {
            case 'low': return [0.25, 2, 6];
            case 'medium': return [0.5, 2, 8];
            case 'high': return [0.5, 4, 12];
            case 'ultra': return [1.0, 4, 16];
        }
    },
    /** The quality named `name`, or null. */
    fromName(name: string | null | undefined): GiQuality | null {
        return name === 'low' || name === 'medium' || name === 'high' || name === 'ultra' ? name : null;
    },
};

/** What `ScreenSpaceGIEffect` starts with. Rust: `ScreenSpaceGIOptions`. */
export interface ScreenSpaceGIOptions {
    quality: GiQuality;
    /** Metres searched around each point for surfaces that bounce light onto it. */
    radiusM: number;
    /** Metres assumed behind each depth sample (thin things let light past them). */
    thicknessM: number;
    /** Scale of the bounce (1 is physical). */
    intensity: number;
    /**
     * How much of the sky's ambient light is taken out where the sky is hidden (needs
     * `setSkyLighting`, and materials lit by the sky's SH as `SKY_LIGHTING_WGSL` does).
     */
    ambientOcclusion: number;
    /** Weight of each new frame in the accumulated result. */
    temporalBlend: number;
}

export function defaultScreenSpaceGIOptions(): ScreenSpaceGIOptions {
    return { quality: 'medium', radiusM: 3, thicknessM: 0.5, intensity: 1, ambientOcclusion: 1, temporalBlend: 0.1 };
}

interface Targets {
    width: number;
    height: number;
    trace: GPUTexture;
    history: [GPUTexture, GPUTexture];
}

interface Gpu {
    params: GPUBuffer;
    trace: GPUComputePipeline;
    traceBGL: GPUBindGroupLayout;
    temporal: GPUComputePipeline;
    temporalBGL: GPUBindGroupLayout;
    composite: GPUComputePipeline;
    compositeBGL: GPUBindGroupLayout;
    sampler: GPUSampler;
    /** Bound as the sky's lighting without one (`hasSky` 0). */
    noSky: GPUBuffer;
    targets: Targets | null;
}

/**
 * Screen-space global illumination: one bounce of the light on screen onto every diffuse
 * surface, and ambient occlusion of the sky's light, from the GBuffer (after Therrien et al.
 * 2023, visibility bitmasks). It works on any scene, with no preprocessing: foliage, instanced
 * geometry and alpha-tested cards included. Light from surfaces off screen or hidden from the
 * camera is not seen. Rust: `postprocessing::effects::ScreenSpaceGIEffect` (`ssgi.rs`), on the
 * same WGSL.
 *
 * Opt-in: add it first in the chain (before the atmosphere and the fogs, so the bounce lies on
 * the surfaces under the aerial perspective). `enabled = false` skips it at no cost, and
 * `quality` trades resolution and samples for time. It adds light only where a material writes
 * the GBuffer's albedo (and uses its normal when written, rebuilding one from depth otherwise).
 *
 * ```ts
 * const gi = new ScreenSpaceGIEffect({ quality: 'high' });
 * gi.setSkyLighting(sky.bindings.skyLighting);   // optional: the sky's ambient occlusion
 * const volume = new PostProcessingVolume(renderer, [gi, new AtmosphereEffect(sky), tonemap]);
 * ```
 */
export class ScreenSpaceGIEffect extends PostProcessingEffect {
    public enabled = true;
    public quality: GiQuality;
    public radiusM: number;
    public thicknessM: number;
    public intensity: number;
    public ambientOcclusion: number;
    public temporalBlend: number;
    /**
     * Debug view: output only the light the bounce adds (albedo / pi times its irradiance),
     * black elsewhere, in place of the lit image.
     */
    public showIndirect = false;

    private skyLighting: GPUBuffer | null = null;
    private prevViewProj: mat4 | null = null;
    private lastCameraFrame: number | null = null;
    private frame = 0;
    private gpu: Gpu | null = null;
    private device: GPUDevice | null = null;
    private readonly paramsData = new ArrayBuffer(SSGI_PARAMS_BYTES);

    constructor(options: Partial<ScreenSpaceGIOptions> = {}) {
        super();
        const o = { ...defaultScreenSpaceGIOptions(), ...options };
        this.quality = o.quality;
        this.radiusM = o.radiusM;
        this.thicknessM = o.thicknessM;
        this.intensity = o.intensity;
        this.ambientOcclusion = o.ambientOcclusion;
        this.temporalBlend = o.temporalBlend;
    }

    /** The sky's lighting (`SkyAtmosphere.bindings.skyLighting`), for ambient occlusion. Null: none. */
    public setSkyLighting(skyLighting: GPUBuffer | null): void {
        this.skyLighting = skyLighting;
    }

    /** Drop the accumulated frames; call on camera cuts. */
    public resetHistory(): void {
        this.prevViewProj = null;
    }

    public isActive(): boolean {
        return this.enabled;
    }

    public initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this.device = device;
        if (!this.gpu) this.gpu = ScreenSpaceGIEffect.initGpu(device);
        this.initialized = true;
    }

    private static initGpu(device: GPUDevice): Gpu {
        const visibility = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } });
        const texture = (binding: number, filterable: boolean): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, texture: { sampleType: filterable ? 'float' : 'unfilterable-float' } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth' } });
        const storage = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, storageTexture: { access: 'write-only', format: 'rgba16float' } });
        const sampler = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, sampler: { type: 'filtering' } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });
        const traceBGL = bgl('SSGI/TraceBGL', [uniform(0), texture(1, false), depth(2), texture(3, false), storage(4)]);
        const temporalBGL = bgl('SSGI/TemporalBGL', [uniform(0), texture(1, false), texture(2, true), depth(3), storage(4), sampler(5)]);
        const compositeBGL = bgl('SSGI/CompositeBGL', [
            uniform(0), texture(1, false), depth(2), texture(3, false), texture(4, false), texture(5, false), uniform(6), storage(7),
        ]);
        const pipeline = (label: string, code: string, layout: GPUBindGroupLayout) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
        });
        return {
            params: device.createBuffer({ label: 'SSGI/Params', size: SSGI_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
            trace: pipeline('SSGI/Trace', SSGI_TRACE_WGSL, traceBGL),
            traceBGL,
            temporal: pipeline('SSGI/Temporal', SSGI_TEMPORAL_WGSL, temporalBGL),
            temporalBGL,
            composite: pipeline('SSGI/Composite', SSGI_COMPOSITE_WGSL, compositeBGL),
            compositeBGL,
            sampler: device.createSampler({
                label: 'SSGI/Linear',
                magFilter: 'linear',
                minFilter: 'linear',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            }),
            noSky: device.createBuffer({ label: 'SSGI/NoSky', size: SKY_LIGHTING_BYTES, usage: GPUBufferUsage.UNIFORM }),
            targets: null,
        };
    }

    private ensureTargets(device: GPUDevice, width: number, height: number): Targets {
        const gpu = this.gpu!;
        const [scale] = GiQuality.settings(this.quality);
        const w = Math.max(Math.ceil(width * scale), 1);
        const h = Math.max(Math.ceil(height * scale), 1);
        if (gpu.targets && gpu.targets.width === w && gpu.targets.height === h) return gpu.targets;
        if (gpu.targets) for (const t of [gpu.targets.trace, ...gpu.targets.history]) t.destroy();
        const target = (label: string) => device.createTexture({
            label,
            size: [w, h],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        gpu.targets = { width: w, height: h, trace: target('SSGI/Trace'), history: [target('SSGI/HistoryA'), target('SSGI/HistoryB')] };
        this.prevViewProj = null;
        return gpu.targets;
    }

    /**
     * Record the trace and the temporal filter, not the composite: the accumulated bounce (rgb
     * its irradiance, a the share of the hemisphere left open) and its size, for an effect that
     * composites it with light from elsewhere (`VoxelGIEffect`'s near field).
     */
    public encodeTrace(
        device: GPUDevice,
        commandEncoder: GPUCommandEncoder,
        gbuffer: GBuffer,
        input: GPUTexture,
        depth: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
    ): { texture: GPUTexture; size: [number, number] } {
        this.device = device;
        if (!this.gpu) this.gpu = ScreenSpaceGIEffect.initGpu(device);
        const gpu = this.gpu;
        const t = this.ensureTargets(device, width, height);
        // a gap in the camera's frames (the effect was off, or a cut) invalidates the history
        const cameraFrame = camera.frame;
        if (this.lastCameraFrame !== null && cameraFrame !== this.lastCameraFrame && cameraFrame !== ((this.lastCameraFrame + 1) >>> 0)) {
            this.prevViewProj = null;
        }
        this.lastCameraFrame = cameraFrame;
        const [, slices, steps] = GiQuality.settings(this.quality);
        const proj = camera.projectionMatrix.internalMat4;
        const view = camera.viewMatrix.internalMat4;
        const viewProj = mat4.multiply(mat4.create(), proj, view);
        const f32 = new Float32Array(this.paramsData);
        const u32 = new Uint32Array(this.paramsData);
        f32.set(proj, 0);
        f32.set(mat4.invert(mat4.create(), proj), 16);
        f32.set(view, 32);
        f32.set(mat4.invert(mat4.create(), view), 48);
        f32.set(this.prevViewProj ?? viewProj, 64);
        f32.set([width, height, t.width, t.height], 80);
        f32[84] = Math.max(this.radiusM, 0.01);
        f32[85] = Math.max(this.thicknessM, 0);
        f32[86] = Math.max(this.intensity, 0);
        f32[87] = Math.min(Math.max(this.ambientOcclusion, 0), 1);
        u32[88] = slices;
        u32[89] = steps;
        u32[90] = this.frame;
        u32[91] = this.prevViewProj ? 1 : 0;
        f32[92] = height * MAX_RADIUS_SCREEN;
        f32[93] = Math.min(Math.max(this.temporalBlend, 0.01), 1);
        u32[94] = this.skyLighting ? 1 : 0;
        u32[95] = this.showIndirect ? 1 : 0;
        device.queue.writeBuffer(gpu.params, 0, this.paramsData);
        const current = this.frame % 2;
        this.frame = (this.frame + 1) >>> 0;
        this.prevViewProj = viewProj;

        const params = { buffer: gpu.params };
        const depthView = depth.createView();
        const traceView = t.trace.createView();
        const trace = device.createBindGroup({
            label: 'SSGI/TraceBG',
            layout: gpu.traceBGL,
            entries: [params, input.createView(), depthView, gbuffer.normalTexture.createView(), traceView]
                .map((resource, binding) => ({ binding, resource })),
        });
        const temporal = device.createBindGroup({
            label: 'SSGI/TemporalBG',
            layout: gpu.temporalBGL,
            entries: [params, traceView, t.history[1 - current].createView(), depthView, t.history[current].createView(), gpu.sampler]
                .map((resource, binding) => ({ binding, resource })),
        });
        const pass = commandEncoder.beginComputePass({ label: 'SSGI', timestampWrites: gpuPass('SSGI') });
        pass.setPipeline(gpu.trace);
        pass.setBindGroup(0, trace);
        pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
        pass.setPipeline(gpu.temporal);
        pass.setBindGroup(0, temporal);
        pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
        pass.end();
        return { texture: t.history[current], size: [t.width, t.height] };
    }

    public render(
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
        if (!gbuffer) throw new Error('ScreenSpaceGIEffect reads the GBuffer\'s normals and albedo (render it through a PostProcessingVolume)');
        const device = this.device!;
        const { texture: gi } = this.encodeTrace(device, commandEncoder, gbuffer, input, depth, camera, width, height);
        const gpu = this.gpu!;
        const composite = device.createBindGroup({
            label: 'SSGI/CompositeBG',
            layout: gpu.compositeBGL,
            entries: [
                { buffer: gpu.params }, input.createView(), depth.createView(), gi.createView(),
                gbuffer.albedoTexture.createView(), gbuffer.normalTexture.createView(),
                { buffer: this.skyLighting ?? gpu.noSky }, output.createView(),
            ].map((resource, binding) => ({ binding, resource })),
        });
        const pass = commandEncoder.beginComputePass({ label: 'SSGI/Composite', timestampWrites: gpuPass('SSGI/Composite') });
        pass.setPipeline(gpu.composite);
        pass.setBindGroup(0, composite);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    public resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    public destroy(): void {
        const gpu = this.gpu;
        if (!gpu) return;
        gpu.params.destroy();
        gpu.noSky.destroy();
        if (gpu.targets) for (const t of [gpu.targets.trace, ...gpu.targets.history]) t.destroy();
        this.gpu = null;
        this.initialized = false;
    }
}
