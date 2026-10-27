import { mat4 } from 'gl-matrix';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { SSAO_BLUR_WGSL, SSAO_COMPOSITE_WGSL, SSAO_WGSL } from '../../materials/shaders/SharedWGSL';
import { SKY_LIGHTING_BYTES } from '../../atmosphere/SkyAtmosphere';
import { gpuPass } from '../../profiling/Profiler';

/** Bytes of the WGSL `SSAOParams` (`ssao_common.wgsl`; Rust `SSAOParams`): four mat4, two vec4, then 12 scalars. */
export const SSAO_PARAMS_BYTES = 336;

/** The most hemisphere samples a pixel takes (the kernel's size). */
const MAX_KERNEL = 32;

/**
 * The sampling radius on screen at most, as a share of the image's height: past it the radius
 * shrinks (a close-up's samples would otherwise spread over the whole screen).
 */
const MAX_RADIUS_SCREEN = 0.1;

/**
 * The ambient light the materials add to the colour, which `SSAOEffect` takes the occluded
 * share of: the occlusion darkens it alone, never the direct light or the emission (Rust
 * `SSAOAmbient`).
 * - `hemisphere`: the albedo times a hemisphere of sky blended by the normal's height,
 *   `mix(down, up, n.y * 0.5 + 0.5)`, as `Material.standardLit` adds it from its `skyUp` and
 *   `skyDown`; `up` equal to `down` for a constant ambient (albedo x ambient).
 * - `sky`: the albedo times the irradiance of the sky's SH around the normal, over pi, for
 *   materials lit by `SKY_LIGHTING_WGSL`'s `skyIrradiance` (a `SkyLighting` uniform).
 * - `unlit`: the whole colour less the GBuffer's emissive part, for unlit materials, whose colour
 *   is all ambient (a texture shown as it is, with no lights).
 */
export type SSAOAmbient =
    | { kind: 'hemisphere'; up: [number, number, number]; down: [number, number, number] }
    | { kind: 'sky'; skyLighting: GPUBuffer }
    | { kind: 'unlit' };

export interface SSAOOptions {
    /**
     * Metres around each point searched for occluders: a few times the size of the creases that
     * should darken (centimetres on a tabletop, half a metre in a room). Default 0.5
     */
    radius?: number;
    /**
     * How much nearer than a sample a surface must be to occlude it, per metre of the point's
     * distance from the camera. Default 0.003
     */
    bias?: number;
    /** Hemisphere samples a pixel (1-32). Default 16 */
    kernelSize?: number;
    /** How much of the occluded ambient is taken out (1: all of it). Default 1 */
    strength?: number;
    /** The ambient light it occludes. */
    ambient: SSAOAmbient;
}

/**
 * The hemisphere kernel: `MAX_KERNEL` points in the +z half of the unit ball, more of them near
 * the centre (scaled by lerp(0.1, 1, (i / n)²)) and none within 8.6 degrees of the base, from
 * Rust's fixed seed (xorshift32).
 */
export function ssaoKernel(): Float32Array {
    let seed = 0x9e3779b9 >>> 0;
    const random = () => {
        seed = (seed ^ (seed << 13)) >>> 0;
        seed = (seed ^ (seed >>> 17)) >>> 0;
        seed = (seed ^ (seed << 5)) >>> 0;
        return Math.fround((seed >>> 8) / (1 << 24));
    };
    const data = new Float32Array(MAX_KERNEL * 4);
    for (let i = 0; i < MAX_KERNEL; i++) {
        const phi = Math.fround(random() * Math.fround(Math.PI * 2));
        const cosTheta = Math.fround(0.15 + Math.fround(0.85 * random()));
        const sinTheta = Math.fround(Math.sqrt(Math.fround(1 - Math.fround(cosTheta * cosTheta))));
        const t = Math.fround(i / MAX_KERNEL);
        const scale = Math.fround(0.1 + Math.fround(Math.fround(t * t) * 0.9));
        data.set([Math.cos(phi) * sinTheta * scale, Math.sin(phi) * sinTheta * scale, cosTheta * scale, 0], i * 4);
    }
    return data;
}

interface Targets {
    width: number;
    height: number;
    /** The occlusion, then blurred across. */
    ao: [GPUTexture, GPUTexture];
}

interface Gpu {
    params: GPUBuffer;
    kernel: GPUBuffer;
    noSky: GPUBuffer;
    ao: GPUComputePipeline;
    aoBGL: GPUBindGroupLayout;
    blur: GPUComputePipeline;
    blurBGL: GPUBindGroupLayout;
    composite: GPUComputePipeline;
    compositeBGL: GPUBindGroupLayout;
    targets: Targets | null;
}

/**
 * Screen-space ambient occlusion (Rust `SSAOEffect`, the same WGSL): a hemisphere of samples
 * around each point's normal (the GBuffer's, or one rebuilt from depth where the material wrote
 * none), turned per pixel and projected back to the screen against the depth there, blurred by
 * depth, then taken out of the ambient light alone (`SSAOAmbient`): flat open surfaces keep their
 * light, creases and contacts lose some of their ambient, and the direct light and emission stay
 * as they are.
 *
 * Put it first in the chain. The GI effects occlude the light they add themselves: with one that
 * replaces the materials' ambient, leave this off, or give it only the ambient that effect leaves.
 */
class SSAOEffect extends PostProcessingEffect {
    public radius: number;
    public bias: number;
    public kernelSize: number;
    public strength: number;
    public ambient: SSAOAmbient;
    /** Debug view: output the occlusion (white open, black occluded) in place of the image. */
    public showOcclusion = false;
    private device: GPUDevice | null = null;
    private gpu: Gpu | null = null;
    private readonly paramsData = new ArrayBuffer(SSAO_PARAMS_BYTES);

    constructor(options: SSAOOptions) {
        super();
        this.radius = options.radius ?? 0.5;
        this.bias = options.bias ?? 0.003;
        this.kernelSize = options.kernelSize ?? 16;
        this.strength = options.strength ?? 1;
        this.ambient = options.ambient;
    }

    public initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this.device = device;
        if (!this.gpu) this.gpu = SSAOEffect.initGpu(device);
        this.initialized = true;
    }

    private static initGpu(device: GPUDevice): Gpu {
        const visibility = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } });
        const texture = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'unfilterable-float' } });
        const depth = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth' } });
        const storage = (binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry =>
            ({ binding, visibility, storageTexture: { access: 'write-only', format } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });
        // bindings 0-2 of every pass: the params, the GBuffer's depth and normal
        const aoBGL = bgl('SSAO/BGL', [uniform(0), depth(1), texture(2), uniform(3), storage(4, 'rg32float')]);
        const blurBGL = bgl('SSAO/BlurBGL', [uniform(0), depth(1), texture(2), texture(3), storage(4, 'rg32float')]);
        const compositeBGL = bgl('SSAO/CompositeBGL', [
            uniform(0), depth(1), texture(2), texture(3), texture(4), texture(5), texture(6), uniform(7), storage(8, 'rgba16float'),
        ]);
        const pipeline = (label: string, code: string, layout: GPUBindGroupLayout) => device.createComputePipeline({
            label,
            layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
        });
        const kernel = device.createBuffer({ label: 'SSAO/Kernel', size: MAX_KERNEL * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(kernel, 0, ssaoKernel());
        return {
            params: device.createBuffer({ label: 'SSAO/Params', size: SSAO_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
            kernel,
            noSky: device.createBuffer({ label: 'SSAO/NoSky', size: SKY_LIGHTING_BYTES, usage: GPUBufferUsage.UNIFORM }),
            ao: pipeline('SSAO', SSAO_WGSL, aoBGL),
            aoBGL,
            blur: pipeline('SSAO/Blur', SSAO_BLUR_WGSL, blurBGL),
            blurBGL,
            composite: pipeline('SSAO/Composite', SSAO_COMPOSITE_WGSL, compositeBGL),
            compositeBGL,
            targets: null,
        };
    }

    private ensureTargets(device: GPUDevice, width: number, height: number): Targets {
        const gpu = this.gpu!;
        if (gpu.targets && gpu.targets.width === width && gpu.targets.height === height) return gpu.targets;
        if (gpu.targets) for (const t of gpu.targets.ao) t.destroy();
        const target = (label: string) => device.createTexture({
            label,
            size: [width, height],
            format: 'rg32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        gpu.targets = { width, height, ao: [target('SSAO/Occlusion'), target('SSAO/OcclusionBlurred')] };
        return gpu.targets;
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
        if (!gbuffer) throw new Error('SSAOEffect reads the GBuffer\'s normals, albedo and emission (render it through a PostProcessingVolume)');
        const device = this.device!;
        if (!this.gpu) this.gpu = SSAOEffect.initGpu(device);
        const gpu = this.gpu;
        const t = this.ensureTargets(device, width, height);
        const proj = camera.projectionMatrix.internalMat4;
        const view = camera.viewMatrix.internalMat4;
        const a = this.ambient;
        const f32 = new Float32Array(this.paramsData);
        const u32 = new Uint32Array(this.paramsData);
        f32.set(proj, 0);
        f32.set(mat4.invert(mat4.create(), proj), 16);
        f32.set(view, 32);
        f32.set(mat4.invert(mat4.create(), view), 48);
        f32.set(a.kind === 'hemisphere' ? [...a.up, 0, ...a.down, 0] : [0, 0, 0, 0, 0, 0, 0, 0], 64);
        f32.set([width, height, Math.max(this.radius, 1e-4), Math.max(this.bias, 0)], 72);
        u32[76] = Math.min(Math.max(Math.round(this.kernelSize), 1), MAX_KERNEL);
        f32[77] = Math.max(this.strength, 0);
        f32[78] = height * MAX_RADIUS_SCREEN;
        u32[79] = a.kind === 'hemisphere' ? 0 : a.kind === 'sky' ? 1 : 2;
        u32[80] = this.showOcclusion ? 1 : 0;
        device.queue.writeBuffer(gpu.params, 0, this.paramsData);

        const params = { buffer: gpu.params };
        const depthView = depth.createView();
        const normalView = gbuffer.normalTexture.createView();
        const [ao, blurred] = [t.ao[0].createView(), t.ao[1].createView()];
        const group = (label: string, layout: GPUBindGroupLayout, resources: GPUBindingResource[]) =>
            device.createBindGroup({ label, layout, entries: resources.map((resource, binding) => ({ binding, resource })) });
        const groups: [GPUComputePipeline, GPUBindGroup][] = [
            [gpu.ao, group('SSAO/BindGroup', gpu.aoBGL, [params, depthView, normalView, { buffer: gpu.kernel }, ao])],
            [gpu.blur, group('SSAO/BlurBindGroup', gpu.blurBGL, [params, depthView, normalView, ao, blurred])],
            [gpu.composite, group('SSAO/CompositeBindGroup', gpu.compositeBGL, [
                params, depthView, normalView, blurred, input.createView(), gbuffer.albedoTexture.createView(),
                gbuffer.emissiveTexture.createView(), { buffer: a.kind === 'sky' ? a.skyLighting : gpu.noSky }, output.createView(),
            ])],
        ];
        const pass = commandEncoder.beginComputePass({ label: 'SSAO', timestampWrites: gpuPass('SSAO') });
        for (const [pipeline, bindGroup] of groups) {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        }
        pass.end();
    }

    public resize(_width: number, _height: number, _gbuffer: GBuffer): void {}

    public destroy(): void {
        const gpu = this.gpu;
        if (!gpu) return;
        gpu.params.destroy();
        gpu.kernel.destroy();
        gpu.noSky.destroy();
        if (gpu.targets) for (const t of gpu.targets.ao) t.destroy();
        this.gpu = null;
        this.initialized = false;
    }
}

export { SSAOEffect };
