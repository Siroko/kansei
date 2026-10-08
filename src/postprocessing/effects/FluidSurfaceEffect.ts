import compositeWgsl from '../../../rust/kansei-core/src/shaders/fluid_surface_composite.wgsl?raw';
import surfaceMeshWgsl from '../../../rust/kansei-core/src/shaders/fluid_surface_mesh.wgsl?raw';
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { Geometry } from '../../buffers/Geometry';
import { Material } from '../../materials/Material';
import { Renderable } from '../../objects/Renderable';
import { Vector4 } from '../../math/Vector4';
import type { FluidDensityField } from '../../simulations/fluid/FluidDensityField';
import type { FluidMarchingCubes } from '../../simulations/fluid/FluidMarchingCubes';
import { gpuPass } from '../../profiling/Profiler';

/**
 * The composite (Rust: `fluid_surface.rs`'s `COMPOSITE_SHADER`): group 0 holds the params (0),
 * the input (1), the GBuffer's background (2), normal (3) and emissive (5), and the output (4).
 */
const FLUID_SURFACE_COMPOSITE_WGSL: string = compositeWgsl;

/**
 * The fluid surface's GBuffer material (Rust: `SURFACE_MESH_WGSL`): `color` at group 0 binding 0;
 * it writes the colour, the world normal unencoded and the emissive-alpha mark.
 */
const FLUID_SURFACE_MESH_WGSL: string = surfaceMeshWgsl;

/** Bytes of the composite's `Params`. */
const PARAMS_BYTES = 176;

/** How `FluidSurfaceEffect` finds the fluid's surface in the GBuffer. */
enum FluidMask {
    /** Any pixel with a normal: for scenes where only the fluid writes the GBuffer's normals. */
    AnyNormal = 0,
    /**
     * Pixels with a normal whose emissive alpha is at least 0.5, which the fluid's surface
     * material writes (and other materials leave at 0): for scenes whose other materials write
     * normals too.
     */
    EmissiveAlpha = 1,
}

export interface FluidSurfaceOptions {
    ior?: number;
    chromaticAberration?: number;
    tintStrength?: number;
    fresnelPower?: number;
    roughness?: number;
    thickness?: number;
    color?: [number, number, number, number];
    /**
     * Key light: the direction the light travels (`DirectionalLight` convention), its
     * intensity, and colour. Drives the specular highlight and tints the rim.
     */
    lightDirection?: [number, number, number];
    lightIntensity?: number;
    lightColor?: [number, number, number];
    /** Strength of the rim glow at grazing angles, tinted by the key light. */
    rim?: number;
    /**
     * The sky's colour, reflected where the reflected view ray points up (open water under the
     * sky), mixed in by `skyReflection` (0: screen-space reflection only).
     */
    skyColor?: [number, number, number];
    skyReflection?: number;
    /** How the composite finds the fluid's pixels (`FluidMask.AnyNormal` by default). */
    mask?: FluidMask;
    /**
     * Where the surface comes from: given, the effect extracts it from the field each frame (as
     * the Rust effect does from the simulation it owns) and `surfaceRenderable` draws it.
     * Without it the caller updates the field and the marching cubes, and draws the mesh.
     */
    source?: FluidSurfaceSource;
}

/** The density field and marching cubes a `FluidSurfaceEffect` extracts the surface with. */
export interface FluidSurfaceSource {
    densityField: FluidDensityField;
    marchingCubes: FluidMarchingCubes;
    /** `marchingCubes.createBindGroup(densityField)`; made from them when unset. */
    extractBindGroup?: GPUBindGroup;
}

/**
 * The fluid surface: a screen-space refraction composite of the marching-cubes mesh drawn into
 * the GBuffer, after the background is copied (the mesh's material is `transmissive`).
 * Port of the Rust engine's `FluidSurfaceEffect` (`postprocessing/effects/fluid_surface.rs`),
 * whose WGSL it imports: refraction with chromatic aberration, Fresnel, screen-space and sky
 * reflection, a key light's GGX specular and a rim. A refracted sample that lands on a pixel the
 * fluid does not cover (something in front of the surface) is replaced by the pixel's own.
 *
 * With a `source`, it also extracts the surface (density field, then marching cubes) in its
 * render, as the Rust effect does, and `surfaceRenderable` puts it into the scene. Was
 * `FluidTransmissionEffect` (the raymarch that had this name is `FluidRaymarchEffect`).
 */
class FluidSurfaceEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;

    public ior: number;
    public chromaticAberration: number;
    public tintStrength: number;
    public fresnelPower: number;
    public roughness: number;
    public thickness: number;
    public color: [number, number, number, number];
    public lightDirection: [number, number, number];
    public lightIntensity: number;
    public lightColor: [number, number, number];
    public rim: number;
    public skyColor: [number, number, number];
    public skyReflection: number;
    public mask: FluidMask;

    /**
     * Whether the surface is extracted from the particles each frame (the default, with a
     * `source`). Off, the last surface extracted keeps drawing: for a simulation that is not
     * stepping.
     */
    public extract: boolean = true;
    /**
     * Whether the effect runs at all (the default): off, it costs nothing and composites
     * nothing, for a fluid out of view.
     */
    public active: boolean = true;

    public readonly source: FluidSurfaceSource | null;
    private _extractBindGroup: GPUBindGroup | null = null;

    private _pipeline: GPUComputePipeline | null = null;
    private _bgl: GPUBindGroupLayout | null = null;
    private _bg: GPUBindGroup | null = null;
    private _paramsBuffer: GPUBuffer | null = null;
    private readonly _params = new ArrayBuffer(PARAMS_BYTES);

    /** The textures `_bg` was made with. */
    private _bound: GPUTexture[] = [];

    constructor(options: FluidSurfaceOptions = {}) {
        super();
        this.ior = options.ior ?? 1.41;
        this.chromaticAberration = options.chromaticAberration ?? 0.05;
        this.tintStrength = options.tintStrength ?? 0.3;
        this.fresnelPower = options.fresnelPower ?? 2.3;
        this.roughness = options.roughness ?? 0.28;
        this.thickness = options.thickness ?? 2.4;
        this.color = options.color ?? [0.77, 0.96, 1.0, 1.0];
        this.lightDirection = options.lightDirection ?? [0.3, -1.0, 0.5];
        this.lightIntensity = options.lightIntensity ?? 2.0;
        this.lightColor = options.lightColor ?? [1, 1, 1];
        this.rim = options.rim ?? 0.15;
        this.skyColor = options.skyColor ?? [1, 1, 1];
        this.skyReflection = options.skyReflection ?? 0;
        this.mask = options.mask ?? FluidMask.AnyNormal;
        this.source = options.source ?? null;
    }

    /**
     * The renderable that puts the fluid's surface (the marching-cubes mesh of the `source`)
     * into the GBuffer, which the composite then refracts: add it to the scene. It writes
     * `color`, the world normal (as the composite reads it, unencoded) and the
     * `FluidMask.EmissiveAlpha` mark, double-sided, casting no shadow, drawn after the
     * background copy (`transmissive`).
     */
    public surfaceRenderable(color: [number, number, number, number]): Renderable {
        if (!this.source) throw new Error('FluidSurfaceEffect.surfaceRenderable needs a source');
        const mc = this.source.marchingCubes;
        const geometry = Geometry.fromGpuBuffers(mc.vertexBuffer, mc.indexBuffer, mc.indirectArgsBuffer, 'uint32');
        const material = new Material(FLUID_SURFACE_MESH_WGSL, {
            bindings: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, value: new Vector4(...color) }],
            cullMode: 'none',
            transmissive: true,
        });
        const renderable = new Renderable(geometry, material);
        renderable.castShadow = false;
        return renderable;
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        this._paramsBuffer = device.createBuffer({
            label: 'FluidSurface/Params',
            size: PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const texture = { sampleType: 'unfilterable-float' as GPUTextureSampleType };
        this._bgl = device.createBindGroupLayout({
            label: 'FluidSurface/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture },
            ],
        });
        const module = device.createShaderModule({ label: 'FluidSurface/Composite', code: FLUID_SURFACE_COMPOSITE_WGSL });
        this._pipeline = device.createComputePipeline({
            label: 'FluidSurface/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
            compute: { module, entryPoint: 'main' },
        });
        this.initialized = true;
    }

    isActive(): boolean {
        return this.active;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        _depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
        _emissive?: GPUTexture,
        gbuffer?: GBuffer,
    ): void {
        if (!this._pipeline || !this._device || !gbuffer) return;
        const device = this._device;

        // 1. the surface: density field, then marching cubes (drawn from the next frame on)
        if (this.source && this.extract) {
            const { densityField, marchingCubes } = this.source;
            this._extractBindGroup ??= this.source.extractBindGroup ?? marchingCubes.createBindGroup(densityField);
            densityField.update(commandEncoder);
            marchingCubes.update(this._extractBindGroup, densityField, commandEncoder);
        }

        // 2. the params
        const f = new Float32Array(this._params);
        const u = new Uint32Array(this._params);
        f.set(camera.viewMatrix.internalMat4 as unknown as Float32Array, 0);
        f.set(this.color, 16);
        f[20] = this.ior;
        f[21] = this.chromaticAberration;
        f[22] = this.tintStrength;
        f[23] = this.fresnelPower;
        f[24] = this.roughness;
        f[25] = this.thickness;
        f[26] = width;
        f[27] = height;
        f.set([...this.lightDirection, this.lightIntensity], 28);
        f.set([...this.lightColor, this.rim], 32);
        f.set([...this.skyColor, this.skyReflection], 36);
        u[40] = this.mask;
        device.queue.writeBuffer(this._paramsBuffer!, 0, this._params);

        // 3. the bind group, when a texture changed (the ping-pong, a resize)
        const textures = [input, output, gbuffer.backgroundTexture, gbuffer.normalTexture, gbuffer.emissiveTexture];
        if (!this._bg || textures.some((t, k) => t !== this._bound[k])) {
            this._bg = device.createBindGroup({
                label: 'FluidSurface/BG',
                layout: this._bgl!,
                entries: [
                    { binding: 0, resource: { buffer: this._paramsBuffer! } },
                    { binding: 1, resource: input.createView() },
                    { binding: 2, resource: gbuffer.backgroundTexture.createView() },
                    { binding: 3, resource: gbuffer.normalTexture.createView() },
                    { binding: 4, resource: output.createView() },
                    { binding: 5, resource: gbuffer.emissiveTexture.createView() },
                ],
            });
            this._bound = textures;
        }

        // 4. the composite
        const pass = commandEncoder.beginComputePass({ label: 'FluidSurface/Composite', timestampWrites: gpuPass('FluidSurface/Composite') });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bg);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    resize(_width: number, _height: number, _gbuffer: GBuffer): void {
        this._bg = null;
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._paramsBuffer = null;
        this._bg = null;
        this.initialized = false;
    }
}

export { FluidSurfaceEffect, FluidMask, FLUID_SURFACE_COMPOSITE_WGSL, FLUID_SURFACE_MESH_WGSL };
