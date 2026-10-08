import { ComputeBuffer } from '../buffers/ComputeBuffer';
import { Texture } from '../buffers/Texture';
import { BindingLayouts } from '../materials/Binding';
import { Compute } from '../materials/Compute';
import { RESOLVE_WGSL, SPLAT_WGSL } from './GiWGSL';
import { ACCUMULATOR_BYTES_PER_VOXEL, Vec3, VoxelVolume } from './VoxelVolume';

/** Most analytic boxes a `ParticleVoxelizer` holds. */
export const MAX_GI_BOXES = 32;

/**
 * The light particles emit (WGSL `ParticleEmission`): a share of them, picked by a hash of their
 * index, glows with `color`, and each adds `speedColor` per unit of speed. Shared by the splat,
 * which puts it in the volume, and the cone shading, which hands it to the particles' material,
 * so both agree on which particles glow. Rust: `gi::ParticleEmission`.
 */
export interface ParticleEmission {
    /** Scene radiance of a glowing particle. */
    color: Vec3;
    /** The share of particles that glow, 0..1. */
    share: number;
    /** Scene radiance added per unit of speed (needs the particles' velocities). */
    speedColor: Vec3;
    seed: number;
}

export function defaultParticleEmission(): ParticleEmission {
    return { color: [0, 0, 0], share: 0, speedColor: [0, 0, 0], seed: 0x9e3779b9 };
}

/**
 * An analytic box in the volume (WGSL `GiBox`): a wall, a container, a collider. It covers its
 * exact share of each voxel it overlaps and reflects the sun (`N.L`, no shadow) and the sky on
 * the face `normal` points out of, plus its own emission (scene radiance). Make it at least a
 * voxel thick, or wide cones see through it at coarse mips. Rust: `gi::GiBox`.
 */
export interface GiBox {
    min: Vec3;
    max: Vec3;
    albedo: Vec3;
    /** The lit face's outward normal. */
    normal: Vec3;
    /** Scene radiance (none by default). */
    emission?: Vec3;
}

/** Bytes of one WGSL `GiBox`: five vec4. */
const GI_BOX_BYTES = 80;

/** How particles fill the volume, and how the boxes are lit (`ParticleVoxelizer.encode`). Rust: `gi::ParticleSplatSettings`. */
export interface ParticleSplatSettings {
    /** Density one particle adds (spread over the voxels it touches). */
    densityPerParticle: number;
    /** Opacity per unit density across a voxel: `1 - exp(-extinction * density)`. */
    extinction: number;
    /**
     * The particle's radius, metres: under a voxel the splat is trilinear (8 voxels), above it
     * spreads over the footprint (up to 3 voxels out).
     */
    particleRadiusM: number;
    /** Direction toward the sun, and its illuminance (scene units), for the boxes. */
    toSun: Vec3;
    sunIlluminance: Vec3;
    /** How much of the sky's irradiance reaches the boxes. */
    boxSkyScale: number;
}

export function defaultParticleSplatSettings(): ParticleSplatSettings {
    return {
        densityPerParticle: 1,
        extinction: 0.5,
        particleRadiusM: 0,
        toSun: [0, 1, 0],
        sunIlluminance: [3, 3, 3],
        boxSkyScale: 1,
    };
}

/** Bytes of the WGSL `SplatParams`, `ResolveParams` and `ParticleEmission`. */
const SPLAT_PARAMS_BYTES = 16;
const RESOLVE_PARAMS_BYTES = 48;
export const PARTICLE_EMISSION_BYTES = 32;

/** A buffer the GI reads: a `GPUBuffer`, or a `ComputeBuffer` (created on the device if it is not yet). */
export type GiBufferSource = GPUBuffer | ComputeBuffer;

/** The `GPUBuffer` behind `source`. */
export function giBuffer(device: GPUDevice, source: GiBufferSource): GPUBuffer {
    if (source instanceof ComputeBuffer) {
        if (!source.initialized) source.initialize(device);
        return source.gpuBuffer!;
    }
    return source;
}

/** `normalize(v)`, or +y for a zero vector (glam's `normalize_or(Vec3::Y)`). */
export function normalizeOrUp(v: Vec3): Vec3 {
    const length = Math.hypot(v[0], v[1], v[2]);
    return length > 0 && Number.isFinite(length) ? [v[0] / length, v[1] / length, v[2] / length] : [0, 1, 0];
}

/** `buffer` bound as `type` (`uniform`, `storage` or `read-only-storage`) in a `Compute`. */
export function computeBinding(binding: number, buffer: GPUBuffer, type: string) {
    return { binding, visibility: GPUShaderStage.COMPUTE, value: ComputeBuffer.fromExternal(buffer, type) };
}

/**
 * Particles and analytic boxes into a `VoxelVolume`'s mip 0, miaumiau.cat/?p=1476's scatter done
 * with atomics. Rust: `gi::ParticleVoxelizer`.
 * - **splat**: one thread per particle adds its density and density-weighted emission into four
 *   u32 accumulators per voxel, trilinearly (or over a smooth footprint for large particles), with
 *   weights that sum to one, so moving a particle changes the volume continuously;
 * - **resolve**: turns density into opacity (Beer-Lambert), emission into its mean times that
 *   opacity, adds the boxes and clears the accumulators.
 */
export class ParticleVoxelizer {
    /**
     * Four u32 per voxel, x fastest: emission r, g, b (fixed point, 1/1024, of stored radiance
     * times density) and density (fixed point, 1/4096). Zero outside a splat-to-resolve span.
     */
    public readonly accumulators: GPUBuffer;
    /** The `ParticleEmission` uniform, for passes that must agree with the splat on it. */
    public readonly emissionBuffer: GPUBuffer;

    private readonly splatParams: GPUBuffer;
    private readonly resolveParams: GPUBuffer;
    private readonly boxes: GPUBuffer;
    private boxCount = 0;
    private readonly hasVelocities: boolean;
    private readonly splat: Compute;
    private resolve!: Compute;
    private readonly dims: Vec3;
    private readonly voxelSize: number;

    /**
     * Particles from `positions` (`array<vec4f>`, xyz in metres) and optionally `velocities` (for
     * speed emission), into `volume`; `sky` is a `SkyLighting` uniform for the boxes.
     */
    constructor(private readonly device: GPUDevice, private readonly volume: VoxelVolume, positions: GiBufferSource, velocities: GiBufferSource | undefined, sky: GPUBuffer) {
        const uniform = (label: string, size: number) =>
            device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.accumulators = device.createBuffer({
            label: 'VoxelGI/Accumulators',
            size: volume.layout.voxelCount() * ACCUMULATOR_BYTES_PER_VOXEL,
            // (COPY_SRC: readable back for checks)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.splatParams = uniform('VoxelGI/SplatParams', SPLAT_PARAMS_BYTES);
        this.resolveParams = uniform('VoxelGI/ResolveParams', RESOLVE_PARAMS_BYTES);
        this.emissionBuffer = uniform('VoxelGI/ParticleEmission', PARTICLE_EMISSION_BYTES);
        this.boxes = device.createBuffer({
            label: 'VoxelGI/Boxes',
            size: MAX_GI_BOXES * GI_BOX_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const positionsBuffer = giBuffer(device, positions);
        this.splat = new Compute(SPLAT_WGSL, [
            computeBinding(0, volume.uniform, 'uniform'),
            computeBinding(1, this.splatParams, 'uniform'),
            computeBinding(2, this.emissionBuffer, 'uniform'),
            computeBinding(3, positionsBuffer, 'read-only-storage'),
            computeBinding(4, velocities ? giBuffer(device, velocities) : positionsBuffer, 'read-only-storage'),
            computeBinding(5, this.accumulators, 'storage'),
        ]);
        this.splat.initialize(device);
        this.hasVelocities = velocities !== undefined;
        this.dims = volume.dims;
        this.voxelSize = volume.voxelSize;
        this.setSky(sky);
    }

    /** Read the sky's light for the boxes from `sky` (a `SkyLighting` uniform). */
    public setSky(sky: GPUBuffer): void {
        this.resolve = new Compute(RESOLVE_WGSL, [
            computeBinding(0, this.volume.uniform, 'uniform'),
            computeBinding(1, this.resolveParams, 'uniform'),
            computeBinding(2, sky, 'uniform'),
            computeBinding(3, this.accumulators, 'storage'),
            computeBinding(4, this.boxes, 'read-only-storage'),
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                value: Texture.fromView('VoxelVolume/Mip0', this.volume.gpuTexture, this.volume.mip0StorageView, '3d'),
                layout: BindingLayouts.storageTexture3d('rgba16float'),
            },
        ]);
        this.resolve.initialize(this.device);
    }

    /** Replace the analytic boxes (at most `MAX_GI_BOXES`; the rest are dropped). */
    public setBoxes(boxes: GiBox[]): void {
        const kept = boxes.slice(0, MAX_GI_BOXES);
        if (kept.length > 0) {
            const data = new Float32Array(kept.length * GI_BOX_BYTES / 4);
            kept.forEach((b, i) => {
                const o = i * GI_BOX_BYTES / 4;
                data.set(b.min, o);
                data.set(b.max, o + 4);
                data.set(b.albedo, o + 8);
                data.set(b.emission ?? [0, 0, 0], o + 12);
                data.set(b.normal, o + 16);
            });
            this.device.queue.writeBuffer(this.boxes, 0, data);
        }
        this.boxCount = kept.length;
    }

    /** Set what the particles emit (speed emission is ignored without velocities). */
    public setEmission(emission: ParticleEmission): void {
        const data = new ArrayBuffer(PARTICLE_EMISSION_BYTES);
        const f32 = new Float32Array(data);
        f32.set(emission.color, 0);
        f32[3] = emission.share;
        f32.set(this.hasVelocities ? emission.speedColor : [0, 0, 0], 4);
        new Uint32Array(data)[7] = emission.seed >>> 0;
        this.device.queue.writeBuffer(this.emissionBuffer, 0, data);
    }

    private writeParams(particleCount: number, settings: ParticleSplatSettings): void {
        const splat = new ArrayBuffer(SPLAT_PARAMS_BYTES);
        new Uint32Array(splat)[0] = particleCount;
        new Float32Array(splat).set([settings.particleRadiusM / this.voxelSize, settings.densityPerParticle], 1);
        this.device.queue.writeBuffer(this.splatParams, 0, splat);
        const resolve = new ArrayBuffer(RESOLVE_PARAMS_BYTES);
        const f32 = new Float32Array(resolve);
        f32.set(normalizeOrUp(settings.toSun), 0);
        f32[3] = settings.extinction;
        f32.set(settings.sunIlluminance, 4);
        new Uint32Array(resolve)[7] = this.boxCount;
        f32[8] = settings.boxSkyScale;
        this.device.queue.writeBuffer(this.resolveParams, 0, resolve);
    }

    /** Record the splat of `particleCount` particles alone (the accumulators then hold them until `encodeResolve`). */
    public encodeSplat(encoder: GPUCommandEncoder, particleCount: number, settings: ParticleSplatSettings): void {
        this.writeParams(particleCount, settings);
        if (particleCount === 0) return;
        const pass = encoder.beginComputePass({ label: 'VoxelGI/Splat' });
        pass.setPipeline(this.splat.pipeline!);
        pass.setBindGroup(0, this.splat.getBindGroup(this.device));
        pass.dispatchWorkgroups(Math.ceil(particleCount / 64));
        pass.end();
    }

    /** Record the resolve into the volume's mip 0 (clearing the accumulators). */
    public encodeResolve(encoder: GPUCommandEncoder): void {
        const [w, h, d] = this.dims;
        const pass = encoder.beginComputePass({ label: 'VoxelGI/Resolve' });
        pass.setPipeline(this.resolve.pipeline!);
        pass.setBindGroup(0, this.resolve.getBindGroup(this.device));
        pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        pass.end();
    }

    /**
     * Record the splat and the resolve: the volume's mip 0 then holds this frame's particles and
     * boxes (build its mips next).
     */
    public encode(encoder: GPUCommandEncoder, particleCount: number, settings: ParticleSplatSettings): void {
        this.encodeSplat(encoder, particleCount, settings);
        this.encodeResolve(encoder);
    }
}
