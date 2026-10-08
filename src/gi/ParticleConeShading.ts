import { ComputeBuffer } from '../buffers/ComputeBuffer';
import { Sampler } from '../buffers/Sampler';
import { Texture } from '../buffers/Texture';
import { BindingLayouts } from '../materials/Binding';
import { Compute } from '../materials/Compute';
import { PARTICLE_CONES_WGSL } from './GiWGSL';
import { computeBinding, GiBufferSource, giBuffer, normalizeOrUp } from './ParticleVoxelizer';
import { Vec3, VoxelVolume } from './VoxelVolume';

/** Bytes per particle of `ParticleConeShading`'s lighting buffer: two vec4. */
export const PARTICLE_LIGHTING_STRIDE = 32;

/** How particles gather light from the volume (`ParticleConeShading.encode`). Rust: `gi::ParticleConeSettings`. */
export interface ParticleConeSettings {
    /** Direction toward the sun: the narrow cone's axis. */
    toSun: Vec3;
    /** tan of the sun cone's half aperture: the softness of the particles' shadow. */
    sunConeTan: number;
    /** tan of the six diffuse cones' half aperture (1: 90-degree cones tiling the sphere). */
    diffuseConeTan: number;
    /** Voxels out the cones start, past the particle's own splat (1 to 2). */
    startVoxels: number;
    /** How far the cones look, metres. */
    maxDistanceM: number;
    /** Weight of this frame in each particle's running average (1: no history). */
    temporalBlend: number;
    /** Voxels the cones' start moves by, per particle and frame (the average smooths it). */
    jitterVoxels: number;
    /** The most steps a cone takes (`VoxelGiQuality.coneSteps`). */
    maxSteps: number;
    /**
     * false: trace no cones and give every particle the whole sky and the sun (voxel GI off; the
     * volume is not read, so it need not be built).
     */
    useVolume: boolean;
    /**
     * The sun's visibility from the volume's distance field (`ParticleConeShading.setSdf`) instead
     * of the sun cone: a sharper soft shadow. Ignored without a field.
     */
    sdfSun: boolean;
    /** The field's soft shadow: Quilez's k, higher is harder. */
    sdfSunHardness: number;
}

export function defaultParticleConeSettings(): ParticleConeSettings {
    return {
        toSun: [0, 1, 0],
        sunConeTan: 0.05,
        diffuseConeTan: 1,
        startVoxels: 1.5,
        maxDistanceM: 1e4,
        temporalBlend: 0.2,
        jitterVoxels: 0.5,
        maxSteps: 48,
        useVolume: true,
        sdfSun: false,
        sdfSunHardness: 6,
    };
}

/** Bytes of the WGSL `ConeParams` (particle_cones.wgsl). */
const CONE_PARAMS_BYTES = 64;

/** Floats of the WGSL `SkyLighting` (`SKY_LIGHTING_WGSL`): 15 vec4. */
export const SKY_LIGHTING_FLOATS = 60;

/**
 * A `SkyLighting` whose radiance runs from `down` (straight below) to `up` (straight above),
 * linearly in the direction's height, with no sun or moon: the sky for cones that leave a volume
 * when the scene has no atmosphere. A constant sky is `up == down`. Rust: `gi::gradient_sky_lighting`.
 */
export function gradientSkyLighting(up: Vec3, down: Vec3): Float32Array {
    const sky = new Float32Array(SKY_LIGHTING_FLOATS);
    for (let c = 0; c < 3; c++) {
        // skyRadiance(d) = 0.282095 sh0 + 0.488603 sh1 d.y
        sky[c] = (up[c] + down[c]) * 0.5 / 0.282095;
        sky[4 + c] = (up[c] - down[c]) * 0.5 / 0.488603;
    }
    return sky;
}

/**
 * Per-particle light from cone tracing a `VoxelVolume`, miaumiau.cat/?p=1476's gather: each
 * particle traces six 90-degree cones along the axes (escaping to the sky) and one narrow cone
 * toward the sun, and keeps a running average, since particle indices are stable. It writes
 * `PARTICLE_LIGHTING_STRIDE` bytes per particle, for the particles' material to read as instance
 * attributes (`lightingInstanceBuffer`). Rust: `gi::ParticleConeShading`.
 * - `vec4(mean incoming radiance, sun visibility)`: a Lambertian particle of albedo `k` reflects
 *   `k * rgb`, plus its sun light times `a`;
 * - `vec4(the particle's emission, 0)`, as the splat put it in the volume.
 */
export class ParticleConeShading {
    /** `PARTICLE_LIGHTING_STRIDE` bytes per particle, in the particles' order. */
    public readonly lightingBuffer: GPUBuffer;
    public readonly capacity: number;

    private readonly params: GPUBuffer;
    private shade!: Compute;
    private frame = 0;
    private sdf?: Texture;
    private readonly noSdf: Texture;
    private readonly sampler: Sampler;

    /**
     * Shade up to `capacity` particles of `positions` (and `velocities`, for speed emission;
     * `emission` is the voxelizer's `ParticleEmission` uniform) from `volume`, escaping to `sky` (a
     * `SkyLighting` uniform).
     */
    constructor(
        private readonly device: GPUDevice,
        volume: VoxelVolume,
        positions: GiBufferSource,
        velocities: GiBufferSource | undefined,
        emission: GPUBuffer,
        sky: GPUBuffer,
        capacity: number,
    ) {
        this.capacity = capacity;
        this.params = device.createBuffer({
            label: 'VoxelGI/ConeParams',
            size: CONE_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.lightingBuffer = device.createBuffer({
            label: 'VoxelGI/ParticleLighting',
            size: Math.max(capacity, 1) * PARTICLE_LIGHTING_STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
        });
        // a 1-texel stand-in for the distance field
        this.noSdf = Texture.fromView('VoxelGI/NoSdf', device.createTexture({
            label: 'VoxelGI/NoSdf',
            size: { width: 1, height: 1, depthOrArrayLayers: 1 },
            dimension: '3d',
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        }), undefined, '3d');
        // the volume's own sampler, as a bindable
        this.sampler = new Sampler('linear', 'linear', 'clamp-to-edge', 1, { mipmapFilter: 'linear' });
        this.bind(volume, positions, velocities, emission, sky);
    }

    /** Rebind the inputs, for example another sky. */
    public bind(volume: VoxelVolume, positions: GiBufferSource, velocities: GiBufferSource | undefined, emission: GPUBuffer, sky: GPUBuffer): void {
        const positionsBuffer = giBuffer(this.device, positions);
        const compute = GPUShaderStage.COMPUTE;
        this.shade = new Compute(PARTICLE_CONES_WGSL, [
            computeBinding(0, volume.uniform, 'uniform'),
            computeBinding(1, this.params, 'uniform'),
            computeBinding(2, emission, 'uniform'),
            computeBinding(3, sky, 'uniform'),
            { binding: 4, visibility: compute, value: volume.asTexture(), layout: BindingLayouts.texture3d() },
            { binding: 5, visibility: compute, value: this.sampler },
            computeBinding(6, positionsBuffer, 'read-only-storage'),
            computeBinding(7, velocities ? giBuffer(this.device, velocities) : positionsBuffer, 'read-only-storage'),
            computeBinding(8, this.lightingBuffer, 'storage'),
            { binding: 9, visibility: compute, value: this.sdf ?? this.noSdf, layout: BindingLayouts.texture3d() },
        ]);
        this.shade.initialize(this.device);
    }

    /**
     * Read `sdf` (a distance field over the same volume, as a 3D `Texture`) for
     * `ParticleConeSettings.sdfSun`, from the next `bind`.
     */
    public setSdf(sdf: Texture | undefined): void {
        this.sdf = sdf;
    }

    /**
     * The lighting buffer as instance attributes for `InstancedGeometry`: the incoming light and
     * sun visibility at `shaderLocation`, the emission at `shaderLocation + 1`.
     */
    public lightingInstanceBuffer(shaderLocation: number): ComputeBuffer {
        const buffer = ComputeBuffer.fromExternal(this.lightingBuffer, 'storage');
        buffer.stride = PARTICLE_LIGHTING_STRIDE;
        buffer.attributes = [
            { shaderLocation, offset: 0, format: 'float32x4' },
            { shaderLocation: shaderLocation + 1, offset: 16, format: 'float32x4' },
        ];
        return buffer;
    }

    /** Start the running average over: the next frame takes its light as it is. */
    public resetHistory(): void {
        this.frame = 0;
    }

    /** Record the shading of the first `particleCount` particles (after the volume's mips are built). */
    public encode(encoder: GPUCommandEncoder, particleCount: number, settings: ParticleConeSettings): void {
        particleCount = Math.min(particleCount, this.capacity);
        const data = new ArrayBuffer(CONE_PARAMS_BYTES);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        f32.set(normalizeOrUp(settings.toSun), 0);
        f32[3] = settings.sunConeTan;
        f32[4] = settings.diffuseConeTan;
        f32[5] = settings.startVoxels;
        f32[6] = settings.maxDistanceM;
        f32[7] = Math.min(Math.max(settings.temporalBlend, 0), 1);
        u32[8] = particleCount;
        u32[9] = settings.maxSteps;
        u32[10] = this.frame;
        f32[11] = settings.jitterVoxels;
        u32[12] = settings.useVolume ? 1 : 0;
        u32[13] = settings.sdfSun && this.sdf ? 1 : 0;
        f32[14] = Math.max(settings.sdfSunHardness, 0.1);
        this.device.queue.writeBuffer(this.params, 0, data);
        this.frame = Math.max((this.frame + 1) >>> 0, 1);
        if (particleCount === 0) return;
        const pass = encoder.beginComputePass({ label: 'VoxelGI/ParticleCones' });
        pass.setPipeline(this.shade.pipeline!);
        pass.setBindGroup(0, this.shade.getBindGroup(this.device));
        pass.dispatchWorkgroups(Math.ceil(particleCount / 64));
        pass.end();
    }
}
