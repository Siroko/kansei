import { ComputeBuffer } from '../buffers/ComputeBuffer';
import { Renderer } from '../renderers/Renderer';
import { defaultParticleConeSettings, gradientSkyLighting, ParticleConeSettings, ParticleConeShading } from './ParticleConeShading';
import {
    defaultParticleEmission,
    defaultParticleSplatSettings,
    GiBox,
    GiBufferSource,
    giBuffer,
    ParticleEmission,
    ParticleSplatSettings,
    ParticleVoxelizer,
} from './ParticleVoxelizer';
import { Vec3, VoxelGiQuality, VoxelVolume } from './VoxelVolume';
import { JumpFloodSdf } from './JumpFloodSdf';

/** What `ParticleGi` builds. Rust: `gi::ParticleGiOptions`. */
export interface ParticleGiOptions {
    /**
     * The tier asked for; `ParticleGi` steps down from it to what the device holds within
     * `budgetBytes` (see `VoxelGiQuality.fit`). Default `medium`.
     */
    quality?: VoxelGiQuality;
    /** The box the volume covers (the particles' container and its walls), metres. */
    boundsMin: Vec3;
    boundsMax: Vec3;
    /** Most particles shaded (the lighting buffer's size). */
    capacity: number;
    /** The reference radiance is stored against (see `VoxelVolume`). Default 1. */
    radianceScale?: number;
    /** Most bytes the volume may take (0, the default: no limit). 24 MiB keeps a phone at `low`. */
    budgetBytes?: number;
}

/** Everything `ParticleGi` reads each frame; change it freely between frames. Rust: `gi::ParticleGiSettings`. */
export class ParticleGiSettings {
    public splat: ParticleSplatSettings = defaultParticleSplatSettings();
    public cones: ParticleConeSettings = defaultParticleConeSettings();
    public emission: ParticleEmission = defaultParticleEmission();

    /** Point the sun (direction toward it, and its illuminance) for both the boxes and the cones. */
    public setSun(toSun: Vec3, illuminance: Vec3): void {
        this.splat.toSun = toSun;
        this.splat.sunIlluminance = illuminance;
        this.cones.toSun = toSun;
    }
}

/**
 * Particles that light, occlude and shadow each other through a voxel volume (miaumiau.cat's
 * "indirect lighting on particles", p=1476, on WebGPU compute). Each frame, in one encoder:
 * 1. the particles splat their density and emission into the volume, and analytic boxes (the
 *    room's walls) add their lit surfaces (`ParticleVoxelizer`);
 * 2. the volume's mips are rebuilt (`Mip3d`);
 * 3. each particle cone traces its incoming light and its sun visibility (`ParticleConeShading`),
 *    which its material reads from `lightingInstanceBuffer`.
 *
 * Rust: `gi::ParticleGi`.
 *
 * ```ts
 * const gi = new ParticleGi(renderer, { boundsMin, boundsMax, capacity }, sim.positionsBufferRef, sim.velocitiesBufferRef);
 * gi.setBoxes(walls);
 * // new InstancedGeometry(billboard, capacity, [positions, gi.lightingInstanceBuffer(4)])
 * // each frame, after the simulation step:
 * const encoder = renderer.createCommandEncoder('GI');
 * gi.encode(encoder, particleCount);
 * renderer.submit(encoder.finish());
 * ```
 */
export class ParticleGi {
    public settings = new ParticleGiSettings();
    /** The tier in use (the one asked for, or lower if the device could not hold it). */
    public readonly quality: VoxelGiQuality;
    public readonly volume: VoxelVolume;
    public readonly voxelizer: ParticleVoxelizer;
    public readonly shading: ParticleConeShading;
    /** The sky buffer the gradient writes (a `SkyLighting` uniform). */
    public readonly skyBuffer: GPUBuffer;

    private readonly device: GPUDevice;
    private _sdf: JumpFloodSdf | null = null;

    /** Particles from `positions` (`array<vec4f>`) and optionally `velocities` (speed emission). */
    constructor(renderer: Renderer, options: ParticleGiOptions, positions: GiBufferSource, velocities?: GiBufferSource) {
        const device = renderer.gpuDevice;
        this.device = device;
        this.quality = VoxelGiQuality.fit(options.quality ?? 'medium', device.limits, options.boundsMin, options.boundsMax, options.budgetBytes ?? 0);
        this.volume = new VoxelVolume(device, options.boundsMin, options.boundsMax, VoxelGiQuality.resolution(this.quality), options.radianceScale ?? 1);
        this.skyBuffer = device.createBuffer({
            label: 'VoxelGI/Sky',
            size: gradientSkyLighting([0, 0, 0], [0, 0, 0]).byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setSkyGradient([0.4, 0.5, 0.7], [0.1, 0.1, 0.1]);
        // the particles' buffers as GPU buffers once, shared by both passes
        const positionsBuffer = giBuffer(device, positions);
        const velocitiesBuffer = velocities ? giBuffer(device, velocities) : undefined;
        this.voxelizer = new ParticleVoxelizer(device, this.volume, positionsBuffer, velocitiesBuffer, this.skyBuffer);
        this.shading = new ParticleConeShading(device, this.volume, positionsBuffer, velocitiesBuffer, this.voxelizer.emissionBuffer, this.skyBuffer, options.capacity);
        this.settings.cones.maxSteps = VoxelGiQuality.coneSteps(this.quality);
    }

    /** Replace the analytic boxes (walls, containers): see `GiBox`. */
    public setBoxes(boxes: GiBox[]): void {
        this.voxelizer.setBoxes(boxes);
    }

    /**
     * The sky past the volume, from `down` to `up` (scene radiance): see `gradientSkyLighting`.
     * Ignored after `useSkyLighting`.
     */
    public setSkyGradient(up: Vec3, down: Vec3): void {
        this.device.queue.writeBuffer(this.skyBuffer, 0, gradientSkyLighting(up, down));
    }

    /**
     * Take the sky from `skyLighting` (a `SkyLighting` uniform, such as an atmosphere's) instead of
     * the gradient. Give the particles' buffers again, as the constructor takes them.
     */
    public useSkyLighting(skyLighting: GPUBuffer, positions: GiBufferSource, velocities?: GiBufferSource): void {
        this.voxelizer.setSky(skyLighting);
        this.shading.bind(this.volume, positions, velocities, this.voxelizer.emissionBuffer, skyLighting);
    }

    /**
     * Keep a distance field of the volume's opaque voxels (the particles' body where it is at
     * least half opaque, and the boxes), flooded every frame after the splat, for the particles'
     * sun (`settings.cones.sdfSun`): a sharper soft shadow than the sun cone. Give the particles'
     * buffers and the sky again, as `useSkyLighting` takes them (`skyBuffer` unless it was
     * called). The field is `r32float`, which needs the device's `float32-filterable` (the
     * renderer requests it by default).
     */
    public enableSdf(positions: GiBufferSource, velocities: GiBufferSource | undefined, sky: GPUBuffer): void {
        if (!this._sdf) {
            if (!this.device.features.has('float32-filterable')) {
                throw new Error("ParticleGi.enableSdf needs the device's 'float32-filterable' (RendererOptions.requireFloat32Filterable)");
            }
            this._sdf = new JumpFloodSdf(this.device, this.volume.layout, { kind: 'opacity', radiance: this.volume.view, threshold: 0.5 });
            this.shading.setSdf(this._sdf.asTexture());
        }
        this.shading.bind(this.volume, positions, velocities, this.voxelizer.emissionBuffer, sky);
    }

    /** The distance field, if enabled. */
    public get sdf(): JumpFloodSdf | null {
        return this._sdf;
    }

    /** See `ParticleConeShading.lightingInstanceBuffer`. */
    public lightingInstanceBuffer(shaderLocation: number): ComputeBuffer {
        return this.shading.lightingInstanceBuffer(shaderLocation);
    }

    public get lightingBuffer(): GPUBuffer {
        return this.shading.lightingBuffer;
    }

    /** Start the particles' running averages over. */
    public resetHistory(): void {
        this.shading.resetHistory();
    }

    /**
     * Record a frame: splat and resolve, mips, then the particles' cones. With
     * `settings.cones.useVolume` false only the last runs, giving the particles the sky and the sun
     * unoccluded: voxel GI off at a fraction of the cost.
     */
    public encode(encoder: GPUCommandEncoder, particleCount: number): void {
        this.voxelizer.setEmission(this.settings.emission);
        if (this.settings.cones.useVolume) {
            this.voxelizer.encode(encoder, particleCount, this.settings.splat);
            if (this._sdf && this.settings.cones.sdfSun) this._sdf.encode(encoder);
            this.volume.buildMips(encoder);
        }
        this.shading.encode(encoder, particleCount, this.settings.cones);
    }
}
