import { Texture } from '../buffers/Texture';
import { MIP3D_WGSL } from './GiWGSL';
import { AnisotropicMips } from './AnisotropicMips';

/**
 * Resolution and cost tiers of voxel GI. Each sets the voxels across the volume's longest axis
 * and the steps a cone may take; `VoxelGiQuality.fit` steps down to a tier the device can hold,
 * so a phone degrades to `low` instead of failing. Rust: `gi::VoxelGiQuality`.
 * - `low`: 64 voxels across, about 7 MiB for a cube (radiance with mips and accumulators), within
 *   a phone's budget of 24 MiB;
 * - `medium`: 96 voxels across, about 21 MiB for a cube;
 * - `high`: 128 voxels across, about 50 MiB for a cube.
 */
export type VoxelGiQuality = 'low' | 'medium' | 'high';

export const VoxelGiQuality = {
    /** Voxels across the volume's longest axis. */
    resolution(q: VoxelGiQuality): number {
        return { low: 64, medium: 96, high: 128 }[q];
    },

    /** The most steps one cone takes (wide cones need far fewer: each step doubles in length). */
    coneSteps(q: VoxelGiQuality): number {
        return { low: 32, medium: 48, high: 64 }[q];
    },

    /** The tier below, if any. */
    lower(q: VoxelGiQuality): VoxelGiQuality | undefined {
        return ({ low: undefined, medium: 'low', high: 'medium' } as const)[q];
    },

    /** `low`, `medium` or `high` as a tier; undefined for any other name. */
    fromName(name: string | null | undefined): VoxelGiQuality | undefined {
        return name === 'low' || name === 'medium' || name === 'high' ? name : undefined;
    },

    /**
     * `q`, or the highest tier below it whose volume over `boundsMin..boundsMax` fits the device's
     * limits (its accumulators in one storage binding, its sides in a 3D texture) and
     * `budgetBytes` (0: no budget). `low` is returned when nothing fits.
     */
    fit(q: VoxelGiQuality, limits: GPUSupportedLimits, boundsMin: Vec3, boundsMax: Vec3, budgetBytes: number = 0): VoxelGiQuality {
        return fitWith(q, limits, boundsMin, boundsMax, budgetBytes, ACCUMULATOR_BYTES_PER_VOXEL, false);
    },
};

/** `VoxelGiQuality.fit` with `bytesPerVoxel` in one storage binding next to the radiance (and its anisotropic chains). */
function fitWith(q: VoxelGiQuality, limits: GPUSupportedLimits, boundsMin: Vec3, boundsMax: Vec3, budgetBytes: number, bytesPerVoxel: number, anisotropic: boolean): VoxelGiQuality {
    for (;;) {
        const layout = new VolumeLayout(boundsMin, boundsMax, VoxelGiQuality.resolution(q));
        const voxels = layout.voxelCount();
        const fits = voxels * bytesPerVoxel <= limits.maxStorageBufferBindingSize
            && layout.dims.every((d) => d <= limits.maxTextureDimension3D)
            && (budgetBytes === 0 || layout.radianceBytes() + (anisotropic ? layout.anisotropicBytes() : 0) + voxels * bytesPerVoxel <= budgetBytes);
        const lower = VoxelGiQuality.lower(q);
        if (fits || !lower) return q;
        q = lower;
    }
}

export type Vec3 = [number, number, number];

/** Bytes of the particle accumulators per voxel (four u32). */
export const ACCUMULATOR_BYTES_PER_VOXEL = 16;

/**
 * Where a volume's voxels lie: cubic voxels over a box, `resolution` across its longest axis,
 * each side rounded up to a multiple of 8 (three mips then halve exactly) and centred on the box.
 * Rust: `gi::VolumeLayout`.
 */
export class VolumeLayout {
    /** World position of voxel (0, 0, 0)'s corner. */
    public readonly origin: Vec3;
    public readonly voxelSize: number;
    public readonly dims: Vec3;

    constructor(boundsMin: Vec3, boundsMax: Vec3, resolution: number) {
        const extent = [0, 1, 2].map((i) => Math.max(Math.abs(boundsMax[i] - boundsMin[i]), 1e-3));
        // f32, as Rust computes it, so both engines place the same voxels
        this.voxelSize = Math.fround(Math.max(...extent) / Math.max(resolution, 1));
        this.dims = [0, 1, 2].map((i) => Math.ceil(Math.max(Math.ceil(extent[i] / this.voxelSize), 1) / 8) * 8) as Vec3;
        this.origin = [0, 1, 2].map((i) => (boundsMin[i] + boundsMax[i]) * 0.5 - this.dims[i] * this.voxelSize * 0.5) as Vec3;
    }

    public voxelCount(): number {
        return this.dims[0] * this.dims[1] * this.dims[2];
    }

    /** The mips of the volume's chain: down to one voxel on its longest side. */
    public mipCount(): number {
        return Texture.fullMipCount(this.dims[0], this.dims[1], this.dims[2]);
    }

    /** The radiance texture with all its mips plus the particle accumulators. */
    public memoryBytes(): number {
        return this.radianceBytes() + this.voxelCount() * ACCUMULATOR_BYTES_PER_VOXEL;
    }

    /** The six anisotropic chains (`VoxelVolume.setAnisotropicMips`): each as the volume's mips 1.. are. */
    public anisotropicBytes(): number {
        return 6 * (this.radianceBytes() - this.voxelCount() * 8);
    }

    /** The radiance texture with all its mips. */
    public radianceBytes(): number {
        let bytes = 0;
        for (let l = 0; l < this.mipCount(); l++) {
            bytes += this.dims.reduce((n, d) => n * Math.max(d >> l, 1), 1) * 8;
        }
        return bytes;
    }
}

/** Bytes of the WGSL `VoxelVolume` (voxel_volume.wgsl; Rust `VoxelVolumeGpu`). */
const VOXEL_VOLUME_BYTES = 48;

/**
 * A voxel volume of the scene's light, the core every voxel-GI producer writes and every
 * consumer reads (`VOXEL_VOLUME_WGSL`). Rust: `gi::VoxelVolume`.
 * - an `rgba16float` 3D texture whose rgb is the radiance leaving each voxel, premultiplied by
 *   its coverage, and a its opacity across one voxel, so mips are a plain 2x2x2 average
 *   (`Mip3d`), rebuilt by `buildMips` after mip 0 is written;
 * - its placement and scale as a uniform (`VoxelVolume` in WGSL);
 * - a trilinear, clamping sampler for cone tracing (`VOXEL_CONES_WGSL`).
 *
 * Radiance is stored divided by `radianceScale`, a reference such as the scene's exposure or its
 * sun's illuminance, so that sums of bright emitters keep to f16 and to the producers' fixed
 * point; readers multiply it back (`voxelConeTrace` does).
 */
export class VoxelVolume {
    public readonly layout: VolumeLayout;
    /** Every mip, to bind as `texture_3d<f32>` and cone trace with `sampler`. */
    public readonly view: GPUTextureView;
    /** Mip 0 alone, for producers to write as `texture_storage_3d<rgba16float, write>`. */
    public readonly mip0StorageView: GPUTextureView;
    /** The WGSL `VoxelVolume` uniform. */
    public readonly uniform: GPUBuffer;
    /** Trilinear across voxels and mips, clamping at the volume's sides. */
    public readonly sampler: GPUSampler;

    private readonly texture: Texture;
    private readonly mips: Mip3d;
    private anisotropic?: AnisotropicMips;
    private readonly gpu = new ArrayBuffer(VOXEL_VOLUME_BYTES);

    /** A volume over `boundsMin..boundsMax`, `resolution` voxels across its longest axis. */
    constructor(private readonly device: GPUDevice, boundsMin: Vec3, boundsMax: Vec3, resolution: number, radianceScale: number = 1) {
        this.layout = new VolumeLayout(boundsMin, boundsMax, resolution);
        const [w, h, d] = this.layout.dims;
        this.texture = Texture.new3D(
            'VoxelVolume/Radiance', w, h, d, 'rgba16float',
            GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
        ).withMipLevels(this.layout.mipCount());
        this.texture.initialize(device);
        this.view = this.texture.resource as GPUTextureView;
        this.mip0StorageView = this.texture.mipView(0)!;
        this.mips = new Mip3d(device, this.texture.gpuTexture!);

        const f32 = new Float32Array(this.gpu);
        const u32 = new Uint32Array(this.gpu);
        f32.set(this.layout.origin, 0);
        f32[3] = this.layout.voxelSize;
        u32.set(this.layout.dims, 4);
        u32[7] = this.layout.mipCount();
        f32.set(this.layout.dims.map((n) => 1 / (n * this.layout.voxelSize)), 8);
        f32[11] = Math.max(radianceScale, 1e-6);
        this.uniform = device.createBuffer({
            label: 'VoxelVolume/Params',
            size: VOXEL_VOLUME_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniform, 0, this.gpu);
        this.sampler = device.createSampler({
            label: 'VoxelVolume/LinearClamp',
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
        });
    }

    public get dims(): Vec3 { return this.layout.dims; }
    public get voxelSize(): number { return this.layout.voxelSize; }
    public get origin(): Vec3 { return this.layout.origin; }
    public get mipCount(): number { return new Uint32Array(this.gpu)[7]; }
    public get radianceScale(): number { return new Float32Array(this.gpu)[11]; }

    /**
     * Change the reference radiance is stored against (takes effect from the next producer pass:
     * last frame's content stays in the old scale until rewritten).
     */
    public setRadianceScale(scale: number): void {
        new Float32Array(this.gpu)[11] = Math.max(scale, 1e-6);
        this.device.queue.writeBuffer(this.uniform, 0, this.gpu);
    }

    /** The radiance texture (all mips). */
    public get gpuTexture(): GPUTexture {
        return this.texture.gpuTexture!;
    }

    /**
     * The volume as a `Texture` (shares the GPU texture), to attach to a material that cone
     * traces it: bind with `BindingLayouts.texture3d()`.
     */
    public asTexture(): Texture {
        return Texture.fromView('VoxelVolume/Radiance', this.gpuTexture, this.view, '3d');
    }

    /**
     * Also build anisotropic mips (Crassin et al. 2011): from mip 1 up, six directional chains
     * whose voxels composite their children front to back along each axis direction, so a cone
     * meets the face of a wall it reaches first instead of the mean of both faces (which halves a
     * lit room's walls at coarse mips). `buildMips` rebuilds them; cones read them through
     * `anisotropicViews`. About 0.86 times the memory of mip 0 more.
     */
    public setAnisotropicMips(anisotropic: boolean): void {
        this.anisotropic?.destroy();
        this.anisotropic = anisotropic ? new AnisotropicMips(this.device, this.gpuTexture) : undefined;
    }

    /**
     * The anisotropic chains (`setAnisotropicMips`), in the order +x, +y, +z, -x, -y, -z of the
     * direction a cone travels; their level 0 is the volume's mip 1.
     */
    public get anisotropicViews(): GPUTextureView[] | undefined {
        return this.anisotropic?.views;
    }

    /** Record the rebuild of mips 1.. (and the anisotropic chains) from mip 0. */
    public buildMips(encoder: GPUCommandEncoder): void {
        this.mips.encode(encoder);
        this.anisotropic?.encode(encoder);
    }

    /** Bytes of the anisotropic chains (0 without them). */
    public anisotropicBytes(): number {
        return this.anisotropic?.memoryBytes() ?? 0;
    }

    /** The radiance texture with all its mips plus the particle accumulators. */
    public memoryBytes(): number {
        return this.layout.memoryBytes();
    }

    public destroy(): void {
        this.anisotropic?.destroy();
        this.gpuTexture.destroy();
        this.uniform.destroy();
    }
}

/**
 * Builds a 3D texture's mip chain on the GPU (`rgba16float`, with `TEXTURE_BINDING` and
 * `STORAGE_BINDING`): one compute pass, one dispatch per level, each a 2x2x2 box filter of the
 * level before. WebGPU has no mip generation. Rust: `gi::Mip3d`.
 */
export class Mip3d {
    private readonly pipeline: GPUComputePipeline;
    /** Per level 1..: its bind group (reading the level before) and its size. */
    private readonly levels: { bindGroup: GPUBindGroup; size: Vec3 }[] = [];

    constructor(device: GPUDevice, texture: GPUTexture) {
        const layout = device.createBindGroupLayout({
            label: 'Mip3d/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
            ],
        });
        this.pipeline = device.createComputePipeline({
            label: 'Mip3d',
            layout: device.createPipelineLayout({ label: 'Mip3d', bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label: 'Mip3d', code: MIP3D_WGSL }), entryPoint: 'main' },
        });
        const levelView = (level: number) =>
            texture.createView({ label: 'Mip3d/Level', dimension: '3d', baseMipLevel: level, mipLevelCount: 1 });
        for (let level = 1; level < texture.mipLevelCount; level++) {
            this.levels.push({
                bindGroup: device.createBindGroup({
                    label: 'Mip3d/Level',
                    layout,
                    entries: [
                        { binding: 0, resource: levelView(level - 1) },
                        { binding: 1, resource: levelView(level) },
                    ],
                }),
                size: [texture.width, texture.height, texture.depthOrArrayLayers].map((d) => Math.max(d >> level, 1)) as Vec3,
            });
        }
    }

    /** Record the chain's rebuild from level 0. */
    public encode(encoder: GPUCommandEncoder): void {
        if (this.levels.length === 0) return;
        const pass = encoder.beginComputePass({ label: 'VoxelGI/Mips' });
        pass.setPipeline(this.pipeline);
        for (const { bindGroup, size: [w, h, d] } of this.levels) {
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        }
        pass.end();
    }
}
