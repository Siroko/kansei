import { ANISO_MIP_WGSL } from './GiWGSL';
import type { Vec3 } from './VoxelVolume';

/**
 * The six directional mip chains of a `VoxelVolume` (`VoxelVolume.setAnisotropicMips`): from its
 * mip 1 up, one `rgba16float` chain per direction a cone travels along an axis (+x, +y, +z, -x,
 * -y, -z), each voxel its children composited front to back along that direction
 * (aniso_mip.wgsl). About 0.86 times the memory of the volume's mip 0. Rust: `gi::aniso`.
 */
export class AnisotropicMips {
    /** Directions of `views`, in order. */
    public static readonly DIRECTIONS = ['+x', '+y', '+z', '-x', '-y', '-z'] as const;

    /** Every level of each direction's chain (in `DIRECTIONS` order), to sample. */
    public readonly views: GPUTextureView[];

    private readonly textures: GPUTexture[];
    private readonly pipelines: GPUComputePipeline[];
    /** Per level and sign: the pipeline (index into `pipelines`), its bind group and its size. */
    private readonly passes: { pipeline: number; bindGroup: GPUBindGroup; size: Vec3 }[] = [];

    /** The chains of `volume` (an `rgba16float` 3D texture with a full mip chain). */
    constructor(device: GPUDevice, volume: GPUTexture) {
        const levels = volume.mipLevelCount - 1;
        const [w, h, d] = [volume.width, volume.height, volume.depthOrArrayLayers].map((s) => Math.max(s >> 1, 1));
        this.textures = AnisotropicMips.DIRECTIONS.map(() => device.createTexture({
            label: 'VoxelGI/AnisotropicMips',
            size: { width: w, height: h, depthOrArrayLayers: d },
            mipLevelCount: Math.max(levels, 1),
            dimension: '3d',
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        }));
        this.views = this.textures.map((t) => t.createView({ dimension: '3d' }));
        const level = (texture: GPUTexture, l: number) =>
            texture.createView({ label: 'VoxelGI/AnisotropicLevel', dimension: '3d', baseMipLevel: l, mipLevelCount: 1 });
        const iso = level(volume, 0);

        const sampled = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } });
        const storage = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } });
        const layout = device.createBindGroupLayout({
            label: 'VoxelGI/AnisotropicMipsBGL',
            entries: [sampled(0), sampled(1), sampled(2), sampled(3), storage(4), storage(5), storage(6)],
        });
        const module = device.createShaderModule({ label: 'VoxelGI/AnisotropicMips', code: ANISO_MIP_WGSL });
        const pipelineLayout = device.createPipelineLayout({ label: 'VoxelGI/AnisotropicMips', bindGroupLayouts: [layout] });
        this.pipelines = ['first_pos', 'first_neg', 'down_pos', 'down_neg'].map((entryPoint) =>
            device.createComputePipeline({ label: 'VoxelGI/AnisotropicMips', layout: pipelineLayout, compute: { module, entryPoint } }));

        for (let l = 0; l < levels; l++) {
            for (let sign = 0; sign < 2; sign++) {
                const first = 3 * sign;
                const dst = [0, 1, 2].map((i) => level(this.textures[first + i], l));
                // level 0 reads the isotropic mip 0 (and binds it in the unused slots); the levels
                // above read their own chains' level below
                const src = [0, 1, 2].map((i) => (l === 0 ? iso : level(this.textures[first + i], l - 1)));
                const bindGroup = device.createBindGroup({
                    label: 'VoxelGI/AnisotropicMipsBG',
                    layout,
                    entries: [iso, ...src, ...dst].map((resource, binding) => ({ binding, resource })),
                });
                this.passes.push({
                    pipeline: l === 0 ? sign : 2 + sign,
                    bindGroup,
                    size: [w, h, d].map((s) => Math.max(s >> l, 1)) as Vec3,
                });
            }
        }
    }

    /** Bytes on the GPU. */
    public memoryBytes(): number {
        let bytes = 0;
        for (const t of this.textures) {
            for (let l = 0; l < t.mipLevelCount; l++) {
                bytes += [t.width, t.height, t.depthOrArrayLayers].reduce((n, d) => n * Math.max(d >> l, 1), 1) * 8;
            }
        }
        return bytes;
    }

    /** Record the chains' rebuild from the volume's mip 0. */
    public encode(encoder: GPUCommandEncoder): void {
        const pass = encoder.beginComputePass({ label: 'VoxelGI/AnisotropicMips' });
        for (const { pipeline, bindGroup, size: [w, h, d] } of this.passes) {
            pass.setPipeline(this.pipelines[pipeline]);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        }
        pass.end();
    }

    public destroy(): void {
        for (const t of this.textures) t.destroy();
    }
}
