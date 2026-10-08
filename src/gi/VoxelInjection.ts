import { ComputeShadows } from '../shadows/ComputeShadows';
import type { ShadowMap } from '../shadows/ShadowMap';
import type { CubeMapShadowMap } from '../shadows/CubeMapShadowMap';
import type { DirectionalLight } from '../lights/DirectionalLight';
import type { PointLight } from '../lights/PointLight';
import type { AreaLight } from '../lights/AreaLight';
import { gpuPass } from '../profiling/Profiler';
import { INJECT_WGSL } from './GiWGSL';
import type { MeshVoxelizer } from './MeshVoxelizer';
import type { VoxelVolume } from './VoxelVolume';

/**
 * Where the injection takes shadows from the distance field (G-3's `SceneVoxelGi.enableSdf`;
 * without a field every setting reads the shadow maps alone). Rust: `gi::SdfShadows`.
 * - `off`: the shadow maps alone (and none where they don't reach);
 * - `fallback`: the field where no shadow map covers the voxel;
 * - `always`: the field for every light.
 */
export type SdfShadows = 'off' | 'fallback' | 'always';

/** How the scene's voxels are lit (`SceneVoxelGi.settings`); change it freely between frames. Rust: `gi::SceneGiSettings`. */
export interface SceneGiSettings {
    /** false: the volume is left as it is (nothing is voxelized or lit). */
    enabled: boolean;
    /**
     * The share of last frame's indirect light each voxel bounces again: 1 is physical (the
     * bounces add up over frames, each dimmed by the albedo), 0 keeps direct light only.
     */
    bounce: number;
    /** The most steps each of a voxel's bounce cones takes. */
    bounceSteps: number;
    /** How much of the sky's light the bounce cones bring in where they leave the volume. */
    skyScale: number;
    /** Scale of the surfaces' emission. */
    emissionScale: number;
    /**
     * Voxels the shadow lookups move out along the normal, so a surface voxel (up to half a voxel
     * inside its surface) does not shadow itself.
     */
    shadowOffsetVoxels: number;
    /** Shadows through the distance field (ignored without one). */
    sdfShadows: SdfShadows;
    /** The field's soft shadows: Quilez's k, higher is harder (8 is a soft sun). */
    sdfShadowHardness: number;
}

export function defaultSceneGiSettings(): SceneGiSettings {
    return {
        enabled: true,
        bounce: 1,
        bounceSteps: 16,
        skyScale: 1,
        emissionScale: 1,
        shadowOffsetVoxels: 1,
        sdfShadows: 'off',
        sdfShadowHardness: 8,
    };
}

/** Bytes of the WGSL `InjectParams` (inject.wgsl; Rust `InjectParamsGpu`). */
export const INJECT_PARAMS_BYTES = 64;

/**
 * A 1-texel stand-in for a distance field, bound while there is none (and never read then).
 * `rgba16float`, filterable on any device. Rust: `gi::inject::no_sdf`.
 */
export function noSdfView(device: GPUDevice): GPUTextureView {
    return device.createTexture({
        label: 'VoxelGI/NoSdf',
        size: [1, 1, 1],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING,
    }).createView();
}

/**
 * The voxelized surfaces into light: the volume's mip 0 (inject.wgsl), from the renderer's lights
 * and shadow maps (`ComputeShadows`), last frame's mips (the bounce) and emission. One thread per
 * voxel; a voxel is a Lambertian surface, albedo / pi times its irradiance plus its emission.
 * Rust: `gi::inject::RadianceInjection`.
 */
export class VoxelInjection {
    /** The renderer's lights and shadow maps as the pass reads them. */
    public readonly shadows = new ComputeShadows();

    private readonly pipeline: GPUComputePipeline;
    private readonly bgl: GPUBindGroupLayout;
    private group: GPUBindGroup | null = null;
    private readonly params: GPUBuffer;
    private readonly previous: GPUTextureView;
    /** A one-voxel stand-in for the dynamic surfaces before there are any. */
    private readonly noSurfaces: GPUBuffer;
    private readonly noSdf: GPUTextureView;
    /** Whether `group` binds the voxelizer's dynamic surfaces. */
    private boundDynamic = false;
    private shadowMap: ShadowMap | null = null;
    private pointShadows: CubeMapShadowMap | null = null;

    constructor(private readonly device: GPUDevice, private readonly volume: VoxelVolume, private sky: GPUBuffer) {
        const visibility = GPUShaderStage.COMPUTE;
        const uniform: GPUBufferBindingLayout = { type: 'uniform' };
        const storage: GPUBufferBindingLayout = { type: 'read-only-storage' };
        const texture3d: GPUTextureBindingLayout = { sampleType: 'float', viewDimension: '3d' };
        const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility, buffer: uniform },
            { binding: 2, visibility, buffer: uniform },
            { binding: 10, visibility, buffer: storage },
            { binding: 11, visibility, buffer: storage },
            { binding: 12, visibility, texture: texture3d },
            { binding: 13, visibility, sampler: { type: 'filtering' } },
            { binding: 14, visibility, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
            { binding: 15, visibility, buffer: uniform },
            ...ComputeShadows.layoutEntries(),
        ];
        // the anisotropic mips and the distance field (voxel_irradiance.wgsl)
        for (let binding = 40; binding < 47; binding++) entries.push({ binding, visibility, texture: texture3d });
        this.bgl = device.createBindGroupLayout({ label: 'VoxelGI/InjectBGL', entries });
        this.pipeline = device.createComputePipeline({
            label: 'VoxelGI/Inject',
            layout: device.createPipelineLayout({ label: 'VoxelGI/Inject', bindGroupLayouts: [this.bgl] }),
            compute: { module: device.createShaderModule({ label: 'VoxelGI/Inject', code: INJECT_WGSL }), entryPoint: 'main' },
        });
        const mips = volume.mipCount;
        this.previous = volume.gpuTexture.createView({
            label: 'VoxelGI/PreviousMips',
            dimension: '3d',
            baseMipLevel: Math.min(1, mips - 1),
            mipLevelCount: Math.max(mips - 1, 1),
        });
        this.params = device.createBuffer({ label: 'VoxelGI/InjectParams', size: INJECT_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.noSurfaces = device.createBuffer({ label: 'VoxelGI/NoSurfaces', size: 16, usage: GPUBufferUsage.STORAGE });
        this.noSdf = noSdfView(device);
    }

    /** Read the sky (past the volume) from `sky`, a `SkyLighting` uniform. */
    public setSky(sky: GPUBuffer): void {
        this.sky = sky;
        this.group = null;
    }

    /**
     * Read the renderer's shadow maps: the directional map materials sample (null when shadows
     * are off) and the point lights' cube map. Only a change rebinds.
     */
    public syncShadowMaps(shadowMap: ShadowMap | null, pointShadows: CubeMapShadowMap | null): void {
        if (shadowMap !== this.shadowMap) {
            this.shadowMap = shadowMap;
            this.shadows.setShadowMap(shadowMap);
        }
        if (pointShadows !== this.pointShadows) {
            this.pointShadows = pointShadows;
            this.shadows.setPointShadows(pointShadows);
        }
    }

    /**
     * Collect the scene's lights, every one of them (not only the volumetric ones). Area lights
     * are points at their position without a shadow: the injection reads the cube map only.
     */
    public updateLights(dir: readonly DirectionalLight[], point: readonly PointLight[], area: readonly AreaLight[]): void {
        this.shadows.updateLights(dir, point, area, false, false);
    }

    /** Record the injection into the volume's mip 0 (build its mips next). */
    public encode(encoder: GPUCommandEncoder, voxelizer: MeshVoxelizer, settings: SceneGiSettings): void {
        const dynamic = voxelizer.dynamicSurfaces;
        if (this.shadows.prepare(this.device) || (dynamic !== null) !== this.boundDynamic) this.group = null;
        if (!this.group) {
            const anisotropic = this.volume.anisotropicViews;
            if (!anisotropic) throw new Error('scene GI volumes have anisotropic mips');
            this.group = this.device.createBindGroup({
                label: 'VoxelGI/InjectBG',
                layout: this.bgl,
                entries: [
                    { binding: 0, resource: { buffer: this.volume.uniform } },
                    { binding: 2, resource: { buffer: this.params } },
                    { binding: 10, resource: { buffer: voxelizer.staticSurfaces } },
                    { binding: 11, resource: { buffer: dynamic ?? this.noSurfaces } },
                    { binding: 12, resource: this.previous },
                    { binding: 13, resource: this.volume.sampler },
                    { binding: 14, resource: this.volume.mip0StorageView },
                    { binding: 15, resource: { buffer: this.sky } },
                    ...this.shadows.entries(),
                    ...anisotropic.map((view, i) => ({ binding: 40 + i, resource: view })),
                    { binding: 46, resource: this.noSdf },
                ],
            });
            this.boundDynamic = dynamic !== null;
        }
        const data = new ArrayBuffer(INJECT_PARAMS_BYTES);
        const u32 = new Uint32Array(data);
        const f32 = new Float32Array(data);
        u32[0] = this.shadows.dirCount;
        u32[1] = this.shadows.pointCount;
        u32[2] = this.shadows.hasShadowMap ? 1 : 0;
        u32[3] = this.shadows.hasPointShadows ? 1 : 0;
        f32[4] = Math.max(settings.bounce, 0);
        f32[5] = Math.max(settings.skyScale, 0);
        f32[6] = Math.max(settings.emissionScale, 0);
        f32[7] = settings.shadowOffsetVoxels;
        u32[8] = Math.max(settings.bounceSteps, 1);
        u32[9] = dynamic ? 1 : 0;
        // no distance field yet: the shadow maps alone (`sdfShadows` needs one)
        u32[10] = 0;
        f32[11] = Math.max(settings.sdfShadowHardness, 0.1);
        this.device.queue.writeBuffer(this.params, 0, data);

        const [w, h, d] = this.volume.dims;
        const pass = encoder.beginComputePass({ label: 'VoxelGI/Inject', timestampWrites: gpuPass('VoxelGI/Inject') });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.group);
        pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        pass.end();
    }

    public destroy(): void {
        this.shadows.destroy();
        this.params.destroy();
        this.noSurfaces.destroy();
    }
}
