import type { Scene } from '../objects/Scene';
import type { InstancedGeometry } from '../geometries/InstancedGeometry';
import type { Renderable } from '../objects/Renderable';
import type { ShadowMap } from '../shadows/ShadowMap';
import type { CubeMapShadowMap } from '../shadows/CubeMapShadowMap';
import { gradientSkyLighting } from './ParticleConeShading';
import { GiSurface, MeshVoxelizer, SurfaceSet } from './MeshVoxelizer';
import { SceneGiSettings, VoxelInjection, defaultSceneGiSettings } from './VoxelInjection';
import { Vec3, VoxelGiQuality, VoxelVolume } from './VoxelVolume';

/** What `Renderer.enableVoxelGI` builds. Rust: `gi::SceneVoxelGiOptions`. */
export interface SceneVoxelGiOptions {
    /**
     * The tier asked for; it steps down to what the device holds within `budgetBytes` (see
     * `VoxelGiQuality.fitScene`). Default `medium`.
     */
    quality?: VoxelGiQuality;
    /**
     * The box the volume covers, metres: the room, or the part of the scene whose light bounces.
     * Outside it there is no voxel GI.
     */
    boundsMin: Vec3;
    boundsMax: Vec3;
    /** The reference radiance is stored against (see `VoxelVolume`). Default 1. */
    radianceScale?: number;
    /** Most bytes the volume and its static surfaces may take (0, the default: no limit). 24 MiB keeps a phone at `low`. */
    budgetBytes?: number;
}

/** A GI renderable's draw this frame. */
interface VoxelDraw {
    renderable: Renderable;
    pipeline: GPURenderPipeline;
    meshOffset: number;
}

/**
 * Voxel GI for a scene's meshes (the renderer's, `Renderer.enableVoxelGI`). Each frame, after the
 * shadow maps and before the GBuffer:
 * 1. the renderables with a `Renderable.gi` surface are drawn into voxels through their own
 *    `vertex_main` (`MeshVoxelizer`): the static ones when they change, the dynamic ones every
 *    frame;
 * 2. the voxels are lit into the volume's mip 0 by the renderer's lights through their shadow
 *    maps, plus their emission and one more bounce of last frame's light (`settings`,
 *    `VoxelInjection`);
 * 3. the volume's mips are rebuilt.
 *
 * Read it with `VoxelGIEffect` (screen-space cones), or with `VOXEL_CONES_WGSL` from any pass or
 * material. Rust: `gi::SceneVoxelGi`.
 */
export class SceneVoxelGi {
    public readonly settings: SceneGiSettings;
    /** The tier in use (the one asked for, or lower if the device could not hold it). */
    public readonly quality: VoxelGiQuality;
    /** The volume of the scene's light, for its consumers. */
    public readonly volume: VoxelVolume;
    public readonly voxelizer: MeshVoxelizer;
    public readonly injection: VoxelInjection;
    /** The gradient sky (`setSkyGradient`): black until set. */
    private readonly sky: GPUBuffer;

    constructor(private readonly device: GPUDevice, options: SceneVoxelGiOptions) {
        this.quality = VoxelGiQuality.fitScene(options.quality ?? 'medium', device.limits, options.boundsMin, options.boundsMax, options.budgetBytes ?? 0);
        this.volume = new VoxelVolume(device, options.boundsMin, options.boundsMax, VoxelGiQuality.resolution(this.quality), options.radianceScale ?? 1);
        // walls show a cone the face it meets first, and stay opaque for it
        this.volume.setAnisotropicMips(true);
        this.voxelizer = new MeshVoxelizer(device, this.volume.layout);
        // no sky past the volume until one is set
        const black = gradientSkyLighting([0, 0, 0], [0, 0, 0]);
        this.sky = device.createBuffer({ label: 'VoxelGI/SceneSky', size: black.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.sky, 0, black);
        this.injection = new VoxelInjection(device, this.volume, this.sky);
        this.settings = { ...defaultSceneGiSettings(), bounceSteps: VoxelGiQuality.coneSteps(this.quality) / 2 };
    }

    /**
     * Voxelize the static renderables again next frame (after changing something the voxelizer
     * can't see, such as a material's texture).
     */
    public invalidate(): void {
        this.voxelizer.invalidate();
    }

    /**
     * The sky past the volume, from `down` to `up` (scene radiance), for the voxels' bounce
     * cones: see `gradientSkyLighting`. Black until set. Ignored after `useSkyLighting`.
     */
    public setSkyGradient(up: Vec3, down: Vec3): void {
        this.device.queue.writeBuffer(this.sky, 0, gradientSkyLighting(up, down));
    }

    /** Take the sky from `skyLighting` (a `SkyLighting` uniform) instead of the gradient. */
    public useSkyLighting(skyLighting: GPUBuffer): void {
        this.injection.setSky(skyLighting);
    }

    /**
     * Bytes on the GPU: the radiance with its mips (and anisotropic chains) and the surface
     * buffers.
     */
    public memoryBytes(): number {
        return this.volume.layout.radianceBytes() + this.volume.anisotropicBytes() + this.voxelizer.memoryBytes();
    }

    /**
     * Record the frame's voxel GI (the renderer's, after the shadow maps and before the GBuffer):
     * voxelize the visible GI renderables of `scene` (prepared this frame) with their matrices at
     * `meshOffset` of `meshBindGroup`, light the voxels through `shadowMap` (the directional map
     * materials sample, or null) and `pointShadows`, rebuild the mips.
     */
    public encode(
        encoder: GPUCommandEncoder,
        scene: Scene,
        meshBindGroup: GPUBindGroup,
        meshOffset: (renderable: Renderable) => number,
        shadowMap: ShadowMap | null,
        pointShadows: CubeMapShadowMap | null,
    ): void {
        if (!this.settings.enabled) return;
        this.injection.syncShadowMaps(shadowMap, pointShadows);
        this.injection.updateLights(scene.directionalLights, scene.pointLights, scene.areaLights);

        // the renderables drawn into voxels: visible, with a surface; in slot order, which (unlike
        // the transparent ones' draw order) stays put from frame to frame
        const voxelizer = this.voxelizer;
        const draws: VoxelDraw[] = [];
        for (const r of scene.getOrderedObjects()) {
            if (!r.gi || !r.geometry.initialized) continue;
            const pipeline = r.material.getVoxelPipeline(this.device, voxelizer.id, r.geometry.vertexBuffersDescriptors,
                voxelizer.bindGroupLayout, voxelizer.fragment, MeshVoxelizer.TARGET_FORMAT, voxelizer.sampleCount);
            draws.push({ renderable: r, pipeline, meshOffset: meshOffset(r) });
        }
        draws.sort((a, b) => a.meshOffset - b.meshOffset);
        // what the static surfaces are made of: a change voxelizes them again
        const key: unknown[] = [];
        for (const { renderable: r, meshOffset: offset } of draws) {
            if (r.dynamic) continue;
            const surface: GiSurface = r.gi!;
            const instances = r.geometry.isInstancedGeometry ? (r.geometry as InstancedGeometry).instanceCount : 1;
            key.push(r, offset, r.geometry, r.geometry.vertexCount, instances, ...r.worldMatrix.internalMat4, ...surface.albedo, ...(surface.emission ?? [0, 0, 0]));
        }
        voxelizer.writeDraws(draws.map((d) => d.renderable.gi!));
        const staticChanged = voxelizer.staticChanged(key);
        if (draws.some((d) => d.renderable.dynamic)) voxelizer.ensureDynamic();

        if (staticChanged) voxelizer.encodeSet(encoder, SurfaceSet.Static, draws, meshBindGroup);
        if (voxelizer.dynamicSurfaces) voxelizer.encodeSet(encoder, SurfaceSet.Dynamic, draws, meshBindGroup);
        this.injection.encode(encoder, voxelizer, this.settings);
        this.volume.buildMips(encoder);
    }

    public destroy(): void {
        this.voxelizer.destroy();
        this.injection.destroy();
        this.volume.destroy();
        this.sky.destroy();
    }
}
