import { ComputeShadows, CascadedShadowSource } from '../shadows/ComputeShadows';
import type { ShadowMap } from '../shadows/ShadowMap';
import type { CubeMapShadowMap } from '../shadows/CubeMapShadowMap';
import { gpuPass } from '../profiling/Profiler';
import { CLIPMAP_INJECT_WGSL } from './GiWGSL';
import type { ClipmapVoxelizer } from './ClipmapVoxelizer';
import { VoxelClipmap, clipmapEntries, clipmapLayoutEntries } from './VoxelClipmap';

/**
 * Where a voxel clipmap's injection takes shadows from cones traced through the clipmap
 * (`ClipmapGiSettings.coneShadows`). Rust: `gi::ConeShadows`.
 * - `off`: the shadow maps alone (and none where they don't reach);
 * - `fallback`: cones where no shadow map covers the voxel (a sun past its cascades, lights
 *   without a map);
 * - `always`: cones for every light.
 */
export type ConeShadows = 'off' | 'fallback' | 'always';
/** `ConeShadows` in the order of the WGSL's `ClipInjectParams.coneShadows`. */
export const CONE_SHADOWS_MODES: readonly ConeShadows[] = ['off', 'fallback', 'always'];

/**
 * How a voxel clipmap's voxels are lit (`SceneVoxelClipmap.settings`); change it freely between
 * frames. Rust: `gi::ClipmapGiSettings`.
 */
export interface ClipmapGiSettings {
    /** false: the clipmap is left as it is (nothing is voxelized, moved or lit). */
    enabled: boolean;
    /** The share of last frame's indirect light each voxel bounces again (1 is physical). */
    bounce: number;
    /**
     * Voxels whose albedo (its largest channel) is below this skip the bounce: they pass on only
     * that share of the light they receive, a dark forest's needles most of the voxels and most of
     * the bounce's cost.
     */
    bounceMinAlbedo: number;
    /** The most steps each of a voxel's bounce cones takes (they widen fast: 12 cross the clipmap). */
    bounceSteps: number;
    /** How much of the sky's light the bounce cones bring in where they leave the clipmap. */
    skyScale: number;
    /** Scale of the surfaces' emission. */
    emissionScale: number;
    /** Voxels the shadow lookups move out along the normal. */
    shadowOffsetVoxels: number;
    /** Shadows from cones through the clipmap, where no map reaches or always. */
    coneShadows: ConeShadows;
    /** Tangent of the shadow cones' half angle: wider is softer (and cheaper). */
    coneShadowTan: number;
    /**
     * The most steps each shadow cone takes (narrow, they widen slowly: 48 reach about 20 times as
     * far as they start).
     */
    coneShadowSteps: number;
    /**
     * Levels lit each frame: the finest every frame and the others in turn (0: all of them every
     * frame). The coarse levels change slowly; lighting fewer a frame saves most of the
     * injection's cost.
     */
    levelsPerFrame: number;
}

export function defaultClipmapGiSettings(): ClipmapGiSettings {
    return {
        enabled: true,
        bounce: 1,
        bounceMinAlbedo: 0.08,
        bounceSteps: 12,
        skyScale: 1,
        emissionScale: 1,
        shadowOffsetVoxels: 1,
        coneShadows: 'fallback',
        coneShadowTan: 0.08,
        coneShadowSteps: 48,
        levelsPerFrame: 2,
    };
}

/** Bytes of the WGSL `ClipInjectParams` (clipmap_inject.wgsl; Rust `ClipInjectParamsGpu`). */
export const CLIP_INJECT_PARAMS_BYTES = 64;

/**
 * The clipmap's voxelized surfaces into light, a level at a time (clipmap_inject.wgsl), from the
 * renderer's lights and shadow maps (`ComputeShadows`), last frame's clipmap (the bounce, the cone
 * shadows) and emission. Each level is written into the scratch level, then copied over it.
 * Rust: `gi::clipmap_inject::ClipmapInjection`.
 */
export class ClipmapInjection {
    /** The renderer's lights and shadow maps as the pass reads them. */
    public readonly shadows = new ComputeShadows();

    private readonly pipeline: GPUComputePipeline;
    private readonly bgl: GPUBindGroupLayout;
    /** Per level, and whether it bound the level's dynamic surfaces. */
    private readonly groups: ({ group: GPUBindGroup, dynamic: boolean } | null)[];
    /** A slot per level (`paramsStride` apart), written once a frame. */
    private readonly params: GPUBuffer;
    private readonly paramsStride: number;
    private readonly noSurfaces: GPUBuffer;
    private shadowMap: ShadowMap | CascadedShadowSource | null = null;
    private pointShadows: CubeMapShadowMap | null = null;

    constructor(private readonly device: GPUDevice, private readonly clipmap: VoxelClipmap, private sky: GPUBuffer) {
        const visibility = GPUShaderStage.COMPUTE;
        const storage: GPUBufferBindingLayout = { type: 'read-only-storage' };
        this.bgl = device.createBindGroupLayout({
            label: 'VoxelClipmap/InjectBGL',
            entries: [
                { binding: 0, visibility, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: CLIP_INJECT_PARAMS_BYTES } },
                { binding: 10, visibility, buffer: storage },
                { binding: 11, visibility, buffer: storage },
                { binding: 14, visibility, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 15, visibility, buffer: { type: 'uniform' } },
                ...ComputeShadows.layoutEntries(),
                ...clipmapLayoutEntries(visibility),
            ],
        });
        this.pipeline = device.createComputePipeline({
            label: 'VoxelClipmap/Inject',
            layout: device.createPipelineLayout({ label: 'VoxelClipmap/Inject', bindGroupLayouts: [this.bgl] }),
            compute: { module: device.createShaderModule({ label: 'VoxelClipmap/Inject', code: CLIPMAP_INJECT_WGSL }), entryPoint: 'main' },
        });
        const levels = clipmap.layout.levels;
        const alignment = device.limits.minUniformBufferOffsetAlignment;
        this.paramsStride = Math.ceil(CLIP_INJECT_PARAMS_BYTES / alignment) * alignment;
        this.params = device.createBuffer({ label: 'VoxelClipmap/InjectParams', size: this.paramsStride * levels, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.groups = Array.from({ length: levels }, () => null);
        this.noSurfaces = device.createBuffer({ label: 'VoxelClipmap/NoSurfaces', size: 16, usage: GPUBufferUsage.STORAGE });
    }

    /** Read the sky (past the clipmap) from `sky`, a `SkyLighting` uniform. */
    public setSky(sky: GPUBuffer): void {
        this.sky = sky;
        this.groups.fill(null);
    }

    /**
     * Read the renderer's shadow maps: the directional map materials sample or a cascaded map
     * (its widest cascade; null when shadows are off) and the point lights' cube map. Only a
     * change rebinds.
     */
    public syncShadowMaps(shadowMap: ShadowMap | CascadedShadowSource | null, pointShadows: CubeMapShadowMap | null): void {
        if (shadowMap !== this.shadowMap) {
            this.shadowMap = shadowMap;
            if (shadowMap && 'farView' in shadowMap) this.shadows.setCascadedShadowMap(shadowMap);
            else this.shadows.setShadowMap(shadowMap);
        }
        if (pointShadows !== this.pointShadows) {
            this.pointShadows = pointShadows;
            this.shadows.setPointShadows(pointShadows);
        }
    }

    /**
     * Record lighting `levels` of the clipmap (those with a window), each into the scratch level
     * then copied over it, with `settings`; `hasDynamic(level)`: its dynamic surfaces hold
     * renderables this frame.
     */
    public encode(encoder: GPUCommandEncoder, voxelizer: ClipmapVoxelizer, levels: readonly number[], hasDynamic: (level: number) => boolean, settings: ClipmapGiSettings): void {
        const clipmap = this.clipmap;
        if (this.shadows.prepare(this.device)) this.groups.fill(null);
        const layout = clipmap.layout;
        // every level's slot in one write (`queue.writeBuffer` lands before the frame's work)
        const data = new ArrayBuffer(this.paramsStride * layout.levels);
        for (let level = 0; level < layout.levels; level++) {
            const u32 = new Uint32Array(data, level * this.paramsStride, CLIP_INJECT_PARAMS_BYTES / 4);
            const f32 = new Float32Array(data, level * this.paramsStride, CLIP_INJECT_PARAMS_BYTES / 4);
            u32[0] = this.shadows.dirCount;
            u32[1] = this.shadows.pointCount;
            u32[2] = this.shadows.hasShadowMap ? 1 : 0;
            u32[3] = this.shadows.hasPointShadows ? 1 : 0;
            f32[4] = Math.max(settings.bounce, 0);
            f32[5] = Math.max(settings.skyScale, 0);
            f32[6] = Math.max(settings.emissionScale, 0);
            f32[7] = settings.shadowOffsetVoxels;
            u32[8] = Math.max(settings.bounceSteps, 1);
            u32[9] = hasDynamic(level) ? 1 : 0;
            u32[10] = level;
            u32[11] = Math.max(CONE_SHADOWS_MODES.indexOf(settings.coneShadows), 0);
            f32[12] = Math.max(settings.coneShadowTan, 1e-3);
            u32[13] = Math.max(settings.coneShadowSteps, 1);
            f32[14] = Math.max(settings.bounceMinAlbedo, 0);
        }
        this.device.queue.writeBuffer(this.params, 0, data);
        const [w, h, d] = layout.dims;
        for (const level of levels) {
            if (!clipmap.origin(level)) continue;
            const dynamic = hasDynamic(level) ? voxelizer.dynamicSurfaces(level) : null;
            let bound = this.groups[level];
            if (!bound || bound.dynamic !== (dynamic !== null)) {
                bound = {
                    group: this.device.createBindGroup({
                        label: 'VoxelClipmap/InjectBG',
                        layout: this.bgl,
                        entries: [
                            { binding: 0, resource: { buffer: this.params, offset: 0, size: CLIP_INJECT_PARAMS_BYTES } },
                            { binding: 10, resource: { buffer: voxelizer.staticSurfaces(level) } },
                            { binding: 11, resource: { buffer: dynamic ?? this.noSurfaces } },
                            { binding: 14, resource: clipmap.scratchView },
                            { binding: 15, resource: { buffer: this.sky } },
                            ...this.shadows.entries(),
                            ...clipmapEntries(clipmap),
                        ],
                    }),
                    dynamic: dynamic !== null,
                };
                this.groups[level] = bound;
            }
            const pass = encoder.beginComputePass({ label: 'VoxelClipmap/Inject', timestampWrites: gpuPass('VoxelClipmap/Inject') });
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, bound.group, [level * this.paramsStride]);
            pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
            pass.end();
            clipmap.copyScratchTo(encoder, level);
        }
    }

    public destroy(): void {
        this.shadows.destroy();
        this.params.destroy();
        this.noSurfaces.destroy();
    }
}
