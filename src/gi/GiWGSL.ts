/**
 * Voxel GI's WGSL, imported from `rust/kansei-core/src/gi/shaders` (Vite `?raw`) as
 * `materials/shaders/SharedWGSL.ts` imports the rest, so both engines run the same source and
 * `cargo test -p kansei-core` validates it. The exported names match the Rust constants
 * (`gi::VOXEL_VOLUME_WGSL`, `gi::VOXEL_CONES_WGSL`, `gi::SDF_WGSL`, `gi::PROBES_WGSL`,
 * `gi::PARTICLE_EMISSION_WGSL`, `gi::CLIPMAP_WGSL`, `gi::CLIPMAP_PROBES_WGSL`,
 * `atmosphere::SKY_LIGHTING_WGSL`); the rest are the passes' own,
 * assembled as Rust's `concat!`.
 */
import { assemble } from '../materials/shaders/ShaderUtils';
import voxelVolume from '../../rust/kansei-core/src/gi/shaders/voxel_volume.wgsl?raw';
import voxelCones from '../../rust/kansei-core/src/gi/shaders/voxel_cones.wgsl?raw';
import mip3d from '../../rust/kansei-core/src/gi/shaders/mip3d.wgsl?raw';
import anisoMip from '../../rust/kansei-core/src/gi/shaders/aniso_mip.wgsl?raw';
import particleEmission from '../../rust/kansei-core/src/gi/shaders/particle_emission.wgsl?raw';
import particleSplat from '../../rust/kansei-core/src/gi/shaders/particle_splat.wgsl?raw';
import particleResolve from '../../rust/kansei-core/src/gi/shaders/particle_resolve.wgsl?raw';
import particleCones from '../../rust/kansei-core/src/gi/shaders/particle_cones.wgsl?raw';
import sdf from '../../rust/kansei-core/src/gi/shaders/sdf.wgsl?raw';
import skyLighting from '../../rust/kansei-core/src/atmosphere/shaders/sky_lighting.wgsl?raw';
import voxelWrite from '../../rust/kansei-core/src/gi/shaders/voxel_write.wgsl?raw';
import voxelFragment from '../../rust/kansei-core/src/gi/shaders/voxel_fragment.wgsl?raw';
import voxelIrradiance from '../../rust/kansei-core/src/gi/shaders/voxel_irradiance.wgsl?raw';
import inject from '../../rust/kansei-core/src/gi/shaders/inject.wgsl?raw';
import screenCommon from '../../rust/kansei-core/src/gi/shaders/screen_common.wgsl?raw';
import screenNormal from '../../rust/kansei-core/src/gi/shaders/screen_normal.wgsl?raw';
import screenTrace from '../../rust/kansei-core/src/gi/shaders/screen_trace.wgsl?raw';
import screenTemporal from '../../rust/kansei-core/src/gi/shaders/screen_temporal.wgsl?raw';
import screenComposite from '../../rust/kansei-core/src/gi/shaders/screen_composite.wgsl?raw';
import probeCommon from '../../rust/kansei-core/src/gi/shaders/probe_common.wgsl?raw';
import probeIrradiance from '../../rust/kansei-core/src/gi/shaders/probe_irradiance.wgsl?raw';
import probeUpdate from '../../rust/kansei-core/src/gi/shaders/probe_update.wgsl?raw';
import jumpFlood from '../../rust/kansei-core/src/gi/shaders/jump_flood.wgsl?raw';
import clipmap from '../../rust/kansei-core/src/gi/shaders/clipmap.wgsl?raw';
import clipmapClear from '../../rust/kansei-core/src/gi/shaders/clipmap_clear.wgsl?raw';
import clipmapInject from '../../rust/kansei-core/src/gi/shaders/clipmap_inject.wgsl?raw';
import clipmapProbes from '../../rust/kansei-core/src/gi/shaders/clipmap_probes.wgsl?raw';
import clipmapProbeUpdate from '../../rust/kansei-core/src/gi/shaders/clipmap_probe_update.wgsl?raw';
import clipmapProbeTrace from '../../rust/kansei-core/src/gi/shaders/clipmap_probe_trace.wgsl?raw';
import clipmapTrace from '../../rust/kansei-core/src/gi/shaders/clipmap_trace.wgsl?raw';
import { COMPUTE_SHADOWS_WGSL } from '../shadows/ComputeShadows';

/**
 * The WGSL `VoxelVolume` struct and `voxelUvw` / `voxelLinearIndex`: bind `VoxelVolume.uniform`
 * as a `VoxelVolume` uniform.
 */
export const VOXEL_VOLUME_WGSL: string = voxelVolume;

/**
 * `VOXEL_VOLUME_WGSL` plus `voxelConeTrace(vol, radiance, sampler, origin, dir, tanHalf,
 * startDist, maxDist, maxSteps)`: the scene radiance a cone gathers (rgb) and the transmittance
 * left past it (a); `voxelConeTraceSplit` (also the transmittance at a near distance) and
 * `voxelHemisphereCone` (the five cones of a hemisphere). Bind `VoxelVolume.view` as
 * `texture_3d<f32>` and `VoxelVolume.sampler`.
 */
export const VOXEL_CONES_WGSL: string = assemble([voxelVolume, voxelCones]);

/**
 * `ParticleEmission` and `particleEmission(e, index, velocity)`, to tell which particles glow as
 * the GI does (it also hands each particle its emission in the lighting buffer).
 */
export const PARTICLE_EMISSION_WGSL: string = particleEmission;

/**
 * The `SkyLighting` uniform (order-2 SH of the sky's radiance, sun and moon) and `skyIrradiance`,
 * `skyRadiance`, `skyInscatter`: the light past a voxel volume (`gradientSkyLighting` writes one
 * without an atmosphere).
 */
export const SKY_LIGHTING_WGSL: string = skyLighting;

/**
 * For a material's `voxelFragmentEntry` (`MaterialOptions`): the voxelizer's group 3 and
 * `kansei_voxel_write(fragPos, front, albedo, emission)`, which puts the surface at a fragment
 * into its voxel. Call it in uniform control flow (it takes derivatives):
 *
 * ```wgsl
 * @fragment
 * fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
 *     let albedo = textureSample(base_texture, base_sampler, in.uv).rgb;
 *     kansei_voxel_write(in.clip, front, albedo, vec3f(0.0));
 * }
 * ```
 *
 * `kansei_voxel_draw.albedo` and `.emission` hold the renderable's `GiSurface`. Group 3 binds
 * 100-102 here, apart from the shadow group's, so a material may use both chunks.
 * Rust: `gi::VOXEL_WRITE_WGSL`.
 */
export const VOXEL_WRITE_WGSL: string = voxelWrite;

/**
 * `VOXEL_VOLUME_WGSL` plus reading a `JumpFloodSdf`: `sdfDistance(vol, sdf, sampler, p)`,
 * `sdfSoftShadow(vol, sdf, sampler, p, toLight, k, maxT)`, `sdfSurfaceShadow(vol, sdf, sampler,
 * p, n, toLight, k, maxT)` (for a point on a surface of normal `n`: its own plane does not shadow
 * it at grazing angles), `sdfLightShape(vol, radius, distance)` (k and the march's length for a
 * light of that radius and distance) and `sdfAo(vol, sdf, sampler, p, n)`.
 * The material helper for distance-field shadows and AO: bind `JumpFloodSdf.asTexture()` as a
 * `texture_3d<f32>`, and the volume's uniform and sampler, in the material's own group.
 * Rust: `gi::SDF_WGSL`.
 */
export const SDF_WGSL: string = assemble([voxelVolume, sdf]);

/**
 * Diffuse light from the scene's probes (`SdfProbes`) for a material: `ProbeGrid` and
 * `kansei_gi_irradiance(p, n)`, the irradiance a surface at world position `p` facing `n`
 * receives (scene units; a diffuse surface adds albedo / pi times it, in place of its own sky
 * ambient), weighed over the eight probes around `p` as DDGI does. Declare its buffers with
 * `SdfProbes.bindingsWGSL(group, first)` in the material's own group and bind them with
 * `SdfProbes.bindGroupEntries` (or as storage and uniform bindables). Rust: `gi::PROBES_WGSL`.
 */
export const PROBES_WGSL: string = assemble([probeCommon, probeIrradiance]);

/**
 * A voxel clipmap for a compute pass (`VoxelClipmap`): the `VoxelClipmap` uniform and the levels
 * bound in group 0 (binding 50 `VoxelClipmap.uniform`, 51-56 `levelViews`, 57 `sampler`;
 * `clipmapLayoutEntries` and `clipmapEntries` lay them out and bind them); `clipSample(level, p)`,
 * `clipContains`, `clipLevelAt(p, first, margin)`, `clipVoxelSize`; `clipConeTrace(origin, dir, n,
 * tanHalf, minDiameter, startDist, maxDist, maxSteps)`, a cone through the levels (radiance
 * gathered, transmittance left), and `clipIrradiance(sky, skyScale, origin, n, angle, minDiameter,
 * startDist, maxDist, maxSteps)`, a surface's irradiance from six cones and the sky past the
 * clipmap (needs `SKY_LIGHTING_WGSL`). Rust: `gi::CLIPMAP_WGSL`.
 */
export const CLIPMAP_WGSL: string = clipmap;

/**
 * The irradiance probes of a voxel clipmap for a material (`ClipmapProbes`): `ClipProbeGrid`,
 * `kansei_clipmap_light(p, n)` (the irradiance a surface at world position `p` facing `n`
 * receives, scene units, and the cosine-weighted share of its hemisphere that sees the sky past
 * the clipmap; a = -1 where no probe holds `p`), `kansei_clipmap_sky_visibility(p, n)` (that share
 * alone, 1 where no probe holds `p`: to dim a material's own sky light by) and
 * `kansei_clipmap_inscatter(p, viewDir, g)` (the light a medium there scatters toward the camera,
 * for fog). Declare its buffers with `ClipmapProbes.bindingsWGSL(group, first)` in the material's
 * own group and bind them with `ClipmapProbes.bindGroupEntries`. Rust: `gi::CLIPMAP_PROBES_WGSL`.
 */
export const CLIPMAP_PROBES_WGSL: string = clipmapProbes;

/** One mip of a 3D chain (`Mip3d`). */
export const MIP3D_WGSL: string = mip3d;
/** The six directional mip chains (`AnisotropicMips`). */
export const ANISO_MIP_WGSL: string = anisoMip;
/** `ParticleVoxelizer`'s splat. */
export const SPLAT_WGSL: string = assemble([voxelVolume, particleEmission, particleSplat]);
/** `ParticleVoxelizer`'s resolve. */
export const RESOLVE_WGSL: string = assemble([voxelVolume, skyLighting, particleResolve]);
/** `ParticleConeShading`'s gather. */
export const PARTICLE_CONES_WGSL: string = assemble([voxelVolume, voxelCones, sdf, particleEmission, skyLighting, particleCones]);
/** `MeshVoxelizer`'s fragment stage for materials without a `voxelFragmentEntry`. */
export const VOXEL_FRAGMENT_WGSL: string = assemble([voxelWrite, voxelFragment]);
/** `VoxelInjection`'s pass: the voxelized surfaces into light (Rust `gi::inject::INJECT_WGSL`). */
export const INJECT_WGSL: string = assemble([voxelVolume, voxelCones, voxelIrradiance, sdf, particleEmission, skyLighting, COMPUTE_SHADOWS_WGSL, inject]);
/** `JumpFloodSdf`'s seed, flood and distance passes (Rust `gi::sdf::JUMP_FLOOD_WGSL`). */
export const JUMP_FLOOD_WGSL: string = jumpFlood;
/** `SdfProbes`' update (Rust `gi::probes::PROBE_UPDATE_WGSL`). */
export const PROBE_UPDATE_WGSL: string = assemble([voxelVolume, voxelCones, voxelIrradiance, skyLighting, probeCommon, probeUpdate]);
/** `VoxelGIEffect`'s trace through a volume (Rust `gi::effect::TRACE_WGSL`). */
export const SCREEN_TRACE_WGSL: string = assemble([screenCommon, screenNormal, voxelVolume, voxelCones, voxelIrradiance, skyLighting, screenTrace]);
/** `VoxelGIEffect`'s temporal filter. */
export const SCREEN_TEMPORAL_WGSL: string = assemble([screenCommon, screenTemporal]);
/** `VoxelGIEffect`'s composite (`main`, and `main_probes` with probes as the far field). */
export const SCREEN_COMPOSITE_WGSL: string = assemble([screenCommon, screenNormal, voxelVolume, sdf, probeCommon, probeIrradiance, skyLighting, screenComposite]);
/** `ClipmapVoxelizer`'s clear of a region (Rust `gi::clipmap_voxelize::CLEAR_WGSL`). */
export const CLIPMAP_CLEAR_WGSL: string = clipmapClear;
/** `ClipmapInjection`'s pass (Rust `gi::clipmap_inject::CLIPMAP_INJECT_WGSL`). */
export const CLIPMAP_INJECT_WGSL: string = assemble([clipmap, particleEmission, skyLighting, COMPUTE_SHADOWS_WGSL, clipmapInject]);
/** `ClipmapProbes`' update (Rust `gi::clipmap_probes::CLIPMAP_PROBE_UPDATE_WGSL`). */
export const CLIPMAP_PROBE_UPDATE_WGSL: string = assemble([clipmap, particleEmission, clipmapProbes, skyLighting, clipmapProbeUpdate]);
/** `VoxelGIEffect`'s trace through a clipmap, and its `show_voxels` (Rust `gi::effect::CLIPMAP_TRACE_WGSL`). */
export const CLIPMAP_TRACE_WGSL: string = assemble([screenCommon, screenNormal, clipmap, skyLighting, clipmapTrace]);
/** `VoxelGIEffect`'s trace from a clipmap's probes (Rust `gi::effect::CLIPMAP_PROBE_TRACE_WGSL`). */
export const CLIPMAP_PROBE_TRACE_WGSL: string = assemble([screenCommon, screenNormal, skyLighting, clipmapProbes, clipmapProbeTrace]);
