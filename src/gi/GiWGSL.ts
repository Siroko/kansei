/**
 * Voxel GI's WGSL, imported from `rust/kansei-core/src/gi/shaders` (Vite `?raw`) as
 * `materials/shaders/SharedWGSL.ts` imports the rest, so both engines run the same source and
 * `cargo test -p kansei-core` validates it. The exported names match the Rust constants
 * (`gi::VOXEL_VOLUME_WGSL`, `gi::VOXEL_CONES_WGSL`, `gi::PARTICLE_EMISSION_WGSL`,
 * `atmosphere::SKY_LIGHTING_WGSL`); the rest are the passes' own, assembled as Rust's `concat!`.
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
