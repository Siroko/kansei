//! Voxel global illumination: a voxel volume of the scene's light that producers write and
//! consumers cone trace, after Hector Arellano's articles on miaumiau.cat (p=1457, mesh
//! voxelization and distance fields; p=1476, indirect light on particles) and Crassin et al.
//! 2011, on WebGPU compute.
//!
//! The shared core:
//! - [`VoxelVolume`]: an `rgba16float` 3D texture with mips (premultiplied radiance, opacity),
//!   its placement as a uniform (`VOXEL_VOLUME_WGSL`) and its sampler; [`Mip3d`] builds the mips,
//!   and optionally six directional (anisotropic) chains above mip 0, so a cone meets the face of
//!   a wall it reaches first; [`VoxelGiQuality`] sets the resolution and cost, stepping down to
//!   what a device can hold;
//! - `VOXEL_CONES_WGSL`: `voxelConeTrace`, for any pass or material that reads the volume, with
//!   the sky (`SKY_LIGHTING_WGSL`'s `skyRadiance`, or [`gradient_sky_lighting`] without an
//!   atmosphere) as the light past it; `voxelConeTraceSplit` also gives the transmittance at a
//!   near distance (occlusion), and `voxelHemisphereCone` the five cones of a hemisphere.
//!
//! Producers write mip 0:
//! - particles and analytic boxes ([`ParticleVoxelizer`]; [`ParticleGi`] runs it with the mips
//!   and the per-particle cones, [`ParticleConeShading`]);
//! - the scene's meshes ([`SceneVoxelGi`], which the renderer runs: `Renderer::enable_voxel_gi`):
//!   a raster voxelizer draws each renderable with a `Renderable::gi` surface through its own
//!   `vertex_main` ([`MeshVoxelizer`]; a constant [`GiSurface`], or a material's
//!   `voxel_fragment_entry` with `VOXEL_WRITE_WGSL` for textured albedo), then a light injection
//!   lights the voxels through the renderer's shadow maps and bounces last frame's light.
//!
//! Consumers: the per-particle cones; [`VoxelGIEffect`], diffuse GI on screen from cones traced
//! per pixel or from probes, optionally under screen-space GI as the near field; a jump-flood
//! distance field of the voxels ([`JumpFloodSdf`]: soft shadows and AO); and irradiance probes
//! traced in it ([`SdfProbes`], `SceneVoxelGi::enable_probes`), read by materials through
//! `PROBES_WGSL`.
//!
//! Outdoors, one box is either too small or too coarse: [`VoxelClipmap`] holds the light in
//! nested windows around the camera instead, each twice as coarse as the one before, stored
//! toroidally so a window that moves rewrites only the slab it moved into. [`SceneVoxelClipmap`]
//! (`Renderer::enable_voxel_clipmap`) voxelizes the scene into it a region at a time
//! ([`ClipmapVoxelizer`]) and lights it as `SceneVoxelGi` lights its volume, with shadows from
//! cones through the clipmap where the shadow maps don't reach; [`VoxelGIEffect::with_clipmap`]
//! traces it on screen, and `CLIPMAP_WGSL` from any compute pass.

mod aniso;
mod clipmap;
mod clipmap_inject;
mod clipmap_probes;
mod clipmap_scene;
mod clipmap_voxelize;
mod cones;
mod effect;
mod inject;
mod particle_gi;
mod particles;
mod probes;
mod scene;
mod sdf;
mod volume;
mod voxelize;

pub use clipmap::{ClipmapLayout, VoxelClipmap, MAX_CLIPMAP_LEVELS};
pub use clipmap_inject::{ClipmapGiSettings, ConeShadows};
pub use clipmap_probes::{ClipmapProbeOptions, ClipmapProbes};
pub use clipmap_scene::{SceneVoxelClipmap, SceneVoxelClipmapOptions};
pub use clipmap_voxelize::{ClipRegion, ClipmapVoxelizer, CLIP_SURFACE_WORDS};
pub(crate) use clipmap_voxelize::ClipSurfaces;
pub use cones::{gradient_sky_lighting, ParticleConeSettings, ParticleConeShading, SkyLightingData, PARTICLE_LIGHTING_STRIDE};
pub use particle_gi::{ParticleGi, ParticleGiOptions, ParticleGiSettings};
pub use particles::{GiBox, ParticleEmission, ParticleSplatSettings, ParticleVoxelizer, MAX_GI_BOXES};
pub use volume::{Mip3d, VolumeLayout, VoxelGiQuality, VoxelVolume};
pub use effect::{VoxelGIEffect, VoxelGIOptions};
pub use inject::{SceneGiSettings, SdfShadows};
pub use scene::{SceneVoxelGi, SceneVoxelGiOptions};
pub use sdf::{JumpFloodSdf, SdfSeeds};
pub use probes::{SdfProbeOptions, SdfProbes, PROBE_RAYS};
pub use voxelize::{GiSurface, MeshVoxelizer, SURFACE_WORDS_PER_VOXEL};
pub(crate) use voxelize::SurfaceSet;

/// For a material's `voxel_fragment_entry` (`MaterialOptions`): the voxelizer's group 3 and
/// `kansei_voxel_write(fragPos, front, albedo, emission)`, which puts the surface at a fragment
/// into its voxel. Call it in uniform control flow (it takes derivatives):
///
/// ```wgsl
/// @fragment
/// fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
///     let albedo = textureSample(base_texture, base_sampler, in.uv).rgb;
///     kansei_voxel_write(in.clip, front, albedo, vec3f(0.0));
/// }
/// ```
///
/// `kansei_voxel_draw.albedo` and `.emission` hold the renderable's `GiSurface`. Group 3 binds
/// 100-102 here, apart from the shadow group's, so a material may use both chunks.
pub const VOXEL_WRITE_WGSL: &str = voxelize::VOXEL_WRITE_WGSL;

/// `VOXEL_VOLUME_WGSL` plus reading a `JumpFloodSdf`: `sdfDistance(vol, sdf, sampler, p)`,
/// `sdfSoftShadow(vol, sdf, sampler, p, toLight, k, maxT)`, `sdfSurfaceShadow(vol, sdf, sampler,
/// p, n, toLight, k, maxT)` (for a point on a surface of normal `n`: its own plane does not shadow
/// it at grazing angles), `sdfLightShape(vol, radius, distance)` (k and the march's length for a
/// light of that radius and distance) and `sdfAo(vol, sdf, sampler, p, n)`.
/// The material helper for distance-field shadows and AO: bind `JumpFloodSdf::as_texture` as a
/// `texture_3d<f32>`, and the volume's uniform and sampler, in the material's own group.
pub const SDF_WGSL: &str = concat!(include_str!("shaders/voxel_volume.wgsl"), include_str!("shaders/sdf.wgsl"));

/// Diffuse light from the scene's probes (`SdfProbes`) for a material: `ProbeGrid` and
/// `kansei_gi_irradiance(p, n)`, the irradiance a surface at world position `p` facing `n`
/// receives (scene units; a diffuse surface adds albedo / pi times it, in place of its own sky
/// ambient), weighed over the eight probes around `p` as DDGI does. Declare its buffers with
/// `SdfProbes::bindings_wgsl(group, first)` in the material's own group and bind them with
/// `SdfProbes::bind_group_entries` (or as storage and uniform bindables).
pub const PROBES_WGSL: &str = concat!(include_str!("shaders/probe_common.wgsl"), include_str!("shaders/probe_irradiance.wgsl"));

/// The WGSL `VoxelVolume` struct and `voxelUvw` / `voxelLinearIndex`: bind
/// `VoxelVolume::uniform` as a `VoxelVolume` uniform.
pub const VOXEL_VOLUME_WGSL: &str = include_str!("shaders/voxel_volume.wgsl");

/// `VOXEL_VOLUME_WGSL` plus `voxelConeTrace(vol, radiance, sampler, origin, dir, tanHalf,
/// startDist, maxDist, maxSteps)`: the scene radiance a cone gathers (rgb) and the transmittance
/// left past it (a). Bind `VoxelVolume::view` as `texture_3d<f32>` and `VoxelVolume::sampler`.
pub const VOXEL_CONES_WGSL: &str = concat!(include_str!("shaders/voxel_volume.wgsl"), include_str!("shaders/voxel_cones.wgsl"));

/// A voxel clipmap for a compute pass (`VoxelClipmap`): the `VoxelClipmap` uniform and the levels
/// bound in group 0 (binding 50 `VoxelClipmap::uniform`, 51-56 `level_views`, 57 `sampler`);
/// `clipSample(level, p)`, `clipContains`, `clipLevelAt(p, first, margin)`, `clipVoxelSize`;
/// `clipConeTrace(origin, dir, n, tanHalf, minDiameter, startDist, maxDist, maxSteps)`, a cone
/// through the levels (radiance gathered, transmittance left), and `clipIrradiance(sky, skyScale,
/// origin, n, angle, minDiameter, startDist, maxDist, maxSteps)`, a surface's irradiance from six
/// cones and the sky past the clipmap (needs `SKY_LIGHTING_WGSL`).
pub const CLIPMAP_WGSL: &str = include_str!("shaders/clipmap.wgsl");

/// The irradiance probes of a voxel clipmap for a material (`ClipmapProbes`): `ClipProbeGrid`,
/// `kansei_clipmap_light(p, n)` (the irradiance a surface at world position `p` facing `n`
/// receives, scene units, and the cosine-weighted share of its hemisphere that sees the sky past
/// the clipmap; a = -1 where no probe holds `p`) and `kansei_clipmap_sky_visibility(p, n)` (that
/// share alone, 1 where no probe holds `p`: to dim a material's own sky light by). Declare its
/// buffers with `ClipmapProbes::bindings_wgsl(group, first)` in the material's own group and bind
/// them with `ClipmapProbes::bind_group_entries`.
pub const CLIPMAP_PROBES_WGSL: &str = include_str!("shaders/clipmap_probes.wgsl");

/// `ParticleEmission` and `particleEmission(e, index, velocity)`, to tell which particles glow as
/// the GI does (it also hands each particle its emission in the lighting buffer).
pub const PARTICLE_EMISSION_WGSL: &str = include_str!("shaders/particle_emission.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, code: &str, sizes: &mut std::collections::HashMap<String, usize>) {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        // the strictest capabilities, closest to core WebGPU
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for (_, ty) in module.types.iter() {
            if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                let span = *span as usize;
                assert_eq!(*sizes.entry(n.clone()).or_insert(span), span, "{name}: {n} differs between modules");
            }
        }
    }

    #[test]
    fn shaders_validate_and_the_uniforms_match() {
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in [
            ("splat", particles::SPLAT_WGSL),
            ("resolve", particles::RESOLVE_WGSL),
            ("cones", cones::PARTICLE_CONES_WGSL),
            ("mip3d", include_str!("shaders/mip3d.wgsl")),
            ("anisotropic mips", include_str!("shaders/aniso_mip.wgsl")),
            ("jump flood", sdf::JUMP_FLOOD_WGSL),
            ("probe update", probes::PROBE_UPDATE_WGSL),
            ("voxel fragment", voxelize::VOXEL_FRAGMENT_WGSL),
            ("inject", inject::INJECT_WGSL),
            ("clipmap inject", clipmap_inject::CLIPMAP_INJECT_WGSL),
            ("clipmap clear", include_str!("shaders/clipmap_clear.wgsl")),
            ("clipmap trace", effect::CLIPMAP_TRACE_WGSL),
            ("clipmap probe update", clipmap_probes::CLIPMAP_PROBE_UPDATE_WGSL),
            ("clipmap composite", effect::CLIPMAP_COMPOSITE_WGSL),
            ("screen trace", effect::TRACE_WGSL),
            ("screen temporal", effect::TEMPORAL_WGSL),
            ("screen composite", effect::COMPOSITE_WGSL),
        ] {
            validate(name, code, &mut sizes);
        }
        // the distance-field library, as a material uses it
        validate(
            "sdf library",
            &format!(
                "{SDF_WGSL}\n@group(0) @binding(0) var<uniform> vol: VoxelVolume;\n@group(0) @binding(1) var t: texture_3d<f32>;\n@group(0) @binding(2) var s: sampler;\n\
                 @compute @workgroup_size(1) fn main() {{ _ = sdfDistance(vol, t, s, vec3f(0.0)) + sdfSoftShadow(vol, t, s, vec3f(0.0), vec3f(0.0, 1.0, 0.0), 8.0, 10.0) + sdfSurfaceShadow(vol, t, s, vec3f(0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 1.0, 0.0), 8.0, 10.0) + sdfAo(vol, t, s, vec3f(0.0), vec3f(0.0, 1.0, 0.0)) + sdfLightShape(vol, 0.1, 3.0).x; }}"
            ),
            &mut sizes,
        );
        // the probes' library, as a material uses it
        validate(
            "probes library",
            &format!(
                "{PROBES_WGSL}\n{}\n@fragment fn main(@location(0) p: vec3f) -> @location(0) vec4f {{ return vec4f(kansei_gi_irradiance(p, vec3f(0.0, 1.0, 0.0)), 1.0); }}",
                SdfProbes::bindings_wgsl(2, 5)
            ),
            &mut sizes,
        );
        // the clipmap probes' library, as a material uses it
        validate(
            "clipmap probes library",
            &format!(
                "{CLIPMAP_PROBES_WGSL}\n{}\n@fragment fn main(@location(0) p: vec3f) -> @location(0) vec4f {{ return vec4f(kansei_clipmap_light(p, vec3f(0.0, 1.0, 0.0)).rgb * kansei_clipmap_sky_visibility(p, vec3f(0.0, 1.0, 0.0)), 1.0); }}",
                ClipmapProbes::bindings_wgsl(2, 5)
            ),
            &mut sizes,
        );
        // a material with a voxel entry next to its lit fragment: the voxelizer's group 3 and
        // the shadow group's bindings don't collide
        validate(
            "material voxel entry",
            &format!(
                "{}\n{VOXEL_WRITE_WGSL}\n@group(0) @binding(0) var t: texture_2d<f32>;\n@group(0) @binding(1) var s: sampler;\n\
                 struct VOut {{ @builtin(position) clip: vec4f, @location(0) uv: vec2f, @location(1) world: vec3f }};\n\
                 @vertex fn vertex_main(@location(0) p: vec4f) -> VOut {{ var o: VOut; o.clip = p; o.uv = p.xy; o.world = p.xyz; return o; }}\n\
                 @fragment fn fragment_main(in: VOut) -> @location(0) vec4f {{ return vec4f(kansei_spot_lights_radiance(in.world, vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(1.0), 1.0, 0.0, in.clip.xy), 1.0); }}\n\
                 @fragment fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {{ kansei_voxel_write(in.clip, front, textureSample(t, s, in.uv).rgb, vec3f(0.0)); }}",
                crate::lights::SPOT_LIGHTS_WGSL
            ),
            &mut sizes,
        );
        // the libraries alone, with a caller each
        validate(
            "voxel cones library",
            &format!(
                "{VOXEL_CONES_WGSL}\n{PARTICLE_EMISSION_WGSL}\n@group(0) @binding(0) var<uniform> vol: VoxelVolume;\n@group(0) @binding(1) var t: texture_3d<f32>;\n@group(0) @binding(2) var s: sampler;\n@group(0) @binding(3) var<uniform> e: ParticleEmission;\n@compute @workgroup_size(1) fn main() {{ let cone = voxelHemisphereCone(vec3f(0.0, 1.0, 0.0), 2u); let split = voxelConeTraceSplit(vol, t, s, vec3f(0.0), cone.xyz, VOXEL_HEMISPHERE_TAN, 0.0, 0.5, 1.0, 4u); _ = voxelConeTrace(vol, t, s, vec3f(0.0), vec3f(0.0, 1.0, 0.0), 1.0, 0.0, 1.0, 4u) + vec4f(particleEmission(e, 0u, vec3f(0.0)), 0.0) + split.far * split.nearOpen * cone.w; }}"
            ),
            &mut sizes,
        );
        assert_eq!(sizes["VoxelVolume"], std::mem::size_of::<volume::VoxelVolumeGpu>());
        assert_eq!(sizes["SplatParams"], std::mem::size_of::<particles::SplatParamsGpu>());
        assert_eq!(sizes["ResolveParams"], std::mem::size_of::<particles::ResolveParamsGpu>());
        assert_eq!(sizes["GiBox"], std::mem::size_of::<GiBox>());
        assert_eq!(sizes["ParticleEmission"], std::mem::size_of::<ParticleEmission>());
        assert_eq!(sizes["ConeParams"], std::mem::size_of::<cones::ConeParamsGpu>());
        assert_eq!(sizes["SkyLighting"], std::mem::size_of::<SkyLightingData>());
        assert_eq!(sizes["KanseiVoxelizeParams"], std::mem::size_of::<voxelize::VoxelizeParamsGpu>());
        assert_eq!(sizes["KanseiVoxelDraw"], std::mem::size_of::<voxelize::VoxelDrawGpu>());
        assert_eq!(sizes["InjectParams"], std::mem::size_of::<inject::InjectParamsGpu>());
        assert_eq!(sizes["VoxelClipmap"], std::mem::size_of::<clipmap::VoxelClipmapGpu>());
        assert_eq!(sizes["ClipInjectParams"], std::mem::size_of::<clipmap_inject::ClipInjectParamsGpu>());
        assert_eq!(sizes["ClipProbeGrid"], std::mem::size_of::<clipmap_probes::ClipProbeGridGpu>());
        assert_eq!(sizes["ClipProbeUpdate"], std::mem::size_of::<clipmap_probes::ClipProbeUpdateGpu>());
        assert_eq!(sizes["VoxelGiParams"], std::mem::size_of::<effect::VoxelGiParamsGpu>());
        assert_eq!(sizes["SdfParams"], std::mem::size_of::<sdf::SdfParamsGpu>());
        assert_eq!(sizes["ProbeGrid"], std::mem::size_of::<probes::ProbeGridGpu>());
        assert_eq!(sizes["ProbeUpdate"], std::mem::size_of::<probes::ProbeUpdateGpu>());
        assert_eq!(sizes["DirLightData"], std::mem::size_of::<crate::shadows::compute_shadows::DirLightGpu>());
        assert_eq!(sizes["PointLightData"], std::mem::size_of::<crate::shadows::compute_shadows::PointLightGpu>());
    }

    #[test]
    fn the_gradient_sky_runs_from_down_to_up() {
        let sky = gradient_sky_lighting([1.0, 2.0, 3.0], [0.5, 0.0, 0.25]);
        // skyRadiance(d) = 0.282095 sh0 + 0.488603 sh1 d.y (the other bands are zero)
        let radiance = |y: f32, c: usize| 0.282095 * sky[0][c] + 0.488603 * sky[1][c] * y;
        for (c, (up, down)) in [(1.0, 0.5), (2.0, 0.0), (3.0, 0.25)].into_iter().enumerate() {
            assert!((radiance(1.0, c) - up).abs() < 1e-5);
            assert!((radiance(-1.0, c) - down).abs() < 1e-5);
        }
    }
}
