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
//!   atmosphere) as the light past it.
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
//! Consumers: the per-particle cones, and [`VoxelGIEffect`], diffuse GI on screen from cones
//! traced per pixel, optionally under screen-space GI as the near field. The planned ones (a
//! jump-flood distance field, probes traced in it) reuse the volume, the mips and the cones.

mod aniso;
mod cones;
mod effect;
mod inject;
mod particle_gi;
mod particles;
mod scene;
mod volume;
mod voxelize;

pub use cones::{gradient_sky_lighting, ParticleConeSettings, ParticleConeShading, SkyLightingData, PARTICLE_LIGHTING_STRIDE};
pub use particle_gi::{ParticleGi, ParticleGiOptions, ParticleGiSettings};
pub use particles::{GiBox, ParticleEmission, ParticleSplatSettings, ParticleVoxelizer, MAX_GI_BOXES};
pub use volume::{Mip3d, VolumeLayout, VoxelGiQuality, VoxelVolume};
pub use effect::{VoxelGIEffect, VoxelGIOptions};
pub use inject::SceneGiSettings;
pub use scene::{SceneVoxelGi, SceneVoxelGiOptions};
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

/// The WGSL `VoxelVolume` struct and `voxelUvw` / `voxelLinearIndex`: bind
/// `VoxelVolume::uniform` as a `VoxelVolume` uniform.
pub const VOXEL_VOLUME_WGSL: &str = include_str!("shaders/voxel_volume.wgsl");

/// `VOXEL_VOLUME_WGSL` plus `voxelConeTrace(vol, radiance, sampler, origin, dir, tanHalf,
/// startDist, maxDist, maxSteps)`: the scene radiance a cone gathers (rgb) and the transmittance
/// left past it (a). Bind `VoxelVolume::view` as `texture_3d<f32>` and `VoxelVolume::sampler`.
pub const VOXEL_CONES_WGSL: &str = concat!(include_str!("shaders/voxel_volume.wgsl"), include_str!("shaders/voxel_cones.wgsl"));

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
            ("voxel fragment", voxelize::VOXEL_FRAGMENT_WGSL),
            ("inject", inject::INJECT_WGSL),
            ("screen trace", effect::TRACE_WGSL),
            ("screen temporal", effect::TEMPORAL_WGSL),
            ("screen composite", effect::COMPOSITE_WGSL),
        ] {
            validate(name, code, &mut sizes);
        }
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
                "{VOXEL_CONES_WGSL}\n{PARTICLE_EMISSION_WGSL}\n@group(0) @binding(0) var<uniform> vol: VoxelVolume;\n@group(0) @binding(1) var t: texture_3d<f32>;\n@group(0) @binding(2) var s: sampler;\n@group(0) @binding(3) var<uniform> e: ParticleEmission;\n@compute @workgroup_size(1) fn main() {{ _ = voxelConeTrace(vol, t, s, vec3f(0.0), vec3f(0.0, 1.0, 0.0), 1.0, 0.0, 1.0, 4u) + vec4f(particleEmission(e, 0u, vec3f(0.0)), 0.0); }}"
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
        assert_eq!(sizes["VoxelGiParams"], std::mem::size_of::<effect::VoxelGiParamsGpu>());
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
