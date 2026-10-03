//! Voxel global illumination: a voxel volume of the scene's light that producers write and
//! consumers cone trace, after Hector Arellano's articles on miaumiau.cat (p=1457, mesh
//! voxelization and distance fields; p=1476, indirect light on particles) and Crassin et al.
//! 2011, on WebGPU compute.
//!
//! The shared core:
//! - [`VoxelVolume`]: an `rgba16float` 3D texture with mips (premultiplied radiance, opacity),
//!   its placement as a uniform (`VOXEL_VOLUME_WGSL`) and its sampler; [`Mip3d`] builds the mips;
//!   [`VoxelGiQuality`] sets the resolution and cost, stepping down to what a device can hold;
//! - `VOXEL_CONES_WGSL`: `voxelConeTrace`, for any pass or material that reads the volume, with
//!   the sky (`SKY_LIGHTING_WGSL`'s `skyRadiance`, or [`gradient_sky_lighting`] without an
//!   atmosphere) as the light past it.
//!
//! Producers write mip 0; today, particles and analytic boxes ([`ParticleVoxelizer`]). Consumers
//! today: per-particle cones ([`ParticleConeShading`]), and [`ParticleGi`] runs the three in
//! order. The planned producers (a raster voxelizer drawing meshes through their own
//! `vertex_main`, with a per-renderable albedo and emission by default and optionally a
//! material's textured albedo; light injection with shadow maps) and consumers (screen-space
//! cones, a jump-flood distance field, probes traced in it under a screen-space near field)
//! reuse the volume, the mips and the cone library.

mod cones;
mod particle_gi;
mod particles;
mod volume;

pub use cones::{gradient_sky_lighting, ParticleConeSettings, ParticleConeShading, SkyLightingData, PARTICLE_LIGHTING_STRIDE};
pub use particle_gi::{ParticleGi, ParticleGiOptions, ParticleGiSettings};
pub use particles::{GiBox, ParticleEmission, ParticleSplatSettings, ParticleVoxelizer, MAX_GI_BOXES};
pub use volume::{Mip3d, VolumeLayout, VoxelGiQuality, VoxelVolume};

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
        ] {
            validate(name, code, &mut sizes);
        }
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
