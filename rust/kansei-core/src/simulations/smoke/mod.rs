//! Volumetric smoke and steam: boxes of gas at three levels of motion, rendered by the
//! volumetric fog (`VolumetricFogEffect::set_smoke`), lit by its lights, shadow maps and sky and
//! composited with it in one march, so smoke and fog read as one medium.
//!
//! - **Waves** ([`SmokeMotion::Waves`]): no state; a plume from each source (or hanging layers)
//!   swayed by sums of sines over height and time, evaluated where the march samples it. Nearly
//!   free: background wisps, many cheap sources.
//! - **Curl noise** ([`SmokeMotion::CurlNoise`]): a density grid advected by curl noise
//!   (Bridson, Hourihan and Nordenstam 2007, "Curl-Noise for Procedural Fluid Flow"), divergence
//!   free, and a rise: swirly, turbulent motion with nothing pushing back.
//! - **Fluid** ([`SmokeMotion::Fluid`]): a grid smoke solver after Stam 1999 ("Stable Fluids")
//!   and Fedkiw, Stam and Jensen 2001 ("Visual Simulation of Smoke"): semi-Lagrangian velocity on
//!   a MAC grid, buoyancy from temperature, vorticity confinement, a Jacobi pressure projection,
//!   the box's walls closed; density and temperature advected by MacCormack with Selle et al.
//!   2008's clamp ("An Unconditionally Stable MacCormack Method").
//!
//! Dense smoke scatters with Wrenninge's multiple-scattering octaves (Wrenninge 2013,
//! "Production Volume Rendering"), written so thin smoke scatters as the fog does.
//!
//! The app owns and steps the simulation, as it does a `FluidSimulation`:
//!
//! ```ignore
//! let mut smoke = SmokeSimulation::new(renderer.device(), renderer.queue());
//! let plume = smoke.add(SmokeVolume::rising_from(SmokeSource { position: cup_top, ..Default::default() }, 0.8, SmokeMotion::fluid()));
//! fog.set_smoke(Some(&smoke.handle()));
//! // each frame
//! for _ in 0..fixed.advance(dt) { smoke.step(fixed.step as f32, &mouse) }
//! ```

mod mouse;
mod simulation;
mod solver;
mod volume;

pub use mouse::SmokeMouse;
pub use simulation::{SmokeGridTextures, SmokeHandle, SmokeSimulation, SmokeVolumeId};
pub use volume::{CurlSettings, FluidSmokeSettings, SmokeFill, SmokeLook, SmokeMotion, SmokeSource, SmokeVolume, WavesSettings};

#[allow(unused_imports)]
pub(crate) use simulation::{SmokeHandleVolume, SmokeShared};
pub(crate) use volume::SmokeVolumeGpu;

/// Volumes a simulation holds, and the fog marches at once.
pub const MAX_SMOKE_VOLUMES: usize = 4;
/// Sources a volume emits from.
pub const MAX_SMOKE_SOURCES: usize = 4;
/// Recent mouse pushes the Waves level keeps.
pub const MAX_SMOKE_PUSHES: usize = 8;
/// Cells along a volume's longest side, at least and at most.
pub const MIN_SMOKE_RESOLUTION: u32 = 16;
pub const MAX_SMOKE_RESOLUTION: u32 = 160;

/// Curl noise in WGSL, self-contained: `fn curlNoisePotential(p: vec3f, time: f32) -> vec3f` and
/// its curl, `fn curlNoise(p: vec3f, time: f32) -> vec3f`, a divergence-free velocity of about
/// unit speed whose eddies are about a unit across.
pub const CURL_NOISE_WGSL: &str = include_str!("shaders/curl_noise.wgsl");

/// Curl noise and the volumes' data and box maths, for every smoke shader.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const SMOKE_COMMON_WGSL: &str = concat!(include_str!("shaders/curl_noise.wgsl"), include_str!("shaders/smoke_common.wgsl"));
const FILL_WGSL: &str = concat!(include_str!("shaders/curl_noise.wgsl"), include_str!("shaders/smoke_common.wgsl"), include_str!("shaders/smoke_fill.wgsl"));
const SOLVER_WGSL: &str = concat!(include_str!("shaders/curl_noise.wgsl"), include_str!("shaders/smoke_common.wgsl"), include_str!("shaders/smoke_solver.wgsl"));

pub(crate) const MOTION_WAVES: u32 = 0;
pub(crate) const MOTION_CURL: u32 = 1;
pub(crate) const MOTION_FLUID: u32 = 2;

#[cfg(test)]
pub(crate) fn shader_sources() -> Vec<(&'static str, &'static str)> {
    vec![("smoke_common", SMOKE_COMMON_WGSL), ("smoke_fill", FILL_WGSL), ("smoke_solver", SOLVER_WGSL)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate every smoke WGSL module with naga, and check that the Rust GPU structs have
    /// exactly the size of their WGSL counterparts.
    #[test]
    fn shaders_validate_and_the_structs_match() {
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in shader_sources() {
            let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (_, ty) in module.types.iter() {
                if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                    sizes.insert(n.clone(), *span as usize);
                }
            }
        }
        assert_eq!(sizes["SmokeVolumeData"], std::mem::size_of::<SmokeVolumeGpu>());
        assert_eq!(sizes["SmokeVolumeData"], 256);
        assert_eq!(sizes["SmokeStep"], std::mem::size_of::<simulation::SmokeStepGpu>());
        assert!(sizes["SmokeStep"] as u64 <= simulation::STEP_SLOT);
        assert_eq!(sizes["SmokeFillParams"], std::mem::size_of::<simulation::SmokeFillGpu>());
    }
}
