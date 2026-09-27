//! Physically based sky and atmosphere (Hillaire 2020): transmittance, multiple-scattering,
//! sky-view and aerial-perspective LUTs built on the GPU, rendered by
//! [`crate::postprocessing::effects::AtmosphereEffect`], and sky lighting for materials.

pub(crate) mod params;
pub(crate) mod sky_atmosphere;

pub use params::{direction_from_elevation_bearing, AtmosphereParams, CelestialLight};
pub use sky_atmosphere::{SkyAtmosphere, SkyAtmosphereBindings, SkyAtmosphereOptions};

/// The WGSL `Atmosphere` and `SkyFrame` structs and the atmosphere helpers (medium, phase
/// functions, ray-sphere tests, LUT parameterisations), for shaders that read the atmosphere.
/// Such a shader declares `atm : Atmosphere` and, for the frame helpers, `frame : SkyFrame`.
pub const ATMOSPHERE_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/frame.wgsl"));

/// The WGSL `SkyLighting` struct and its helpers, for materials and media lit by the sky:
/// `skyIrradiance(sky, n)`, `skyRadiance(sky, d)` and `skyInscatter(sky, viewDir, g)`. Bind
/// `SkyAtmosphereBindings::sky_lighting` as a uniform of type `SkyLighting`, for example with
/// `ComputeBuffer::from_external(.., BufferType::Uniform)`.
pub const SKY_LIGHTING_WGSL: &str = include_str!("shaders/sky_lighting.wgsl");

/// `skyEnvironment(env, envSampler, r, roughness)`: the prefiltered sky along a reflection, from
/// `SkyAtmosphereBindings::environment` (bind as `texture_cube<f32>`, e.g. `Binding::texture_cube`
/// with `Texture::from_view`) and `environment_sampler`; and `skyEnvironmentBrdf(f0, roughness,
/// n_dot_v)`, the split sum's analytic environment BRDF to scale it by.
pub const SKY_ENVIRONMENT_WGSL: &str = include_str!("shaders/sky_environment.wgsl");
