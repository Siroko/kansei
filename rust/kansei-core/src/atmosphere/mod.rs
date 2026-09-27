//! Physically based sky and atmosphere (Hillaire 2020): transmittance, multiple-scattering and
//! sky-view LUTs built on the GPU, rendered by [`crate::postprocessing::effects::AtmosphereEffect`].

pub(crate) mod params;
pub(crate) mod sky_atmosphere;

pub use params::{direction_from_elevation_bearing, AtmosphereParams, CelestialLight};
pub use sky_atmosphere::{SkyAtmosphere, SkyAtmosphereBindings, SkyAtmosphereOptions};

/// The WGSL `Atmosphere` and `SkyFrame` structs and the atmosphere helpers (medium, phase
/// functions, ray-sphere tests, LUT parameterisations), for shaders that read the atmosphere.
/// Such a shader declares `atm : Atmosphere` and, for the frame helpers, `frame : SkyFrame`.
pub const ATMOSPHERE_WGSL: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/frame.wgsl"));
