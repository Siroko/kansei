//! Post-processing: a [`PostProcessingVolume`] runs a chain of compute effects over the GBuffer
//! and blits the last one to the surface.
//!
//! Effects up to the tonemapper work on scene-linear HDR light; the tonemapper turns it into the
//! display signal. A physically ordered chain:
//!
//! 1. the sky and aerial perspective ([`effects::AtmosphereEffect`]), then the far height fog
//!    ([`effects::HeightFogEffect`]);
//! 2. volumetric fog (composited over the lit scene by depth);
//! 3. anti-aliasing, while samples are still linear ([`effects::TemporalAAEffect`], which also
//!    upscales to the display size under `Renderer::set_render_scale`; what follows runs at it);
//! 4. motion blur ([`effects::MotionBlurEffect`]), on the anti-aliased frame (the TAA's history
//!    stays sharp) while its colour still lines up with the depth and velocity;
//! 5. depth of field (lens blur of scene light);
//! 6. bloom (scattering in the lens, on scene light);
//! 7. [`effects::ToneMapEffect`] (exposure, lens vignetting and fringing, grade, tone curve, grain,
//!    output encoding);
//! 8. display-space tweaks such as [`effects::ColorGradingEffect`].

pub mod effects;

mod effect;
mod volume;

pub use effect::PostProcessingEffect;
pub use volume::PostProcessingVolume;
// Re-export GBuffer from renderers since it's tightly coupled
pub use crate::renderers::GBuffer;
