//! Post-processing: a [`PostProcessingVolume`] runs a chain of compute effects over the GBuffer
//! and blits the last one to the surface.
//!
//! Effects up to the tonemapper work on scene-linear HDR light; the tonemapper turns it into the
//! display signal. A physically ordered chain:
//!
//! 1. volumetric fog (composited over the lit scene by depth);
//! 2. anti-aliasing, while samples are still linear;
//! 3. depth of field (lens blur of scene light);
//! 4. bloom (scattering in the lens, on scene light);
//! 5. [`effects::ToneMapEffect`] (exposure, lens vignetting and fringing, grade, tone curve, grain,
//!    output encoding);
//! 6. display-space tweaks such as [`effects::ColorGradingEffect`].

pub mod effects;

mod effect;
mod volume;

pub use effect::PostProcessingEffect;
pub use volume::PostProcessingVolume;
// Re-export GBuffer from renderers since it's tightly coupled
pub use crate::renderers::GBuffer;
