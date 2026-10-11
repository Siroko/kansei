//! Caustics: a showcase of ray-traced glass and water and the light they focus, in a plain white
//! studio. Two scenes (`scene=`, a reload):
//!
//! - `box` (the default): the engine's SPH (or PBF) water in a white tank on the studio floor, its
//!   surface drawn as ray-traced glass, its caustics on the tank's bed following the waves, and a
//!   glass sphere resting in it. Drag in the view to push the water along the mouse's motion
//!   (`push`: a column round the point under the cursor made to follow it).
//! - `dragon`: the glass Stanford dragon at full tessellation on the floor, its caustics, soft
//!   shadows and ambient occlusion on a white (or grey) floor that reaches the horizon.
//!
//! Both share the studio (`studio`: the floor, a backdrop as bright as the lit floor far off, the
//! key light as a sun or a spot) and the chain: ray-traced direct light and soft shadows
//! (`RtShadowsEffect`), the hybrid RT GI under the white sky for the bounce and the occlusion (or
//! SSAO, or a flat ambient), the caustics (`RtCausticsEffect`), the reflections and the glass
//! (`RtReflectionsEffect`), TAA, bloom and AgX. Every setting is a URL parameter and a control in
//! the page's panel (`settings`); `?perf=1` or the backquote key shows the performance overlay.

// (the page runs in the browser only: natively, `cargo test` builds the push pass's shader test)
#[cfg(target_arch = "wasm32")]
mod input;
#[cfg(target_arch = "wasm32")]
mod page;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod push;
#[cfg(target_arch = "wasm32")]
mod settings;
#[cfg(target_arch = "wasm32")]
mod studio;
#[cfg(target_arch = "wasm32")]
mod water;

#[cfg(target_arch = "wasm32")]
pub use page::*;
