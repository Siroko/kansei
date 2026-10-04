//! The lake: a course of boxes and, east of it, a small lake of SPH water with a water cannon on
//! its bank and a water mill turning in it, under the sun with cascaded shadows and TAA. No
//! animation pack needed.
//!
//! - `lake`: the water, the engine's fluid (`simulations::fluid`) in a container shaped like the
//!   lake, with the bed and the shore in the collision world. It rests when it can (culled out of
//!   view, asleep when settled) and its surface refracts the bed and reflects the sky.
//! - `cannon`: pours new particles into the lake (`FluidNozzle` into the simulation's spare
//!   capacity), raising its level; drained back with R.
//! - `mill`: paddles that move with the wheel as fluid colliders (`FluidCapsule::rigid`).
//! - `course`, `props`: the boxes, the ground, the sky and the props' blockout look.
//!
//! As a library it is the motion-matching example's world: [`World::new`] builds it into a scene,
//! [`World::post_processing`] makes the chain its surface needs, and [`World::update`] steps it
//! with a character's legs, a landing and the cannon's trigger ([`WorldInput`]). A page's state
//! implements [`Host`] and [`register`]s itself, and the exports in `panel` (the P panel's
//! `lake_settings`, `lake_set`, …, the cannon's `cannon_prompt` and `cannon_fire`) act on its
//! world, in this crate's module or in any that depends on it.
//!
//! The page itself (the `demo` feature, on by default: `start`) has no character: an orbit
//! camera, the cannon always ready (E, the gamepad's X, or a click on its prompt; R or Y drains
//! the lake), and the P panel. URL parameters: `course=0`, `lake=0`, `rest=0` (the water never
//! rests), `mill=0` (the mill stands still), `taa=0`, `profile=1` (the renderer's GPU/CPU profile
//! every 3 s), `debug=1` (allows `lake_regions()`, a GPU readback).

pub mod cannon;
pub mod course;
pub mod lake;
pub mod mill;
#[cfg(feature = "demo")]
mod page;
pub mod panel;
pub mod props;
pub mod world;

pub use panel::{register, Host};
pub use world::{renderer, World, WorldInput, WorldOptions};

/// The sun's travel direction, its illuminance (lux) and the sky's luminance (cd/m²).
pub const SUN_DIR: [f32; 3] = [-0.45, -0.6, -0.66];
pub const SUN: [f32; 3] = [80000.0, 72000.0, 60000.0];
pub const SKY: [f32; 3] = [4000.0, 5000.0, 7000.0];
