//! Motion matching: animate a character by searching a database of animation frames, every few
//! frames, for the one whose pose and future trajectory best match where the character is and
//! where the input says it is going, then playing on from there.
//!
//! Clean-room, after Simon Clavet's "Motion Matching and The Road to Next-Gen Animation" (GDC
//! 2016), Kristjan Zadziuk's (GDC 2016), and Holden, Kanoun, Perepichka and Popa's "Learned
//! Motion Matching" (SIGGRAPH 2020) with Daniel Holden's MIT-licensed reference implementation
//! (<https://github.com/orangeduck/Motion-Matching>).
//!
//! - `DatabaseBuilder` turns clips (on one skeleton, at one rate) into a `Database`: poses with
//!   quantized rotations, the character root's motion (the root joint's position and heading),
//!   foot contacts, and 27 search features per frame (feet positions and velocities, hips
//!   velocity, the root's future positions and facings at 1/3, 2/3 and 1 s), normalized per group
//!   and weighted, with bounding boxes over runs of 16 and 64 frames.
//! - `Database::search` is a brute-force nearest neighbour over the features that skips runs whose
//!   box is already too far: on the CPU (in WASM, with SIMD when built with `+simd128`), cheap
//!   enough for a few characters each frame.
//! - `MotionMatcher` is the runtime character: spring-simulated trajectory, periodic and forced
//!   searches, root motion, inertialized transitions, the animated character kept near the
//!   simulation, and foot locking with two-joint IK.
//! - `pack` stores a database with its skinned meshes in one binary file (`.kmm`).

mod controller;
mod database;
pub mod pack;
mod search;
pub mod traversal;

pub use controller::{Action, MotionInput, MotionMatcher, MotionMatchingSettings, RootPath, SearchInfo, Simulation};
pub use database::{wrap_angle, yaw_of, yaw_rotation, ClipInfo, ACTION_TAG, ContactThresholds, Database, DatabaseBuilder, FeatureWeights, JointRoles, BOUND_LARGE, BOUND_SMALL, FEATURES, FORWARD, STRIDE, TRAJECTORY_TIMES};
pub use search::{Match, SearchFilter};

#[cfg(test)]
mod tests;
