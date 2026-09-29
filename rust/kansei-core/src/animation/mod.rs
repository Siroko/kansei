//! Skeletal animation: skeletons, poses and clips on quaternion joint transforms, glTF import of
//! skins and animations, and vertex-shader skinning.
//!
//! - `Skeleton` (joints parents first, rest pose), `Pose` (local transforms; `to_model` is the
//!   forward kinematics), `Clip` (every joint's transform at a fixed rate, sampled with nlerp).
//! - `SkinnedGltf` imports a glTF's skeleton, skinned meshes and animations.
//! - Skinning runs in the vertex shader by vertex pulling (`SKINNING_WGSL`): the geometry keeps
//!   the standard vertex layout, and a skinned material reads the bone palette and each vertex's
//!   joints and weights from its own storage buffers (group 0). Shadow, reflection and velocity
//!   passes redraw materials through their `vertex_main`, so they see the skinned mesh too; the
//!   palette also holds last frame's matrices, for motion vectors. Mark skinned renderables
//!   `dynamic`, update a `BonePalette` each frame and upload it into the material's
//!   `bindable_buffer(PALETTE_BINDING)`.
//! - The single directional shadow map (`Renderer::enable_shadows`) draws every caster with one
//!   shared depth shader, so a skinned mesh casts its bind pose there; cascaded shadows, spot
//!   shadows and sky occlusion use the material's own vertex stage.
//! - Cluster LOD (`Renderable::clusters`) cannot feed `vertex_index`; skinned renderables keep
//!   their geometry.

mod clip;
mod gltf;
pub mod ik;
pub mod inertialization;
pub mod motion_matching;
pub mod retarget;
mod pose;
mod skeleton;
mod skin;
mod skinning;
pub mod springs;
mod transform;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod gpu_tests;

pub use clip::Clip;
pub use gltf::SkinnedGltf;
pub use pose::Pose;
pub use skeleton::Skeleton;
pub use skin::{strongest_influences, SkinnedMesh, MAX_INFLUENCES};
pub use skinning::{skin_buffer, skinned_lit_material, skinned_material, BonePalette, SkinnedLitParams, PALETTE_BINDING, SKINNED_LIT_WGSL, SKINNING_WGSL, SKIN_BINDING};
pub use transform::{angular_velocity, nlerp, quat_abs, quat_exp, quat_from_scaled_angle_axis, quat_log, quat_to_scaled_angle_axis, Transform};
