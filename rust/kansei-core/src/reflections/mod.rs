//! Reflections: planar mirror views of the scene for materials to sample (K9).

mod planar_reflection;

pub use planar_reflection::{oblique_near_plane, reflection_matrix, PlanarReflection, PlanarReflectionOptions};

/// WGSL for sampling a [`PlanarReflection`] in a material: `kansei_screen_uv`,
/// `kansei_reflection_offset` (ripples) and `kansei_planar_reflection` (roughness picks the mip).
pub const PLANAR_REFLECTION_WGSL: &str = include_str!("../shaders/planar_reflection_sample.wgsl");
