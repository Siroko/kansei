pub(crate) mod camera;

pub use camera::Camera;

/// WGSL for materials that write motion vectors (TAA): the camera's temporal uniform (group 1,
/// binding 3: unjittered and previous view-projection, jitter), `KanseiMeshTransforms` (group 2,
/// binding 1: world and previous world matrix) and `kansei_motion_vector`.
pub const MOTION_VECTORS_WGSL: &str = include_str!("../shaders/motion_vectors.wgsl");
