//! GPU culling: per-view frustum and LOD culling of instanced renderables, compacted into
//! indirect draws, so every view (the camera, each shadow map) draws what it sees; and the
//! hierarchical depth pyramid occlusion tests read.

mod depth_pyramid;
mod instance_culling;

pub use depth_pyramid::{DepthPyramid, DepthReduction};
pub use instance_culling::{frustum_planes, InstanceCulling};
pub(crate) use instance_culling::{CullPipeline, CullView};
