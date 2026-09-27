//! GPU culling: per-view frustum and LOD culling of instanced renderables, compacted into
//! indirect draws, so every view (the camera, each shadow map) draws what it sees.

mod instance_culling;

pub use instance_culling::{frustum_planes, InstanceCulling};
pub(crate) use instance_culling::{CullPipeline, CullView};
