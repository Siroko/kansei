//! GPU culling: per-view frustum and LOD culling of instanced renderables, compacted into
//! indirect draws, so every view (the camera, each shadow map) draws what it sees; two-phase
//! occlusion culling for the camera against a hierarchical depth pyramid; and their statistics.

mod depth_pyramid;
mod instance_culling;
mod occlusion;
mod stats;

pub use depth_pyramid::{DepthPyramid, DepthReduction};
pub use instance_culling::{frustum_planes, InstanceCulling};
pub use stats::{CullStats, CullViewKind, CullingStats};
pub(crate) use instance_culling::{CullPipeline, CulledDraw, CullView, OcclusionView};
pub(crate) use occlusion::Occlusion;
pub(crate) use stats::StatsReadback;
