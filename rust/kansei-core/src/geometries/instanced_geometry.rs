use crate::buffers::ComputeBuffer;
use super::geometry::Geometry;

/// An instanced geometry — wraps a base `Geometry` with per-instance
/// `ComputeBuffer`s that step once per instance during draw.
/// Mirrors the TS `InstancedGeometry(baseGeo, count, extraBuffers)` pattern.
///
/// Converts into `Geometry` via `Into<Geometry>`, so `Renderable::new` accepts
/// both plain `Geometry` and `InstancedGeometry` polymorphically.
pub struct InstancedGeometry {
    geometry: Geometry,
}

impl InstancedGeometry {
    pub fn new(mut geometry: Geometry, instance_count: u32, extra_buffers: Vec<ComputeBuffer>) -> Self {
        geometry.instance_count = instance_count;
        geometry.instance_buffers = extra_buffers;
        Self { geometry }
    }
}

impl From<InstancedGeometry> for Geometry {
    fn from(ig: InstancedGeometry) -> Self {
        ig.geometry
    }
}
