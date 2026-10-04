mod geometry;
mod plane;
mod box_geo;
mod sphere;
mod instanced_geometry;
mod heightfield;
mod cylinder;
mod icosphere;
mod spruce;

pub use geometry::{Geometry, Vertex, VertexAttribute};
pub use plane::PlaneGeometry;
pub use box_geo::BoxGeometry;
pub use sphere::SphereGeometry;
pub use instanced_geometry::InstancedGeometry;
pub use heightfield::HeightfieldGeometry;
pub use cylinder::CylinderGeometry;
pub use icosphere::IcosphereGeometry;
pub use spruce::SpruceGeometry;

#[cfg(test)]
mod tests;
