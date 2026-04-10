use super::geometry::{Geometry, Vertex};

/// A flat plane in the XY plane, centered at origin.
pub struct PlaneGeometry;

impl PlaneGeometry {
    pub fn new(width: f32, height: f32) -> Geometry {
        let hw = width * 0.5;
        let hh = height * 0.5;

        let vertices = vec![
            Vertex { position: [-hw, -hh, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0] },
            Vertex { position: [ hw, -hh, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0] },
            Vertex { position: [ hw,  hh, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0] },
            Vertex { position: [-hw,  hh, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] },
        ];

        let indices = vec![0, 1, 2, 0, 2, 3];

        Geometry::new("PlaneGeometry", vertices, indices)
    }
}
