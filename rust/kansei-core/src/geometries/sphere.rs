use super::geometry::{Geometry, Vertex};

/// A UV sphere centered at origin.
pub struct SphereGeometry;

impl SphereGeometry {
    pub fn new(radius: f32, segments: u32, rings: u32) -> Geometry {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for y in 0..=rings {
            let v = y as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;

            for x in 0..=segments {
                let u = x as f32 / segments as f32;
                let theta = u * std::f32::consts::TAU;

                let nx = theta.cos() * phi.sin();
                let ny = phi.cos();
                let nz = theta.sin() * phi.sin();

                vertices.push(Vertex {
                    position: [nx * radius, ny * radius, nz * radius, 1.0],
                    normal: [nx, ny, nz],
                    uv: [u, v],
                });
            }
        }

        for y in 0..rings {
            for x in 0..segments {
                let a = y * (segments + 1) + x;
                let b = a + segments + 1;
                indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
            }
        }

        Geometry::new("SphereGeometry", vertices, indices)
    }
}
