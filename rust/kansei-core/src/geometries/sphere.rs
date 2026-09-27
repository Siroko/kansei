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
                // counter-clockwise seen from outside, the front face with back-face culling
                indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
            }
        }

        Geometry::new("SphereGeometry", vertices, indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::geometries::{BoxGeometry, Geometry, PlaneGeometry, SphereGeometry};

    /// Every triangle winds counter-clockwise seen from the side its vertex normals face, which is
    /// the front face of kansei's pipelines (wgpu's default) and survives back-face culling.
    fn assert_front_faces_out(g: &Geometry) {
        for (i, t) in g.indices.chunks_exact(3).enumerate() {
            let p = |k: usize| glam::Vec3::from_slice(&g.vertices[t[k] as usize].position[..3]);
            let face = (p(1) - p(0)).cross(p(2) - p(0));
            if face.length_squared() < 1e-12 {
                continue; // degenerate triangles at the poles
            }
            let n: glam::Vec3 = t.iter().map(|&v| glam::Vec3::from(g.vertices[v as usize].normal)).sum();
            assert!(face.dot(n) > 0.0, "{}: triangle {i} faces inward", g.label);
        }
    }

    #[test]
    fn geometries_wind_counter_clockwise_outward() {
        assert_front_faces_out(&SphereGeometry::new(1.0, 16, 8));
        assert_front_faces_out(&BoxGeometry::new(1.0, 2.0, 3.0));
        assert_front_faces_out(&PlaneGeometry::new(2.0, 2.0));
    }
}
