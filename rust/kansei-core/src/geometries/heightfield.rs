use super::geometry::{Geometry, Vertex};

/// A terrain from a height function: a grid of `cells.0` x `cells.1` quads over the rectangle
/// `min`..`max` (x, z), each vertex at `height(x, z)`, with normals from the function's slope
/// and uvs spanning 0..1. Front faces point up.
pub struct HeightfieldGeometry;

impl HeightfieldGeometry {
    pub fn new(min: [f32; 2], max: [f32; 2], cells: (u32, u32), height: impl Fn(f32, f32) -> f32) -> Geometry {
        let (nx, nz) = (cells.0.max(1), cells.1.max(1));
        let step = [(max[0] - min[0]) / nx as f32, (max[1] - min[1]) / nz as f32];
        // the slope over a fraction of a cell
        let (ex, ez) = (step[0] * 0.25, step[1] * 0.25);
        let mut vertices = Vec::with_capacity(((nx + 1) * (nz + 1)) as usize);
        for j in 0..=nz {
            for i in 0..=nx {
                let (x, z) = (min[0] + i as f32 * step[0], min[1] + j as f32 * step[1]);
                let dx = (height(x + ex, z) - height(x - ex, z)) / (2.0 * ex);
                let dz = (height(x, z + ez) - height(x, z - ez)) / (2.0 * ez);
                let normal = glam::Vec3::new(-dx, 1.0, -dz).normalize();
                vertices.push(Vertex { position: [x, height(x, z), z, 1.0], normal: normal.to_array(), uv: [i as f32 / nx as f32, j as f32 / nz as f32] });
            }
        }
        let at = |i: u32, j: u32| j * (nx + 1) + i;
        let mut indices = Vec::with_capacity((nx * nz * 6) as usize);
        for j in 0..nz {
            for i in 0..nx {
                let (p00, p10, p01, p11) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
                indices.extend_from_slice(&[p00, p01, p10, p10, p01, p11]);
            }
        }
        Geometry::new("HeightfieldGeometry", vertices, indices)
    }
}
