use std::collections::HashMap;

use super::geometry::{Geometry, Vertex};

/// A sphere of near-equal triangles: an icosahedron whose faces are split in four
/// `subdivisions` times (20 · 4ⁿ triangles), its vertices pushed out to `radius`. Unlike
/// `SphereGeometry` it has no poles, so it suits displacement and cluster LOD.
pub struct IcosphereGeometry;

impl IcosphereGeometry {
    pub fn new(radius: f32, subdivisions: u32) -> Geometry {
        let t = (1.0 + 5f32.sqrt()) / 2.0;
        let mut points: Vec<glam::Vec3> = [
            [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
            [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
            [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
        ]
        .iter()
        .map(|v| glam::Vec3::from(*v).normalize())
        .collect();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        for _ in 0..subdivisions {
            let mut midpoints = HashMap::new();
            let mut next = Vec::with_capacity(faces.len() * 4);
            for [a, b, c] in faces {
                let mut mid = |x: u32, y: u32| {
                    *midpoints.entry((x.min(y), x.max(y))).or_insert_with(|| {
                        points.push(((points[x as usize] + points[y as usize]) * 0.5).normalize());
                        points.len() as u32 - 1
                    })
                };
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            faces = next;
        }
        let vertices = points
            .iter()
            .map(|&n| {
                let p = n * radius;
                Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: [n.x * 0.5 + 0.5, n.y * 0.5 + 0.5] }
            })
            .collect();
        Geometry::new("IcosphereGeometry", vertices, faces.into_iter().flatten().collect())
    }
}
