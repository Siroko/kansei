use super::geometry::{Geometry, Vertex};

/// An axis-aligned box centered at origin.
pub struct BoxGeometry;

impl BoxGeometry {
    pub fn new(width: f32, height: f32, depth: f32) -> Geometry {
        let hw = width * 0.5;
        let hh = height * 0.5;
        let hd = depth * 0.5;

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            // +Z
            ([0.0, 0.0, 1.0], [[-hw,-hh, hd],[ hw,-hh, hd],[ hw, hh, hd],[-hw, hh, hd]]),
            // -Z
            ([0.0, 0.0,-1.0], [[ hw,-hh,-hd],[-hw,-hh,-hd],[-hw, hh,-hd],[ hw, hh,-hd]]),
            // +Y
            ([0.0, 1.0, 0.0], [[-hw, hh, hd],[ hw, hh, hd],[ hw, hh,-hd],[-hw, hh,-hd]]),
            // -Y
            ([0.0,-1.0, 0.0], [[-hw,-hh,-hd],[ hw,-hh,-hd],[ hw,-hh, hd],[-hw,-hh, hd]]),
            // +X
            ([1.0, 0.0, 0.0], [[ hw,-hh, hd],[ hw,-hh,-hd],[ hw, hh,-hd],[ hw, hh, hd]]),
            // -X
            ([-1.0,0.0, 0.0], [[-hw,-hh,-hd],[-hw,-hh, hd],[-hw, hh, hd],[-hw, hh,-hd]]),
        ];

        let uvs = [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]];

        for (normal, corners) in &faces {
            let base = vertices.len() as u32;
            for (i, pos) in corners.iter().enumerate() {
                vertices.push(Vertex {
                    position: [pos[0], pos[1], pos[2], 1.0],
                    normal: *normal,
                    uv: uvs[i],
                });
            }
            indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        }

        Geometry::new("BoxGeometry", vertices, indices)
    }
}
