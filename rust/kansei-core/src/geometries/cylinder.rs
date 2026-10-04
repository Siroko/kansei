use super::geometry::{Geometry, Vertex};

/// A (truncated) cone or cylinder round the y axis, standing on y = 0: radius `radius_bottom` at
/// the bottom, `radius_top` at `height`, in `rings` bands of `segments` quads, with a cap on each
/// end whose radius is not zero. A `radius_top` of 0 makes a cone.
pub struct CylinderGeometry;

impl CylinderGeometry {
    pub fn new(radius_bottom: f32, radius_top: f32, height: f32, segments: u32, rings: u32) -> Geometry {
        let (segments, rings) = (segments.max(3), rings.max(1));
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        for k in 0..=rings {
            let f = k as f32 / rings as f32;
            let (y, r) = (height * f, radius_bottom + (radius_top - radius_bottom) * f);
            for s in 0..=segments {
                let a = s as f32 / segments as f32 * std::f32::consts::TAU;
                let normal = glam::Vec3::new(a.cos() * height, radius_bottom - radius_top, a.sin() * height).normalize();
                vertices.push(Vertex { position: [r * a.cos(), y, r * a.sin(), 1.0], normal: normal.to_array(), uv: [s as f32 / segments as f32, f] });
            }
        }
        let row = segments + 1;
        for k in 0..rings {
            for s in 0..segments {
                let (a, b, c, d) = (k * row + s, k * row + s + 1, (k + 1) * row + s, (k + 1) * row + s + 1);
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        for (y, r, up) in [(0.0, radius_bottom, false), (height, radius_top, true)] {
            if r <= 0.0 {
                continue;
            }
            let normal = [0.0, if up { 1.0 } else { -1.0 }, 0.0];
            let centre = vertices.len() as u32;
            vertices.push(Vertex { position: [0.0, y, 0.0, 1.0], normal, uv: [0.5, 0.5] });
            for s in 0..=segments {
                let a = s as f32 / segments as f32 * std::f32::consts::TAU;
                vertices.push(Vertex { position: [r * a.cos(), y, r * a.sin(), 1.0], normal, uv: [0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()] });
            }
            for s in 0..segments {
                let (rim, next) = (centre + 1 + s, centre + 2 + s);
                indices.extend_from_slice(&if up { [centre, next, rim] } else { [centre, rim, next] });
            }
        }
        Geometry::new("CylinderGeometry", vertices, indices)
    }
}
