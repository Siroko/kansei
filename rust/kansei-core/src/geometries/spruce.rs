use super::cylinder::CylinderGeometry;
use super::geometry::Geometry;

/// A spruce 1 high standing on y = 0, for forests (the impostors and occlusion-culling examples
/// place thousands, scaled per instance): a trunk under `cones` stacked cones of `segments` x
/// `rings` quads, each narrower and shorter than the one below, as one merged mesh. The trunk is
/// within 0.04 of the axis and the crowns outside it, so a shader can tell bark from needles by
/// `length(position.xz)`. Fewer segments, rings and cones make its mesh LODs.
pub struct SpruceGeometry;

impl SpruceGeometry {
    pub fn new(segments: u32, rings: u32, cones: u32) -> Geometry {
        let trunk = CylinderGeometry::new(0.035, 0.025, 0.3, segments.min(8), 1);
        let crowns: Vec<(Geometry, f32)> = (0..cones)
            .map(|k| {
                let f = k as f32 / cones as f32;
                let y0 = 0.15 + 0.62 * f;
                let y1 = if k + 1 == cones { 1.0 } else { y0 + 0.42 - 0.12 * f };
                (CylinderGeometry::new(0.24 * (1.0 - 0.55 * f), 0.0, y1 - y0, segments, rings), y0)
            })
            .collect();
        let mut parts = vec![(&trunk, glam::Mat4::IDENTITY)];
        parts.extend(crowns.iter().map(|(cone, y0)| (cone, glam::Mat4::from_translation(glam::Vec3::new(0.0, *y0, 0.0)))));
        Geometry::merged("Spruce", &parts)
    }
}
