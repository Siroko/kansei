//! Crop per-glyph SDF from a `FontAtlas` and extrude to a 3D volume.

use crate::sdf::{FontAtlas, GlyphMetrics};

/// A single glyph's SDF cropped to a fixed square resolution, values in [-1, 1]
/// where positive is inside the glyph.
pub struct GlyphSdf2d {
    pub res: u32,
    /// `res * res` signed values, row-major, +inside / -outside.
    pub data: Vec<f32>,
}

/// Sample the atlas alpha (SDF) for `glyph`, resampled to `res × res`.
/// Atlas alpha stores SDF as unsigned [0,255] with 0.5 (=127.5) at the outline;
/// we remap to signed [-1, 1] (+inside).
pub fn crop_glyph_sdf(atlas: &FontAtlas, glyph: &GlyphMetrics, res: u32) -> GlyphSdf2d {
    let [l, b, r, t] = glyph.image_bounds;
    let mut data = vec![0.0f32; (res * res) as usize];
    let aw = atlas.width as f32;
    let ah = atlas.height as f32;
    for y in 0..res {
        for x in 0..res {
            // Map output cell to atlas pixel (bilinear-nearest is fine for the field).
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            let ax = (l + u * (r - l)).clamp(0.0, aw - 1.0);
            // image_bounds y is bottom-up; atlas rows are top-down.
            let ay = (ah - (b + v * (t - b))).clamp(0.0, ah - 1.0);
            let idx = ((ay as u32 * atlas.width + ax as u32) * 4 + 3) as usize;
            let alpha = atlas.rgba[idx] as f32 / 255.0;
            data[(y * res + x) as usize] = (alpha - 0.5) * 2.0; // [-1,1], +inside
        }
    }
    GlyphSdf2d { res, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::FontAtlas;

    const FONT: &[u8] = include_bytes!("../../tests/fixtures/L10-medium.arfont");

    #[test]
    fn glyph_center_is_inside_edges_outside() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let zero = atlas.glyphs.iter().find(|g| g.codepoint == '0' as u32).unwrap();
        let sdf = crop_glyph_sdf(&atlas, zero, 32);
        // For '0', a point on the left stroke should be inside; the very center is the hole (outside).
        let at = |x: u32, y: u32| sdf.data[(y * 32 + x) as usize];
        assert!(at(4, 16) > 0.0 || at(28, 16) > 0.0, "a stroke sample should be inside the glyph");
        assert!(at(0, 0) < 0.0, "the corner should be outside the glyph");
    }
}
