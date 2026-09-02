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

/// A glyph's SDF extruded into a 3D volume of `res_xy × res_xy × res_z` cells.
/// Values are signed (+inside). Z spans [-1, 1] scaled so `half_depth` is the
/// front/back face of the slab.
pub struct GlyphVolume {
    pub res_xy: u32,
    pub res_z: u32,
    pub half_depth: f32,
    /// Row-major `x + res_xy*(y + res_xy_z_stride)`; index as ((z*res_xy)+y)*res_xy + x.
    pub data: Vec<f32>,
}

impl GlyphVolume {
    /// Extrude a 2D glyph SDF along Z: `sdf3d = min(sdf2d, half_depth - |z|)`.
    /// (Signed convention is +inside, so the slab cap is `half_depth - |z|`.)
    pub fn extrude(
        atlas: &FontAtlas,
        glyph: &GlyphMetrics,
        res_xy: u32,
        res_z: u32,
        half_depth: f32,
    ) -> GlyphVolume {
        let sdf2d = crop_glyph_sdf(atlas, glyph, res_xy);
        let mut data = vec![0.0f32; (res_xy * res_xy * res_z) as usize];
        for z in 0..res_z {
            // z in [-1, 1]
            let zc = if res_z > 1 {
                (z as f32 / (res_z - 1) as f32) * 2.0 - 1.0
            } else {
                0.0
            };
            let cap = half_depth - zc.abs(); // +inside slab
            for y in 0..res_xy {
                for x in 0..res_xy {
                    let s2 = sdf2d.data[(y * res_xy + x) as usize];
                    let s3 = s2.min(cap);
                    data[(((z * res_xy) + y) * res_xy + x) as usize] = s3;
                }
            }
        }
        GlyphVolume { res_xy, res_z, half_depth, data }
    }
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

    #[test]
    fn extrudes_symmetrically_along_z() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let one = atlas.glyphs.iter().find(|g| g.codepoint == '1' as u32).unwrap();
        let vol = GlyphVolume::extrude(&atlas, one, 32, 8, 0.5);
        assert_eq!(vol.res_xy, 32);
        assert_eq!(vol.res_z, 8);
        assert_eq!(vol.data.len(), (32 * 32 * 8) as usize);

        // A cell that is inside the 2D glyph and near mid-depth stays inside;
        // the same (x,y) at the front/back cap is pushed outside by the |z| term.
        let idx = |x: u32, y: u32, z: u32| ((z * 32 + y) * 32 + x) as usize;
        // Find an inside 2D cell.
        let sdf2d = crop_glyph_sdf(&atlas, one, 32);
        let mut inside_xy = None;
        for y in 0..32 { for x in 0..32 {
            if sdf2d.data[(y * 32 + x) as usize] > 0.2 { inside_xy = Some((x, y)); }
        }}
        let (ix, iy) = inside_xy.expect("glyph '1' must have interior cells");
        assert!(vol.data[idx(ix, iy, 4)] > 0.0, "mid-depth interior should be inside");
        assert!(vol.data[idx(ix, iy, 0)] <= vol.data[idx(ix, iy, 4)], "cap should be <= mid");
    }
}
