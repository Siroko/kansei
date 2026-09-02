//! Additive glyph attractor for the fluid sim: pulls tagged particles into the
//! extruded glyph SDF volumes (from `crate::sdf`). Runs as its own compute pass
//! after the SPH solver; modifies velocities only, leaving the solver untouched.

use crate::sdf::GlyphVolumeSet;

/// Number of glyphs packed into the atlas (`0`–`9` and `:`).
const ATLAS_GLYPHS: u32 = 11;

/// All 11 glyph SDF volumes flattened into a single 3D field of size
/// `(res_xy, res_xy, res_z * 11)`, stacked along Z. Glyph `g` occupies
/// depth `[g*res_z, (g+1)*res_z)`. Missing glyphs are filled with `-1.0`
/// (fully outside), so no particle is ever attracted to an absent glyph.
pub struct GlyphVolumeAtlas {
    pub res_xy: u32,
    pub res_z: u32,
    pub glyph_count: u32,
    /// `res_xy * res_xy * (res_z * glyph_count)` f32 SDF values, +inside.
    pub data: Vec<f32>,
}

impl GlyphVolumeAtlas {
    /// Total depth of the packed texture (`res_z * glyph_count`).
    pub fn depth(&self) -> u32 {
        self.res_z * self.glyph_count
    }

    /// Flatten a `GlyphVolumeSet` into the packed atlas.
    pub fn from_set(set: &GlyphVolumeSet) -> GlyphVolumeAtlas {
        let res_xy = set.res_xy;
        let res_z = set.res_z;
        let glyph_count = ATLAS_GLYPHS;
        let slab = (res_xy * res_xy * res_z) as usize;
        let mut data = vec![-1.0f32; slab * glyph_count as usize];

        for (g, vol) in set.volumes().iter().enumerate() {
            if let Some(v) = vol {
                let base = g * slab;
                // v.data is already ((z*res_xy)+y)*res_xy + x, same order as a slab.
                data[base..base + slab].copy_from_slice(&v.data);
            }
        }
        GlyphVolumeAtlas { res_xy, res_z, glyph_count, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::{FontAtlas, GlyphVolumeSet};

    const FONT: &[u8] = include_bytes!("../../../tests/fixtures/L10-medium.arfont");

    #[test]
    fn atlas_dimensions_and_layout() {
        let font = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);
        let atlas = GlyphVolumeAtlas::from_set(&set);
        assert_eq!(atlas.res_xy, 32);
        assert_eq!(atlas.res_z, 8);
        assert_eq!(atlas.glyph_count, 11);
        assert_eq!(atlas.depth(), 8 * 11);
        assert_eq!(atlas.data.len(), (32 * 32 * 8 * 11) as usize);
    }

    #[test]
    fn digit_slab_matches_source_volume() {
        let font = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);
        let atlas = GlyphVolumeAtlas::from_set(&set);
        // Digit '3' is glyph index 3; its slab in the atlas must equal its source volume.
        let three = set.volume_for_digit(3).unwrap();
        let slab = (32 * 32 * 8) as usize;
        let base = 3 * slab;
        assert_eq!(&atlas.data[base..base + slab], &three.data[..]);
    }
}
