//! Crop per-glyph SDF from a `FontAtlas` and extrude to a 3D volume.

use crate::sdf::{FontAtlas, GlyphMetrics};

/// A single glyph's SDF cropped to a fixed square resolution, values in [-1, 1]
/// where positive is inside the glyph.
/// data is bottom-up in Y (row 0 = glyph bottom), matching the atlas crop.
pub struct GlyphSdf2d {
    pub res: u32,
    /// `res * res` signed values, row-major, +inside / -outside.
    pub data: Vec<f32>,
}

/// Build a full-range signed distance field for `glyph`, resampled to `res × res`.
///
/// The baked atlas alpha is only a *narrow-band* SDF (a usable gradient exists
/// within a few texels of the outline; beyond that it saturates flat). A flat
/// field gives an attractor no direction, so particles far from a stroke never
/// get pulled onto the glyph. Instead we threshold the atlas alpha into an
/// inside/outside mask and run a signed Euclidean distance transform, producing
/// a field whose gradient points toward the glyph from *anywhere* in the cell.
/// Values are normalized to [-1, 1] (+inside), so the pull reaches the whole box.
pub fn crop_glyph_sdf(atlas: &FontAtlas, glyph: &GlyphMetrics, res: u32) -> GlyphSdf2d {
    crop_glyph_sdf_with_threshold(atlas, glyph, res, 0.5)
}

/// Like [`crop_glyph_sdf`], with an explicit inside threshold on the atlas
/// alpha. 0.5 is the true outline; lower values dilate the strokes (bolder),
/// which is the lever for how much fluid a glyph can hold.
pub fn crop_glyph_sdf_with_threshold(atlas: &FontAtlas, glyph: &GlyphMetrics, res: u32, threshold: f32) -> GlyphSdf2d {
    let [l, b, r, t] = glyph.image_bounds;
    let aw = atlas.width as f32;
    let ah = atlas.height as f32;
    let n = (res * res) as usize;

    // 1. Sample the atlas alpha (bilinear, so a high `res` doesn't inherit the
    //    atlas's pixel staircase) into an inside/outside mask.
    let sample_alpha = |ax: f32, ay: f32| -> f32 {
        let x0 = ax.floor().clamp(0.0, aw - 1.0);
        let y0 = ay.floor().clamp(0.0, ah - 1.0);
        let x1 = (x0 + 1.0).min(aw - 1.0);
        let y1 = (y0 + 1.0).min(ah - 1.0);
        let fx = (ax - x0).clamp(0.0, 1.0);
        let fy = (ay - y0).clamp(0.0, 1.0);
        let a = |x: f32, y: f32| atlas.rgba[((y as u32 * atlas.width + x as u32) * 4 + 3) as usize] as f32 / 255.0;
        let top = a(x0, y0) * (1.0 - fx) + a(x1, y0) * fx;
        let bot = a(x0, y1) * (1.0 - fx) + a(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    };
    let mut inside = vec![false; n];
    for y in 0..res {
        for x in 0..res {
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            // Pixel centers: atlas texel (i, j) covers [i, i+1); sample at -0.5.
            let ax = (l + u * (r - l) - 0.5).clamp(0.0, aw - 1.0);
            // image_bounds y is bottom-up; atlas rows are top-down.
            let ay = (ah - (b + v * (t - b)) - 0.5).clamp(0.0, ah - 1.0);
            inside[(y * res + x) as usize] = sample_alpha(ax, ay) > threshold;
        }
    }

    // 2. Signed Euclidean distance transform: distance to the nearest cell of
    //    opposite membership, signed +inside, normalized so a half-box
    //    distance maps to ~1. Exact separable transform (Felzenszwalb &
    //    Huttenlocher), O(res²) — the old brute force was O(res⁴).
    let norm = (res as f32) * 0.5;
    let d_to_outside = edt_squared(&inside, res, false); // for inside cells
    let d_to_inside = edt_squared(&inside, res, true); // for outside cells
    let mut data = vec![0.0f32; n];
    for i in 0..n {
        let here = inside[i];
        let d2 = if here { d_to_outside[i] } else { d_to_inside[i] };
        let dist = if d2.is_finite() { d2.sqrt() } else { norm };
        let signed = if here { dist } else { -dist };
        data[i] = (signed / norm).clamp(-1.0, 1.0);
    }
    GlyphSdf2d { res, data }
}

/// Squared Euclidean distance from every cell to the nearest cell whose mask
/// value equals `target` (INFINITY if there is none). Cells at distance 0
/// are those already equal to `target`. Felzenszwalb & Huttenlocher's
/// separable lower-envelope transform: rows, then columns.
fn edt_squared(mask: &[bool], res: u32, target: bool) -> Vec<f32> {
    let n = res as usize;
    let inf = f32::INFINITY;
    let mut f = vec![inf; n * n];
    for i in 0..n * n {
        if mask[i] == target {
            f[i] = 0.0;
        }
    }
    let mut d = vec![0.0f32; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f32; n + 1];
    let mut row = vec![0.0f32; n];
    // 1D transform of `row` into `d`.
    let mut dt1d = |row: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]| {
        let mut k = 0usize;
        v[0] = 0;
        z[0] = -inf;
        z[1] = inf;
        for q in 1..n {
            if row[q] == inf {
                continue;
            }
            loop {
                let vk = v[k];
                if row[vk] == inf {
                    // Parabola at vk is absent; replace it.
                    v[k] = q;
                    if k == 0 { z[0] = -inf; z[1] = inf; }
                    break;
                }
                let s = ((row[q] + (q * q) as f32) - (row[vk] + (vk * vk) as f32)) / (2.0 * (q as f32 - vk as f32));
                if s <= z[k] {
                    if k == 0 {
                        v[0] = q;
                        z[0] = -inf;
                        z[1] = inf;
                        break;
                    }
                    k -= 1;
                } else {
                    k += 1;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = inf;
                    break;
                }
            }
        }
        let mut k = 0usize;
        for q in 0..n {
            while z[k + 1] < q as f32 {
                k += 1;
            }
            let vk = v[k];
            d[q] = if row[vk] == inf { inf } else { (q as f32 - vk as f32).powi(2) + row[vk] };
        }
    };
    // Rows.
    for y in 0..n {
        row.copy_from_slice(&f[y * n..(y + 1) * n]);
        dt1d(&row, &mut d, &mut v, &mut z);
        f[y * n..(y + 1) * n].copy_from_slice(&d);
    }
    // Columns.
    for x in 0..n {
        for y in 0..n {
            row[y] = f[y * n + x];
        }
        dt1d(&row, &mut d, &mut v, &mut z);
        for y in 0..n {
            f[y * n + x] = d[y];
        }
    }
    f
}

/// A glyph's SDF extruded into a 3D volume of `res_xy × res_xy × res_z` cells.
/// Values are signed (+inside). Z spans [-1, 1] scaled so `half_depth` is the
/// front/back face of the slab.
/// data is bottom-up in Y (row 0 = glyph bottom), matching the atlas crop.
pub struct GlyphVolume {
    pub res_xy: u32,
    pub res_z: u32,
    pub half_depth: f32,
    /// Row-major; index as ((z*res_xy)+y)*res_xy + x.
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
        Self::extrude_with_threshold(atlas, glyph, res_xy, res_z, half_depth, 0.5)
    }

    /// [`extrude`](Self::extrude) with an explicit inside threshold (see
    /// [`crop_glyph_sdf_with_threshold`]).
    pub fn extrude_with_threshold(
        atlas: &FontAtlas,
        glyph: &GlyphMetrics,
        res_xy: u32,
        res_z: u32,
        half_depth: f32,
        threshold: f32,
    ) -> GlyphVolume {
        let sdf2d = crop_glyph_sdf_with_threshold(atlas, glyph, res_xy, threshold);
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

/// The 11 glyph volumes a clock needs: digits `0`–`9` (indices 0..=9) and `:` (index 10).
pub struct GlyphVolumeSet {
    pub res_xy: u32,
    pub res_z: u32,
    /// 11 volumes; `[0..=9]` = digits, `[10]` = colon. `None` if a glyph was missing.
    volumes: Vec<Option<GlyphVolume>>,
}

impl GlyphVolumeSet {
    /// Build volumes for `'0'..'9'` and `':'`. Missing glyphs yield `None` slots.
    pub fn for_clock(atlas: &FontAtlas, res_xy: u32, res_z: u32, half_depth: f32) -> GlyphVolumeSet {
        Self::for_clock_with_threshold(atlas, res_xy, res_z, half_depth, 0.5)
    }

    /// [`for_clock`](Self::for_clock) with an explicit inside threshold on the
    /// atlas alpha: 0.5 = true outline, lower = bolder strokes.
    pub fn for_clock_with_threshold(atlas: &FontAtlas, res_xy: u32, res_z: u32, half_depth: f32, threshold: f32) -> GlyphVolumeSet {
        let codepoints: Vec<u32> = ('0'..='9').chain([':']).map(|c| c as u32).collect();
        let volumes = codepoints
            .iter()
            .map(|cp| {
                atlas
                    .glyphs
                    .iter()
                    .find(|g| g.codepoint == *cp)
                    .map(|g| GlyphVolume::extrude_with_threshold(atlas, g, res_xy, res_z, half_depth, threshold))
            })
            .collect();
        GlyphVolumeSet { res_xy, res_z, volumes }
    }

    /// Volume for digit `d` (0..=9), or `None` if `d > 9` or the glyph was missing.
    pub fn volume_for_digit(&self, d: u32) -> Option<&GlyphVolume> {
        if d > 9 {
            return None;
        }
        self.volumes.get(d as usize).and_then(|v| v.as_ref())
    }

    /// The colon (`:`) volume, or `None` if it was missing.
    pub fn colon(&self) -> Option<&GlyphVolume> {
        self.volumes.get(10).and_then(|v| v.as_ref())
    }

    /// All 11 volume slots in order: indices 0..=9 are digits, index 10 is `:`.
    /// `None` marks a glyph that was missing from the atlas.
    pub fn volumes(&self) -> &[Option<GlyphVolume>] {
        &self.volumes
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

    #[test]
    fn builds_all_eleven_clock_glyphs() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&atlas, 32, 8, 0.5);
        // Indices 0..=9 are digits; index 10 is ':'.
        for d in 0u32..=9 {
            assert!(set.volume_for_digit(d).is_some(), "missing digit {d}");
        }
        assert!(set.colon().is_some(), "missing colon volume");
        assert_eq!(set.res_xy, 32);
        assert_eq!(set.res_z, 8);
    }

    #[test]
    fn digit_lookup_out_of_range_is_none() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&atlas, 16, 4, 0.5);
        assert!(set.volume_for_digit(10).is_none());
    }

    /// DEBUG (run with --nocapture): print a glyph field as ASCII to verify the
    /// SDF actually carries the digit's shape.
    #[test]
    fn sdf_peak_per_digit() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        for res in [32u32, 64] {
            for c in ('0'..='9').chain([':']) {
                let g = atlas.glyphs.iter().find(|g| g.codepoint == c as u32).unwrap();
                let sdf = crop_glyph_sdf(&atlas, g, res);
                let mx = sdf.data.iter().cloned().fold(f32::MIN, f32::max);
                let inside = sdf.data.iter().filter(|v| **v > 0.0).count();
                println!("res {res} glyph '{c}': max sdf {mx:.3}, inside frac {:.3}", inside as f32 / sdf.data.len() as f32);
            }
        }
    }

    #[test]
    fn edt_matches_brute_force() {
        let res = 24u32;
        let n = (res * res) as usize;
        let mut rng: u64 = 0x1234_5678;
        let mask: Vec<bool> = (0..n)
            .map(|_| {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                (rng >> 33) % 5 == 0
            })
            .collect();
        for target in [true, false] {
            let fast = edt_squared(&mask, res, target);
            let r = res as i32;
            for y in 0..r {
                for x in 0..r {
                    let mut best = f32::INFINITY;
                    for yy in 0..r {
                        for xx in 0..r {
                            if mask[(yy * r + xx) as usize] == target {
                                let d2 = ((x - xx) * (x - xx) + (y - yy) * (y - yy)) as f32;
                                best = best.min(d2);
                            }
                        }
                    }
                    let got = fast[(y * r + x) as usize];
                    assert!((got - best).abs() < 1e-3 || (got.is_infinite() && best.is_infinite()),
                        "mismatch at ({x},{y}) target={target}: fast={got} brute={best}");
                }
            }
        }
    }

    #[test]
    fn dump_glyph_ascii() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let res = 40u32;
        for target in ['2', '5'] {
            let g = atlas.glyphs.iter().find(|g| g.codepoint == target as u32).unwrap();
            let sdf = crop_glyph_sdf(&atlas, g, res);
            println!("--- glyph '{target}' SDF sign map ({res}x{res}), # inside / . near-edge ---");
            for y in (0..res).rev() {
                let mut line = String::new();
                for x in 0..res {
                    let v = sdf.data[(y * res + x) as usize];
                    line.push(if v > 0.0 { '#' } else if v > -0.15 { '.' } else { ' ' });
                }
                println!("|{line}|");
            }
        }
    }
}
