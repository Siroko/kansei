//! A tiny CPU painter for the procedural tree and ground-cover textures: antialiased tapered
//! strokes and ellipses that carry colour, coverage and a height (for the normal map), composited
//! front-to-back by height. Everything is deterministic (seeded). From the Raggare intro's
//! renderer (crates/intro/src/trees/canvas.rs in raggare-web, the same author's), unchanged.

use glam::{Vec2, Vec3};

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub color: Vec<Vec3>,
    pub alpha: Vec<f32>,
    pub height: Vec<f32>,
    /// Painting is clipped to this pixel rectangle (x0, y0, x1, y1), exclusive of x1/y1.
    pub clip: (usize, usize, usize, usize),
}

/// Small deterministic PRNG (xorshift32) so generated textures and meshes are stable.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9) | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform in [0, 1).
    pub fn f(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [a, b).
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, color: vec![Vec3::ZERO; w * h], alpha: vec![0.0; w * h], height: vec![0.0; w * h], clip: (0, 0, w, h) }
    }

    fn plot(&mut self, x: usize, y: usize, cov: f32, color: Vec3, height: f32) {
        let (cx0, cy0, cx1, cy1) = self.clip;
        if x < cx0 || y < cy0 || x >= cx1 || y >= cy1 {
            return;
        }
        let i = y * self.w + x;
        // front-to-back by height: a higher element paints over, a lower one only fills gaps
        let over = height >= self.height[i] || self.alpha[i] < 0.5;
        if over {
            self.color[i] = self.color[i].lerp(color, cov);
            if cov > 0.5 {
                self.height[i] = height;
            }
        }
        self.alpha[i] = self.alpha[i].max(cov);
    }

    /// A tapered capsule from `a` (width `wa`) to `b` (width `wb`), in pixels, with a rounded
    /// height profile peaking at `height`.
    pub fn stroke(&mut self, a: Vec2, b: Vec2, wa: f32, wb: f32, color: Vec3, height: f32) {
        let r_max = wa.max(wb) * 0.5 + 1.0;
        let x0 = (a.x.min(b.x) - r_max).floor().max(0.0) as usize;
        let x1 = (a.x.max(b.x) + r_max).ceil().min(self.w as f32 - 1.0) as usize;
        let y0 = (a.y.min(b.y) - r_max).floor().max(0.0) as usize;
        let y1 = (a.y.max(b.y) + r_max).ceil().min(self.h as f32 - 1.0) as usize;
        if x0 > x1 || y0 > y1 {
            return;
        }
        let ab = b - a;
        let len2 = ab.length_squared().max(1e-6);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
                let d = (p - (a + ab * t)).length();
                let r = (wa + (wb - wa) * t) * 0.5;
                let cov = (r - d + 0.5).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                let profile = (1.0 - (d / r.max(0.5)).min(1.0).powi(2)).sqrt();
                self.plot(x, y, cov, color * (0.75 + 0.25 * profile), height + profile * r.min(2.0) * 0.5);
            }
        }
    }

    /// A filled ellipse centred at `c` with half-axes `ra` along `dir` and `rb` across it.
    pub fn ellipse(&mut self, c: Vec2, dir: Vec2, ra: f32, rb: f32, color: Vec3, height: f32) {
        let r = ra.max(rb) + 1.0;
        let (x0, x1) = ((c.x - r).floor().max(0.0) as usize, (c.x + r).ceil().min(self.w as f32 - 1.0) as usize);
        let (y0, y1) = ((c.y - r).floor().max(0.0) as usize, (c.y + r).ceil().min(self.h as f32 - 1.0) as usize);
        if x0 > x1 || y0 > y1 {
            return;
        }
        let d = dir.normalize_or_zero();
        let n = Vec2::new(-d.y, d.x);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - c;
                let (u, v) = (p.dot(d) / ra, p.dot(n) / rb);
                let q = u * u + v * v;
                let edge = (1.0 - q.sqrt()) * ra.min(rb);
                let cov = (edge + 0.5).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                // a leaf: domed, with a faint midrib
                let dome = (1.0 - q).max(0.0).sqrt();
                let rib = (1.0 - (v.abs() * rb).min(1.0)) * 0.3;
                self.plot(x, y, cov, color * (0.8 + 0.2 * dome - rib * 0.3), height + dome * 1.5 - rib);
            }
        }
    }

    /// RGBA8 colour (linear albedo, sRGB-encoded: upload as `Rgba8UnormSrgb`) and a tangent-space normal map
    /// (x along +u, y along +v, z out of the card), both row-major.
    pub fn finish(&self, normal_strength: f32, wrap: bool) -> (Vec<u8>, Vec<u8>) {
        let (w, h) = (self.w, self.h);
        let mut rgba = vec![0u8; w * h * 4];
        let mut nrm = vec![0u8; w * h * 4];
        let at = |x: isize, y: isize| -> f32 {
            let (x, y) = if wrap {
                (x.rem_euclid(w as isize) as usize, y.rem_euclid(h as isize) as usize)
            } else {
                (x.clamp(0, w as isize - 1) as usize, y.clamp(0, h as isize - 1) as usize)
            };
            self.height[y * w + x]
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let c = self.color[i];
                let a = self.alpha[i];
                rgba[i * 4] = srgb8(c.x);
                rgba[i * 4 + 1] = srgb8(c.y);
                rgba[i * 4 + 2] = srgb8(c.z);
                rgba[i * 4 + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
                let (xi, yi) = (x as isize, y as isize);
                let dx = (at(xi + 1, yi) - at(xi - 1, yi)) * 0.5;
                let dy = (at(xi, yi + 1) - at(xi, yi - 1)) * 0.5;
                let n = Vec3::new(-dx * normal_strength, -dy * normal_strength, 1.0).normalize();
                nrm[i * 4] = ((n.x * 0.5 + 0.5) * 255.0).round() as u8;
                nrm[i * 4 + 1] = ((n.y * 0.5 + 0.5) * 255.0).round() as u8;
                nrm[i * 4 + 2] = ((n.z * 0.5 + 0.5) * 255.0).round() as u8;
                nrm[i * 4 + 3] = 255;
            }
        }
        (rgba, nrm)
    }

    /// Dilate colour and height into transparent pixels (a few passes), so bilinear filtering
    /// and mips at card edges pick up foliage colour instead of black.
    pub fn bleed(&mut self, passes: usize) {
        let (w, h) = (self.w, self.h);
        for _ in 0..passes {
            let (c0, h0, a0) = (self.color.clone(), self.height.clone(), self.alpha.clone());
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if a0[i] > 0.0 {
                        continue;
                    }
                    let mut sum = Vec3::ZERO;
                    let mut hs = 0.0;
                    let mut n = 0.0;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                            continue;
                        }
                        let j = yy as usize * w + xx as usize;
                        if a0[j] > 0.0 || c0[j] != Vec3::ZERO {
                            sum += c0[j];
                            hs += h0[j];
                            n += 1.0;
                        }
                    }
                    if n > 0.0 {
                        self.color[i] = sum / n;
                        self.height[i] = hs / n;
                    }
                }
            }
        }
    }
}

fn srgb8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

/// Box-filtered mip chain of an RGBA8 image. With `alpha_test = Some(threshold)`, each level's
/// alpha is rescaled so the fraction of texels passing the alpha test matches level 0 (keeps
/// alpha-tested foliage from thinning out with distance).
pub fn mip_chain(w: usize, h: usize, level0: Vec<u8>, alpha_test: Option<f32>) -> Vec<(usize, usize, Vec<u8>)> {
    let coverage = |data: &[u8], scale: f32, t: f32| {
        let n = data.len() / 4;
        let pass = data.chunks_exact(4).filter(|p| (p[3] as f32 / 255.0 * scale) >= t).count();
        pass as f32 / n as f32
    };
    let target = alpha_test.map(|t| coverage(&level0, 1.0, t));
    let mut levels = vec![(w, h, level0)];
    while levels.last().map(|l| l.0 > 1 || l.1 > 1).unwrap_or(false) {
        let (pw, ph, prev) = levels.last().unwrap();
        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
        let mut next = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut s = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((x * 2 + dx).min(pw - 1), (y * 2 + dy).min(ph - 1));
                        s += prev[(sy * pw + sx) * 4 + c] as u32;
                    }
                    next[(y * nw + x) * 4 + c] = ((s + 2) / 4) as u8;
                }
            }
        }
        if let (Some(t), Some(goal)) = (alpha_test, target) {
            // binary search the alpha scale that restores the coverage
            let (mut lo, mut hi) = (0.5f32, 8.0f32);
            for _ in 0..16 {
                let mid = 0.5 * (lo + hi);
                if coverage(&next, mid, t) < goal { lo = mid } else { hi = mid }
            }
            for p in next.chunks_exact_mut(4) {
                p[3] = ((p[3] as f32) * hi).min(255.0) as u8;
            }
        }
        levels.push((nw, nh, next));
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strokes_cover_and_mips_keep_alpha_test_coverage() {
        let mut c = Canvas::new(64, 64);
        let mut rng = Rng::new(7);
        for _ in 0..40 {
            let a = Vec2::new(rng.range(4.0, 60.0), rng.range(4.0, 60.0));
            c.stroke(a, a + Vec2::new(rng.range(-8.0, 8.0), rng.range(-8.0, 8.0)), 1.5, 1.0, Vec3::new(0.1, 0.3, 0.1), 1.0);
        }
        let (rgba, nrm) = c.finish(1.0, false);
        let covered = rgba.chunks_exact(4).filter(|p| p[3] > 127).count();
        assert!(covered > 100, "{covered}");
        assert!(nrm.chunks_exact(4).all(|p| p[2] >= 127), "normals face out of the card");
        let chain = mip_chain(64, 64, rgba, Some(0.5));
        assert_eq!(chain.len(), 7);
        let frac = |d: &[u8]| d.chunks_exact(4).filter(|p| p[3] >= 128).count() as f32 / (d.len() / 4) as f32;
        let (f0, f2) = (frac(&chain[0].2), frac(&chain[2].2));
        assert!((f0 - f2).abs() < 0.08, "coverage {f0} vs {f2}");
    }
}
