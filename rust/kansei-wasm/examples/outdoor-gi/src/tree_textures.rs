//! Procedural tree textures, painted on the CPU at load (no image files ship).
//!
//! Foliage atlas (1024², RGBA: sRGB albedo + alpha, and a matching tangent-space normal map):
//! ```text
//!   +--------------------+--------------------+
//!   | A spruce spray     | B spruce curtain   |   A: a branch seen from above, base at the bottom
//!   |   (fishbone)       |   (hanging comb)   |      edge, tip at the top (v = 1 -> 0 along it)
//!   +--------------------+--------------------+   B: pendulous branchlets hanging from the top edge
//!   | C birch, weeping   | D birch, spray     |   C: leafy twigs hanging from the top edge
//!   |   twigs            |   (fan of twigs)   |   D: a fan of leafy twigs from the bottom centre
//!   +--------------------+--------------------+
//! ```
//! Bark: one tileable 256x512 set per species (u around the trunk, v along it).
//!
//! From the Raggare intro's renderer (raggare-web's crates/intro/src/trees/textures.rs, the same author's).

use glam::{Vec2, Vec3};

use crate::canvas::{Canvas, Rng};

pub const ATLAS: usize = 1024;
const R: usize = 512; // region size

/// UV rectangle (u0, v0, u1, v1) of an atlas region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region(pub [f32; 4]);

pub const SPRUCE_SPRAY: Region = Region([0.0, 0.0, 0.5, 0.5]);
pub const SPRUCE_CURTAIN: Region = Region([0.5, 0.0, 1.0, 0.5]);
pub const BIRCH_HANGING: Region = Region([0.0, 0.5, 0.5, 1.0]);
pub const BIRCH_SPRAY: Region = Region([0.5, 0.5, 1.0, 1.0]);

impl Region {
    /// Map region-local (s, t) in 0..1 to atlas UV, inset half a texel against bleeding.
    pub fn uv(&self, s: f32, t: f32) -> Vec2 {
        let [u0, v0, u1, v1] = self.0;
        let inset = 2.0 / ATLAS as f32;
        Vec2::new(u0 + inset + (u1 - u0 - 2.0 * inset) * s, v0 + inset + (v1 - v0 - 2.0 * inset) * t)
    }
}

pub struct Textures {
    pub foliage: (Vec<u8>, Vec<u8>),
    pub spruce_bark: (Vec<u8>, Vec<u8>),
    pub birch_bark: (Vec<u8>, Vec<u8>),
}

pub const BARK_W: usize = 256;
pub const BARK_H: usize = 512;

pub fn generate() -> Textures {
    Textures { foliage: foliage_atlas(), spruce_bark: spruce_bark(), birch_bark: birch_bark() }
}

// ── foliage ──────────────────────────────────────────────────────────────────────────────

const TWIG: Vec3 = Vec3::new(0.045, 0.032, 0.02);

fn needle_colour(rng: &mut Rng, fresh: f32) -> Vec3 {
    // dark blue-green Norway spruce needles, lighter and yellower at the new growth
    let old = Vec3::new(0.022, 0.05, 0.03) * rng.range(0.75, 1.25);
    let new = Vec3::new(0.06, 0.10, 0.035) * rng.range(0.85, 1.15);
    old.lerp(new, fresh.clamp(0.0, 1.0))
}

/// Needles along a polyline, alternating sides, angled forward along `dir`.
#[allow(clippy::too_many_arguments)]
fn needles_along(c: &mut Canvas, rng: &mut Rng, pts: &[Vec2], len: f32, spacing: f32, spread_deg: (f32, f32), fresh_from: f32, droop: Vec2) {
    let total: f32 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let mut travelled = 0.0;
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg = (b - a).length();
        let dir = (b - a) / seg.max(1e-4);
        let mut s = 0.0;
        while s < seg {
            let p = a + dir * s;
            let t = (travelled + s) / total.max(1e-4);
            for side in [-1.0f32, 1.0] {
                let ang = rng.range(spread_deg.0, spread_deg.1).to_radians() * side;
                let (sn, cs) = ang.sin_cos();
                let d = Vec2::new(dir.x * cs - dir.y * sn, dir.x * sn + dir.y * cs);
                let d = (d + droop * rng.f()).normalize();
                let l = len * rng.range(0.75, 1.15) * (1.0 - 0.35 * t);
                let fresh = ((t - fresh_from) / (1.0 - fresh_from)).max(0.0);
                c.stroke(p, p + d * l, 1.8, 0.7, needle_colour(rng, fresh), 2.0 + rng.f());
            }
            s += spacing * rng.range(0.8, 1.2);
        }
        travelled += seg;
    }
}

/// A gently curving polyline from `a` in direction `dir`, `n` segments of `step` pixels.
fn curve(rng: &mut Rng, a: Vec2, dir: Vec2, step: f32, n: usize, bend: f32, pull: Vec2) -> Vec<Vec2> {
    let mut pts = vec![a];
    let mut d = dir.normalize();
    let mut p = a;
    for _ in 0..n {
        let turn = rng.range(-bend, bend);
        let (sn, cs) = turn.sin_cos();
        d = Vec2::new(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
        d = (d + pull).normalize();
        p += d * step;
        pts.push(p);
    }
    pts
}

fn draw_polyline(c: &mut Canvas, pts: &[Vec2], w0: f32, w1: f32, colour: Vec3, height: f32) {
    let n = (pts.len() - 1).max(1) as f32;
    for (i, w) in pts.windows(2).enumerate() {
        let (ta, tb) = (i as f32 / n, (i + 1) as f32 / n);
        c.stroke(w[0], w[1], w0 + (w1 - w0) * ta, w0 + (w1 - w0) * tb, colour, height);
    }
}

fn spruce_spray(c: &mut Canvas, o: Vec2, rng: &mut Rng) {
    let s = R as f32;
    // main axis: base at the bottom edge, tip near the top
    let axis = curve(rng, o + Vec2::new(s * 0.5, s - 6.0), Vec2::new(0.0, -1.0), 24.0, 20, 0.02, Vec2::new(0.0, -0.05));
    draw_polyline(c, &axis, 5.0, 2.0, TWIG, 1.0);
    needles_along(c, rng, &axis, 20.0, 2.0, (35.0, 75.0), 0.7, Vec2::ZERO);
    // side twigs, alternating, shorter toward the tip; second-order twigs on the long ones
    let n = axis.len();
    for (i, &p) in axis.iter().enumerate().skip(1).take(n - 3) {
        let t = i as f32 / n as f32;
        for side in [-1.0f32, 1.0] {
            if rng.f() < 0.12 {
                continue;
            }
            let ang = rng.range(52.0, 72.0).to_radians() * side;
            let d = Vec2::new(ang.sin(), -ang.cos());
            let len = s * 0.44 * (1.0 - 0.7 * t) * rng.range(0.8, 1.1);
            let steps = (len / 12.0).max(2.0) as usize;
            let twig = curve(rng, p, d, len / steps as f32, steps, 0.08, Vec2::new(0.0, -0.04));
            draw_polyline(c, &twig, 2.5, 1.0, TWIG, 1.0);
            needles_along(c, rng, &twig, 16.0, 2.0, (40.0, 80.0), 0.6, Vec2::ZERO);
            if len > s * 0.2 {
                for &q in twig.iter().skip(2).step_by(2).take(3) {
                    let a2 = ang + rng.range(0.5, 0.8) * side;
                    let d2 = Vec2::new(a2.sin(), -a2.cos());
                    let sub = curve(rng, q, d2, 9.0, 4, 0.1, Vec2::ZERO);
                    draw_polyline(c, &sub, 1.6, 0.8, TWIG, 1.0);
                    needles_along(c, rng, &sub, 13.0, 2.2, (40.0, 80.0), 0.5, Vec2::ZERO);
                }
            }
        }
    }
}

fn spruce_curtain(c: &mut Canvas, o: Vec2, rng: &mut Rng) {
    let s = R as f32;
    let count = 16;
    for i in 0..count {
        let x = s * (0.07 + 0.86 * (i as f32 + rng.range(-0.3, 0.3)) / (count - 1) as f32);
        let len = s * rng.range(0.55, 0.95);
        let steps = 14;
        let lean = rng.range(-0.25, 0.25);
        let hang = curve(rng, o + Vec2::new(x, 4.0), Vec2::new(lean, 1.0), len / steps as f32, steps, 0.07, Vec2::new(0.0, 0.05));
        draw_polyline(c, &hang, 2.4, 1.0, TWIG, 1.0);
        needles_along(c, rng, &hang, 17.0, 2.0, (35.0, 75.0), 0.75, Vec2::new(0.0, 0.3));
    }
}

fn leaf_colour(rng: &mut Rng) -> Vec3 {
    let yellow = rng.f();
    Vec3::new(0.04 + 0.03 * yellow, 0.085 + 0.02 * yellow, 0.025) * rng.range(0.7, 1.3)
}

fn leafy_twig(c: &mut Canvas, rng: &mut Rng, pts: &[Vec2], leaf: f32, every: f32) {
    draw_polyline(c, pts, 1.6, 0.8, Vec3::new(0.06, 0.04, 0.03), 1.0);
    let mut acc = 0.0;
    let mut side = 1.0f32;
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg = (b - a).length();
        let dir = (b - a) / seg.max(1e-4);
        let mut s = acc;
        while s < seg {
            let p = a + dir * s;
            let ang = rng.range(35.0, 65.0).to_radians() * side;
            let d = Vec2::new(dir.x * ang.cos() - dir.y * ang.sin(), dir.x * ang.sin() + dir.y * ang.cos());
            let r = leaf * rng.range(0.8, 1.2);
            let stalk = p + d * r * 0.5;
            c.stroke(p, stalk, 1.0, 0.8, Vec3::new(0.06, 0.05, 0.02), 1.5);
            c.ellipse(stalk + d * r, d, r, r * rng.range(0.6, 0.75), leaf_colour(rng), 2.0 + rng.f());
            side = -side;
            s += every * rng.range(0.7, 1.3);
        }
        acc = s - seg;
    }
}

fn birch_hanging(c: &mut Canvas, o: Vec2, rng: &mut Rng) {
    let s = R as f32;
    for i in 0..15 {
        let x = s * (0.06 + 0.88 * (i as f32 + rng.range(-0.4, 0.4)) / 14.0);
        let len = s * rng.range(0.6, 0.95);
        let lean = rng.range(-0.3, 0.3);
        let pts = curve(rng, o + Vec2::new(x, 4.0), Vec2::new(lean, 1.0), len / 16.0, 16, 0.08, Vec2::new(0.0, 0.06));
        leafy_twig(c, rng, &pts, 7.0, 8.5);
    }
}

fn birch_spray(c: &mut Canvas, o: Vec2, rng: &mut Rng) {
    let s = R as f32;
    let base = o + Vec2::new(s * 0.5, s - 6.0);
    for i in 0..13 {
        let ang = (-75.0 + 150.0 * i as f32 / 12.0 + rng.range(-8.0, 8.0)).to_radians();
        let d = Vec2::new(ang.sin(), -ang.cos());
        let len = s * rng.range(0.55, 0.85);
        let pts = curve(rng, base, d, len / 14.0, 14, 0.07, Vec2::new(0.0, 0.03));
        leafy_twig(c, rng, &pts, 7.0, 8.0);
    }
}

fn foliage_atlas() -> (Vec<u8>, Vec<u8>) {
    let mut c = Canvas::new(ATLAS, ATLAS);
    let r = R as f32;
    type Painter = fn(&mut Canvas, Vec2, &mut Rng);
    let regions: [(Painter, Vec2, u32); 4] = [
        (spruce_spray, Vec2::ZERO, 11),
        (spruce_curtain, Vec2::new(r, 0.0), 12),
        (birch_hanging, Vec2::new(0.0, r), 13),
        (birch_spray, Vec2::new(r, r), 14),
    ];
    for (paint, o, seed) in regions {
        // keep each painting inside its region, with a small gutter against mip bleeding
        let (x, y) = (o.x as usize, o.y as usize);
        c.clip = (x + 2, y + 2, x + R - 2, y + R - 2);
        paint(&mut c, o, &mut Rng::new(seed));
    }
    c.clip = (0, 0, ATLAS, ATLAS);
    c.bleed(6);
    c.finish(0.9, false)
}

// ── bark ─────────────────────────────────────────────────────────────────────────────────

/// Wrapping cellular noise: (F1, F2) distances to the nearest seeds, in cell units.
fn cells(p: Vec2, grid: (usize, usize), seeds: &[Vec2]) -> (f32, f32) {
    let (gx, gy) = (grid.0 as i32, grid.1 as i32);
    let cell = Vec2::new(p.x.floor(), p.y.floor());
    let (mut f1, mut f2) = (f32::MAX, f32::MAX);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = cell.x as i32 + dx;
            let cy = cell.y as i32 + dy;
            let seed = seeds[(cy.rem_euclid(gy) * gx + cx.rem_euclid(gx)) as usize];
            let d = (Vec2::new(cx as f32, cy as f32) + seed - p).length();
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    (f1, f2)
}

fn spruce_bark() -> (Vec<u8>, Vec<u8>) {
    // reddish grey-brown bark in thin, rounded scales
    let mut c = Canvas::new(BARK_W, BARK_H);
    let mut rng = Rng::new(21);
    let grid = (18usize, 44usize);
    let seeds: Vec<Vec2> = (0..grid.0 * grid.1).map(|_| Vec2::new(rng.range(0.1, 0.9), rng.range(0.1, 0.9))).collect();
    let tones: Vec<f32> = (0..grid.0 * grid.1).map(|_| rng.range(0.7, 1.3)).collect();
    for y in 0..BARK_H {
        for x in 0..BARK_W {
            let p = Vec2::new(x as f32 / BARK_W as f32 * grid.0 as f32, y as f32 / BARK_H as f32 * grid.1 as f32);
            let (f1, f2) = cells(p, grid, &seeds);
            let edge = ((f2 - f1) * 5.0).min(1.0);
            let cell = (p.y.floor() as usize % grid.1) * grid.0 + p.x.floor() as usize % grid.0;
            let tone = tones[cell];
            let base = Vec3::new(0.13, 0.095, 0.075) * (0.85 + 0.15 * tone) * (0.9 + 0.2 * (p.y * 0.7 + p.x * 1.3).sin().abs());
            let crack = Vec3::new(0.05, 0.04, 0.034);
            let i = y * BARK_W + x;
            c.color[i] = crack.lerp(base, edge.powf(0.4));
            c.height[i] = edge.sqrt() * 2.0 - f1 * 0.6;
            c.alpha[i] = 1.0;
        }
    }
    c.finish(1.2, true)
}

fn birch_bark() -> (Vec<u8>, Vec<u8>) {
    // silver birch: chalky white with dark horizontal lenticels and the odd black scar
    let mut c = Canvas::new(BARK_W, BARK_H);
    let mut rng = Rng::new(22);
    for y in 0..BARK_H {
        let band = 0.94 + 0.06 * ((y as f32 * 0.09).sin() * (y as f32 * 0.023).cos());
        for x in 0..BARK_W {
            let i = y * BARK_W + x;
            c.color[i] = Vec3::new(0.62, 0.61, 0.56) * band * rng.range(0.96, 1.04);
            c.height[i] = 0.0;
            c.alpha[i] = 1.0;
        }
    }
    let w = BARK_W as f32;
    let mark = |c: &mut Canvas, a: Vec2, b: Vec2, width: f32, colour: Vec3| {
        for off in [-w, 0.0, w] {
            // wrap around the trunk
            c.stroke(a + Vec2::new(off, 0.0), b + Vec2::new(off, 0.0), width, width * 0.6, colour, 0.5);
        }
    };
    for _ in 0..140 {
        let p = Vec2::new(rng.range(0.0, w), rng.range(0.0, BARK_H as f32));
        let len = rng.range(8.0, 38.0);
        mark(&mut c, p, p + Vec2::new(len, rng.range(-1.5, 1.5)), rng.range(1.5, 3.0), Vec3::new(0.07, 0.06, 0.05));
    }
    for _ in 0..8 {
        let p = Vec2::new(rng.range(0.0, w), rng.range(0.0, BARK_H as f32));
        let len = rng.range(10.0, 22.0);
        mark(&mut c, p, p + Vec2::new(rng.range(-4.0, 4.0), len), rng.range(6.0, 12.0), Vec3::new(0.035, 0.03, 0.028));
    }
    c.finish(0.8, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_regions_are_painted_and_bark_is_opaque() {
        let t = generate();
        let (rgba, nrm) = &t.foliage;
        assert_eq!(rgba.len(), ATLAS * ATLAS * 4);
        assert_eq!(nrm.len(), rgba.len());
        for (name, r) in [("spray", SPRUCE_SPRAY), ("curtain", SPRUCE_CURTAIN), ("birch hanging", BIRCH_HANGING), ("birch spray", BIRCH_SPRAY)] {
            let [u0, v0, u1, v1] = r.0;
            let (x0, x1, y0, y1) = ((u0 * ATLAS as f32) as usize, (u1 * ATLAS as f32) as usize, (v0 * ATLAS as f32) as usize, (v1 * ATLAS as f32) as usize);
            let mut covered = 0;
            for y in y0..y1 {
                for x in x0..x1 {
                    covered += (rgba[(y * ATLAS + x) * 4 + 3] > 127) as usize;
                }
            }
            let frac = covered as f32 / ((x1 - x0) * (y1 - y0)) as f32;
            assert!(frac > 0.1 && frac < 0.9, "{name}: coverage {frac}");
        }
        for (rgba, _) in [&t.spruce_bark, &t.birch_bark] {
            assert_eq!(rgba.len(), BARK_W * BARK_H * 4);
            assert!(rgba.chunks_exact(4).all(|p| p[3] == 255));
        }
    }
}
