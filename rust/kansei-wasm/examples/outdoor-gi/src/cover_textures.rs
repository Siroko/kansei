//! The ground-cover atlas (1024², sRGB albedo + alpha and a tangent-space normal map), painted
//! on the CPU at load with the trees' canvas:
//! ```text
//!   +--------------------+--------------------+
//!   | GRASS  meadow tuft | SEEDS  grass with  |   all four are side views of a plant standing
//!   |                    |   flowering heads  |   on the bottom edge (v = 1 at the ground)
//!   +--------------------+--------------------+
//!   | FLOWERS  daisies,  | SHRUB  leafy twigs |
//!   |  buttercups, campion|                   |
//!   +--------------------+--------------------+
//! ```
//! Midsummer verges in Dalarna: tall green-to-straw grass, ox-eye daisies, buttercups and red
//! campion, and low broadleaf shrubs at the forest edge.
//!
//! From the Raggare intro's renderer (raggare-web's crates/intro/src/undergrowth/textures.rs, the same author's).

use glam::{Vec2, Vec3};

use crate::canvas::{Canvas, Rng};
use crate::tree_textures::Region;

pub const ATLAS: usize = 1024;
const R: usize = 512;

pub const GRASS: Region = Region([0.0, 0.0, 0.5, 0.5]);
pub const SEEDS: Region = Region([0.5, 0.0, 1.0, 0.5]);
pub const FLOWERS: Region = Region([0.0, 0.5, 0.5, 1.0]);
pub const SHRUB: Region = Region([0.5, 0.5, 1.0, 1.0]);

fn grass_colour(rng: &mut Rng) -> Vec3 {
    // fresh green through to the first straw of midsummer
    // Light Foliage's verge grass: a lighter, yellower green than the forest's
    let green = Vec3::new(0.07, 0.13, 0.035);
    let straw = Vec3::new(0.17, 0.155, 0.07);
    green.lerp(straw, rng.f().powi(3)) * rng.range(0.7, 1.25)
}

/// A curved, tapering blade from `base` rising by `height` pixels, leaning by `lean`.
fn blade(c: &mut Canvas, rng: &mut Rng, base: Vec2, height: f32, lean: f32, width: f32, colour: Vec3) -> Vec2 {
    let steps = 10;
    let mut prev = base;
    let droop = rng.range(0.0, 0.5);
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        // leaning more toward the tip, and the tallest blades bowing over
        let x = base.x + lean * t * t + droop * lean.signum() * height * 0.25 * t.powi(4);
        let y = base.y - height * t + droop * height * 0.15 * t.powi(3);
        let p = Vec2::new(x, y);
        let w0 = width * (1.0 - (t - 1.0 / steps as f32) * 0.9);
        let w1 = width * (1.0 - t * 0.9);
        c.stroke(prev, p, w0.max(0.6), w1.max(0.5), colour * (0.8 + 0.3 * t), 1.0 + t);
        prev = p;
    }
    prev
}

fn tuft(c: &mut Canvas, o: Vec2, rng: &mut Rng, blades: usize, seed_heads: bool) {
    let s = R as f32;
    for _ in 0..blades {
        let base = o + Vec2::new(s * 0.5 + rng.range(-0.12, 0.12) * s, s - 3.0);
        let h = s * rng.range(0.45, 0.96);
        let lean = rng.range(-0.38, 0.38) * s * (h / s);
        let colour = grass_colour(rng);
        let width = rng.range(3.5, 7.0);
        blade(c, rng, base, h, lean, width, colour);
    }
    if seed_heads {
        // flowering stalks (timothy, cocksfoot): thin stems with a dense head near the top
        for _ in 0..9 {
            let base = o + Vec2::new(s * 0.5 + rng.range(-0.1, 0.1) * s, s - 3.0);
            let h = s * rng.range(0.7, 0.97);
            let lean = rng.range(-0.2, 0.2) * s;
            let top = blade(c, rng, base, h, lean, 2.2, Vec3::new(0.07, 0.085, 0.035));
            let dir = (top - (base + Vec2::new(lean * 0.8, -h * 0.8))).normalize_or(Vec2::new(0.0, -1.0));
            let head = Vec3::new(0.11, 0.1, 0.06) * rng.range(0.8, 1.2);
            for k in 0..14 {
                let p = top - dir * (k as f32 * 3.5);
                c.ellipse(p, dir, 3.2, 2.2, head * rng.range(0.85, 1.15), 3.0);
            }
        }
    }
}

fn flower_head(c: &mut Canvas, rng: &mut Rng, p: Vec2, kind: u32) {
    match kind {
        // ox-eye daisy: white rays round a yellow disc
        0 => {
            let petals = 14;
            for k in 0..petals {
                let a = k as f32 / petals as f32 * std::f32::consts::TAU + rng.range(-0.1, 0.1);
                let d = Vec2::new(a.cos(), a.sin() * 0.55);
                c.ellipse(p + d * 9.0, d, 7.0, 2.4, Vec3::new(0.72, 0.72, 0.66) * rng.range(0.9, 1.05), 3.0);
            }
            c.ellipse(p, Vec2::X, 4.5, 3.2, Vec3::new(0.55, 0.38, 0.03), 4.0);
        }
        // buttercup: five glossy yellow petals
        1 => {
            for k in 0..5 {
                let a = k as f32 / 5.0 * std::f32::consts::TAU;
                let d = Vec2::new(a.cos(), a.sin() * 0.7);
                c.ellipse(p + d * 4.0, d, 4.5, 3.5, Vec3::new(0.62, 0.45, 0.02), 3.0);
            }
        }
        // red campion: five notched pink petals
        _ => {
            for k in 0..5 {
                let a = k as f32 / 5.0 * std::f32::consts::TAU + 0.3;
                let d = Vec2::new(a.cos(), a.sin() * 0.7);
                c.ellipse(p + d * 5.0, d, 5.5, 3.0, Vec3::new(0.5, 0.08, 0.17), 3.0);
            }
            c.ellipse(p, Vec2::X, 2.0, 2.0, Vec3::new(0.3, 0.05, 0.1), 4.0);
        }
    }
}

fn flowers(c: &mut Canvas, o: Vec2, rng: &mut Rng) {
    let s = R as f32;
    // a few grass blades behind, so the clump sits in the verge
    for _ in 0..40 {
        let base = o + Vec2::new(s * 0.5 + rng.range(-0.3, 0.3) * s, s - 3.0);
        let colour = grass_colour(rng);
        let (h, lean, width) = (s * rng.range(0.25, 0.55), rng.range(-0.2, 0.2) * s * 0.4, rng.range(3.0, 5.0));
        blade(c, rng, base, h, lean, width, colour);
    }
    for i in 0..16 {
        let kind = [0, 0, 0, 1, 1, 2][i % 6];
        let base = o + Vec2::new(s * 0.5 + rng.range(-0.33, 0.33) * s, s - 3.0);
        let h = s * rng.range(0.4, 0.9);
        let lean = rng.range(-0.12, 0.12) * s;
        let top = blade(c, rng, base, h, lean, 2.6, Vec3::new(0.05, 0.08, 0.025));
        // a leaf or two up the stem
        for _ in 0..2 {
            let t = rng.range(0.2, 0.6);
            let at = base.lerp(top, t);
            let side = if rng.f() < 0.5 { -1.0 } else { 1.0 };
            c.ellipse(at + Vec2::new(side * 9.0, -4.0), Vec2::new(side, -0.6), 11.0, 3.0, Vec3::new(0.05, 0.09, 0.03), 2.0);
        }
        flower_head(c, rng, top, kind);
    }
}

fn shrub(c: &mut Canvas, o: Vec2, rng: &mut Rng) {
    let s = R as f32;
    let twig = Vec3::new(0.05, 0.035, 0.025);
    for _ in 0..16 {
        let base = o + Vec2::new(s * 0.5 + rng.range(-0.08, 0.08) * s, s - 3.0);
        let a = rng.range(-1.1, 1.1);
        let len = s * rng.range(0.45, 0.9);
        let end = base + Vec2::new(a.sin(), -a.cos()) * len;
        c.stroke(base, end, 4.0, 1.2, twig, 1.0);
        let n = (len / 13.0) as usize;
        for k in 2..n {
            let p = base.lerp(end, k as f32 / n as f32);
            let side = if k % 2 == 0 { -1.0 } else { 1.0 };
            let d = Vec2::new(side * 0.8, -0.6).normalize();
            let leaf = Vec3::new(0.045, 0.08, 0.03) * rng.range(0.7, 1.3);
            c.ellipse(p + d * 9.0, d, 10.0, 6.5, leaf, 2.0 + rng.f());
        }
    }
}

/// The atlas: (albedo RGBA8 sRGB, normal RGBA8).
pub fn atlas() -> (Vec<u8>, Vec<u8>) {
    let mut c = Canvas::new(ATLAS, ATLAS);
    let r = R as f32;
    type Painter = fn(&mut Canvas, Vec2, &mut Rng);
    let regions: [(Painter, Vec2, u32); 4] = [
        (|c, o, rng| tuft(c, o, rng, 150, false), Vec2::ZERO, 31),
        (|c, o, rng| tuft(c, o, rng, 110, true), Vec2::new(r, 0.0), 32),
        (flowers, Vec2::new(0.0, r), 33),
        (shrub, Vec2::new(r, r), 34),
    ];
    for (paint, o, seed) in regions {
        let (x, y) = (o.x as usize, o.y as usize);
        c.clip = (x + 2, y + 2, x + R - 2, y + R - 2);
        paint(&mut c, o, &mut Rng::new(seed));
    }
    c.clip = (0, 0, ATLAS, ATLAS);
    c.bleed(6);
    c.finish(0.9, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_region_is_painted() {
        let (rgba, _) = atlas();
        for (name, r) in [("grass", GRASS), ("seeds", SEEDS), ("flowers", FLOWERS), ("shrub", SHRUB)] {
            let [u0, v0, u1, v1] = r.0;
            let (x0, x1, y0, y1) = ((u0 * ATLAS as f32) as usize, (u1 * ATLAS as f32) as usize, (v0 * ATLAS as f32) as usize, (v1 * ATLAS as f32) as usize);
            let covered = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| rgba[(y * ATLAS + x) * 4 + 3] > 127).count();
            let frac = covered as f32 / ((x1 - x0) * (y1 - y0)) as f32;
            assert!(frac > 0.05 && frac < 0.8, "{name}: coverage {frac}");
        }
    }
}
