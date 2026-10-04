//! A CPU path-traced reference of the `gi-box` example (rust/kansei-wasm/examples/gi-box), to judge
//! its global illumination modes against: the same room, blocks, rug, downlight (inverse square
//! with the engine's range window and cone, a 0.15 m source for soft shadows) and camera, every
//! surface Lambertian, light bouncing until it leaves through the open front. Exposed and tone
//! mapped as the example (EV100 5, ACES fitted, sRGB).
//!
//! ```text
//! cargo run --release -p kansei-native --example gi_box_reference -- out.ppm [samples] [indirect]
//! ```
//!
//! `indirect` renders only the light the bounces add, two stops brighter (the example's
//! `view=indirect`).

use glam::{Mat3, Vec3};

const W: usize = 400;
const H: usize = 300;

struct Block {
    centre: Vec3,
    half: Vec3,
    /// world from the box's frame (a rotation about y)
    rotation: Mat3,
    albedo: fn(Vec3) -> Vec3,
}

fn white(_: Vec3) -> Vec3 {
    Vec3::splat(0.73)
}
fn red(_: Vec3) -> Vec3 {
    Vec3::new(0.63, 0.065, 0.05)
}
fn green(_: Vec3) -> Vec3 {
    Vec3::new(0.14, 0.45, 0.09)
}
/// The rug: orange on its left half, blue on its right (in its own frame), a pale border.
fn rug(local: Vec3) -> Vec3 {
    if local.x.abs() > 1.2 - 3.0 / 64.0 * 2.4 || local.z.abs() > 0.4 - 3.0 / 64.0 * 0.8 {
        Vec3::new(0.8, 0.75, 0.6)
    } else if local.x < 0.0 {
        Vec3::new(0.85, 0.35, 0.05)
    } else {
        Vec3::new(0.05, 0.25, 0.8)
    }
}

fn scene() -> Vec<Block> {
    let block = |size: [f32; 3], centre: [f32; 3], yaw: f32, albedo: fn(Vec3) -> Vec3| Block {
        centre: Vec3::from(centre),
        half: Vec3::from(size) * 0.5,
        rotation: Mat3::from_rotation_y(yaw),
        albedo,
    };
    vec![
        block([4.4, 0.2, 4.2], [0.0, -0.1, -2.1], 0.0, white),
        block([4.4, 0.2, 4.2], [0.0, 4.1, -2.1], 0.0, white),
        block([4.4, 4.4, 0.2], [0.0, 2.0, -4.1], 0.0, white),
        block([0.2, 4.4, 4.2], [-2.1, 2.0, -2.1], 0.0, red),
        block([0.2, 4.4, 4.2], [2.1, 2.0, -2.1], 0.0, green),
        block([1.2, 2.4, 1.2], [-0.75, 1.2, -2.6], 0.33, white),
        block([1.2, 1.2, 1.2], [0.8, 0.6, -1.5], -0.3, white),
        block([2.4, 0.02, 0.8], [0.0, 0.01, -0.55], 0.0, rug),
    ]
}

/// The nearest hit along the ray: (distance, normal, albedo).
fn trace(blocks: &[Block], o: Vec3, d: Vec3) -> Option<(f32, Vec3, Vec3)> {
    let mut best: Option<(f32, Vec3, Vec3)> = None;
    for b in blocks {
        let inv = b.rotation.transpose();
        let lo = inv * (o - b.centre);
        let ld = inv * d;
        let t1 = (-b.half - lo) / ld;
        let t2 = (b.half - lo) / ld;
        let (tmin, tmax) = (t1.min(t2), t1.max(t2));
        let near = tmin.max_element();
        let far = tmax.min_element();
        if near > far || far < 1e-4 || near < 1e-4 || best.is_some_and(|h| h.0 <= near) {
            continue;
        }
        let axis = if tmin.x >= tmin.y && tmin.x >= tmin.z { 0 } else if tmin.y >= tmin.z { 1 } else { 2 };
        let mut n = Vec3::ZERO;
        n[axis] = -ld[axis].signum();
        let local = lo + ld * near;
        best = Some((near, b.rotation * n, (b.albedo)(local)));
    }
    best
}

const LIGHT_POS: Vec3 = Vec3::new(0.0, 3.85, -2.0);
const LIGHT_DIR: Vec3 = Vec3::new(0.0, -1.0, 0.0);
const LIGHT_RADIUS: f32 = 0.15;

/// The downlight's illuminance at `p` on a surface of normal `n`, shadowed (one sample of the
/// source's disk): spot_light_types.wgsl's kansei_spot_sample.
fn direct(blocks: &[Block], p: Vec3, n: Vec3, u: (f32, f32)) -> Vec3 {
    let color = Vec3::new(1.0, 0.92, 0.8) * 1600.0;
    let (range, cos_inner, cos_outer) = (12.0f32, 50f32.to_radians().cos(), 76f32.to_radians().cos());
    let d = LIGHT_POS - p;
    let dist2 = d.length_squared().max(1e-4);
    let l = d / dist2.sqrt();
    let r = dist2 / (range * range);
    let window = (1.0 - r * r).clamp(0.0, 1.0);
    let cone = ((-l).dot(LIGHT_DIR) - cos_outer) / (cos_inner - cos_outer);
    let cone = cone.clamp(0.0, 1.0);
    let ndl = n.dot(l);
    if ndl <= 0.0 || cone <= 0.0 {
        return Vec3::ZERO;
    }
    // a point of the source's disk (facing the receiver)
    let (t, b) = l.any_orthonormal_pair();
    let (rr, phi) = (u.0.sqrt() * LIGHT_RADIUS, u.1 * std::f32::consts::TAU);
    let target = LIGHT_POS + t * (rr * phi.cos()) + b * (rr * phi.sin());
    let to = target - p;
    let len = to.length();
    let shadowed = trace(blocks, p + n * 1e-3, to / len).is_some_and(|h| h.0 < len - 1e-3);
    if shadowed {
        return Vec3::ZERO;
    }
    color * (window * window * cone * cone / dist2 * ndl)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Radiance toward the camera along a ray: direct light at the first hit (unless `indirect`
/// only), plus the bounces.
fn radiance(blocks: &[Block], mut o: Vec3, mut d: Vec3, rng: &mut Rng, indirect_only: bool) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    for bounce in 0..12 {
        let Some((t, n, albedo)) = trace(blocks, o, d) else { break };
        let p = o + d * t;
        let n = if n.dot(d) > 0.0 { -n } else { n };
        if bounce > 0 || !indirect_only {
            sum += throughput * albedo / std::f32::consts::PI * direct(blocks, p, n, (rng.next(), rng.next()));
        }
        // cosine-weighted next direction: throughput times albedo
        let (u1, u2) = (rng.next(), rng.next());
        let (tt, bb) = n.any_orthonormal_pair();
        let (r, phi) = (u1.sqrt(), std::f32::consts::TAU * u2);
        d = (tt * (r * phi.cos()) + bb * (r * phi.sin()) + n * (1.0 - u1).max(0.0).sqrt()).normalize();
        o = p + n * 1e-3;
        throughput *= albedo;
        if bounce > 2 {
            let survive = throughput.max_element().min(0.95);
            if rng.next() > survive {
                break;
            }
            throughput /= survive;
        }
    }
    sum
}

fn aces_fitted(c: Vec3) -> Vec3 {
    let input = Mat3::from_cols_array(&[0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777]);
    let output = Mat3::from_cols_array(&[1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602]);
    let v = input * c;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    (output * (a / b)).clamp(Vec3::ZERO, Vec3::ONE)
}

fn srgb(c: f32) -> u8 {
    let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "gi_box_reference.ppm".into());
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let indirect_only = args.get(3).is_some_and(|s| s == "indirect");
    let blocks = scene();
    // the example's camera: 38 degrees vertical, at (0, 2, 6.3) looking at (0, 2, -2)
    let eye = Vec3::new(0.0, 2.0, 6.3);
    let forward = (Vec3::new(0.0, 2.0, -2.0) - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    let tan_half = (38f32.to_radians() * 0.5).tan();
    let aspect = W as f32 / H as f32;
    let exposure = 1.0 / 32.0 * if indirect_only { 4.0 } else { 1.0 };
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let rows: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let blocks = &blocks;
                s.spawn(move || {
                    let mut out = Vec::new();
                    for y in (k..H).step_by(threads) {
                        let mut row = Vec::with_capacity(W * 3);
                        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (y as u64 * 0x1000_0001));
                        for x in 0..W {
                            let mut sum = Vec3::ZERO;
                            for _ in 0..samples {
                                let px = ((x as f32 + rng.next()) / W as f32 * 2.0 - 1.0) * tan_half * aspect;
                                let py = (1.0 - (y as f32 + rng.next()) / H as f32 * 2.0) * tan_half;
                                let d = (forward + right * px + up * py).normalize();
                                sum += radiance(blocks, eye, d, &mut rng, indirect_only);
                            }
                            let c = aces_fitted(sum / samples as f32 * exposure);
                            row.extend([srgb(c.x), srgb(c.y), srgb(c.z)]);
                        }
                        out.push((y, row));
                    }
                    out
                })
            })
            .collect();
        let mut rows = vec![Vec::new(); H];
        for h in handles {
            for (y, row) in h.join().unwrap() {
                rows[y] = row;
            }
        }
        rows
    });
    let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
    for row in rows {
        ppm.extend(row);
    }
    std::fs::write(&out, ppm).expect("write the image");
    println!("{out}: {W}x{H}, {samples} samples per pixel{}", if indirect_only { ", indirect light only" } else { "" });
}
