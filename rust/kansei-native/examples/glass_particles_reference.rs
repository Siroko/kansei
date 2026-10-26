//! A CPU reference of the `glass-particles` demo's glass (rust/kansei-wasm/demos/glass-particles),
//! to judge its ray-traced and screen-space glass against: the same repellers as exact solids
//! (spheres, boxes and pyramids, intersected analytically), each a dielectric of index 1.5 whose
//! tint absorbs over the path inside (Beer), the light split between reflection and refraction by
//! the exact Fresnel equations and reflected whole where it can't leave (total internal
//! reflection); the floor's technical grid, polished (Schlick's Fresnel of F0 0.04, a glossy lobe),
//! lit by the spot light from above and the studio preset's environment; the lamp's face; the
//! environment past it all. No particles, mist, AO or depth of field (turn them off in the demo to
//! compare). The glass shadows the spot light as the demo's shadow maps make it (opaque).
//! Exposed and tone mapped as the demo (EV100 3.4, ACES fitted, sRGB).
//!
//! ```text
//! cargo run --release -p kansei-native --example glass_particles_reference -- out.ppm [samples] [width height]
//! ```
//!
//! The camera is the demo's default orbit (`set_camera(0, 1.5, 0, 7.2, 0.35, 0.16)`, 38 degrees).

use glam::{EulerRot, Mat3, Vec3};

const IOR: f32 = 1.5;
/// The light left after a metre inside (the demo's GLASS_TINT).
const TINT: Vec3 = Vec3::new(0.92, 0.96, 0.98);
/// The studio preset: the spot light (cd, outer cone degrees, inner 0.6 of it, 14 m range) at
/// 5.5 m, its colour (the ember palette's), the environment (cd/m² up and down, a cool white).
const LIGHT: f32 = 6000.0;
const CONE_DEG: f32 = 42.0;
const LIGHT_POS: Vec3 = Vec3::new(0.0, 5.5, 0.0);
const LIGHT_COLOUR: Vec3 = Vec3::new(1.0, 0.86, 0.72);
const LAMP_RADIUS: f32 = 0.2;
const SKY_UP: f32 = 6.0;
const SKY_DOWN: f32 = 2.5;
const EV100: f32 = 3.4;

fn sky_rgb(radiance: f32) -> Vec3 {
    Vec3::new(radiance * 0.96, radiance, radiance * 1.06)
}

/// The environment along `d`: up and down mixed by the height of the direction, as
/// `gi::gradient_sky_lighting` lays it out.
fn sky(d: Vec3) -> Vec3 {
    sky_rgb(SKY_DOWN).lerp(sky_rgb(SKY_UP), d.y * 0.5 + 0.5)
}

#[derive(Clone, Copy)]
enum Shape {
    /// radius
    Sphere(f32),
    /// half extents
    Box(Vec3),
    /// base side, height
    Pyramid(f32, f32),
}

struct Body {
    shape: Shape,
    centre: Vec3,
    /// world from local
    rotation: Mat3,
    /// local planes (outward normal, offset: n . p <= d inside) of the boxes and pyramids
    planes: Vec<(Vec3, f32)>,
}

/// The demo's `colliders()`.
fn bodies() -> Vec<Body> {
    use Shape::*;
    let raw: [(Shape, [f32; 3], [f32; 3]); 10] = [
        (Sphere(0.5), [0.0, 1.55, 0.0], [0.0; 3]),
        (Box(Vec3::splat(0.35)), [-1.2, 1.0, 0.7], [24.0, 38.0, 12.0]),
        (Pyramid(1.0, 1.1), [1.1, 0.0, 0.3], [0.0, 20.0, 0.0]),
        (Sphere(0.3), [1.25, 2.35, 0.9], [0.0; 3]),
        (Box(Vec3::new(0.65, 0.125, 0.275)), [0.4, 0.55, -1.2], [0.0, -30.0, 0.0]),
        (Pyramid(0.75, 0.8), [-0.9, 2.9, -0.8], [180.0, 15.0, 10.0]),
        (Sphere(0.24), [-1.8, 2.0, -0.3], [0.0; 3]),
        (Box(Vec3::splat(0.225)), [0.3, 2.7, -1.4], [40.0, 10.0, 45.0]),
        (Pyramid(0.6, 0.55), [-0.3, 0.0, 1.6], [0.0, 45.0, 0.0]),
        (Sphere(0.35), [1.9, 1.2, -1.1], [0.0; 3]),
    ];
    raw.iter()
        .map(|&(shape, centre, rot)| {
            let [x, y, z] = rot.map(f32::to_radians);
            let planes = match shape {
                Sphere(_) => Vec::new(),
                Box(h) => vec![(Vec3::X, h.x), (Vec3::NEG_X, h.x), (Vec3::Y, h.y), (Vec3::NEG_Y, h.y), (Vec3::Z, h.z), (Vec3::NEG_Z, h.z)],
                Pyramid(side, height) => {
                    let h = side * 0.5;
                    let apex = Vec3::new(0.0, height, 0.0);
                    let base = [Vec3::new(-h, 0.0, -h), Vec3::new(h, 0.0, -h), Vec3::new(h, 0.0, h), Vec3::new(-h, 0.0, h)];
                    let mut planes = vec![(Vec3::NEG_Y, 0.0)];
                    for k in 0..4 {
                        let (a, b) = (base[k], base[(k + 1) % 4]);
                        let mut n = (b - a).cross(apex - a).normalize();
                        if n.dot((a + b) * 0.5 - Vec3::new(0.0, height * 0.25, 0.0)) < 0.0 {
                            n = -n;
                        }
                        planes.push((n, n.dot(a)));
                    }
                    planes
                }
            };
            Body { shape, centre: Vec3::from(centre), rotation: Mat3::from_euler(EulerRot::ZYX, z, y, x), planes }
        })
        .collect()
}

/// Where a ray from `o` along `d` (both world) is inside `body`: (t in, its outward normal, t
/// out, its outward normal), or None.
fn span(body: &Body, o: Vec3, d: Vec3) -> Option<(f32, Vec3, f32, Vec3)> {
    let inv = body.rotation.transpose();
    let (lo, ld) = (inv * (o - body.centre), inv * d);
    let (t0, n0, t1, n1) = match body.shape {
        Shape::Sphere(r) => {
            let b = lo.dot(ld);
            let c = lo.dot(lo) - r * r;
            let disc = b * b - c;
            if disc < 0.0 {
                return None;
            }
            let s = disc.sqrt();
            let (t0, t1) = (-b - s, -b + s);
            ((t0), (lo + ld * t0) / r, t1, (lo + ld * t1) / r)
        }
        _ => {
            let (mut t0, mut t1) = (f32::NEG_INFINITY, f32::INFINITY);
            let (mut n0, mut n1) = (Vec3::ZERO, Vec3::ZERO);
            for &(n, off) in &body.planes {
                let denom = n.dot(ld);
                let dist = off - n.dot(lo);
                if denom.abs() < 1e-9 {
                    if dist < 0.0 {
                        return None;
                    }
                    continue;
                }
                let t = dist / denom;
                if denom < 0.0 {
                    if t > t0 {
                        t0 = t;
                        n0 = n;
                    }
                } else if t < t1 {
                    t1 = t;
                    n1 = n;
                }
            }
            if t0 > t1 {
                return None;
            }
            (t0, n0, t1, n1)
        }
    };
    (t1 > 1e-4).then(|| (t0, body.rotation * n0, t1, body.rotation * n1))
}

/// The floor's grid albedo at (x, z): the demo's defaults (0.1 m minor cells, majors every 10, the
/// crosses at every second major intersection), lines hard (the samples filter them).
fn floor_albedo(x: f32, z: f32) -> Vec3 {
    let (base, minor, major, cross) = (Vec3::new(0.035, 0.036, 0.038), Vec3::new(0.08, 0.082, 0.085), Vec3::new(0.17, 0.174, 0.18), Vec3::new(0.34, 0.345, 0.35));
    let (cell, every) = (0.1f32, 10.0f32);
    let (wm, wj, wc, arm) = (0.0035f32, 0.011f32, 0.016f32, 0.07f32);
    let near = |v: f32, s: f32| ((v / s) - (v / s).round()).abs() * s;
    let big = cell * every;
    let mut a = base;
    if near(x, cell) < wm * 0.5 || near(z, cell) < wm * 0.5 {
        a = minor;
    }
    if near(x, big) < wj * 0.5 || near(z, big) < wj * 0.5 {
        a = major;
    }
    let (nx, nz) = ((x / big).round(), (z / big).round());
    let even = (nx * 0.5).fract().abs() < 0.25 && (nz * 0.5).fract().abs() < 0.25;
    let (qx, qz) = ((x / big - nx).abs(), (z / big - nz).abs());
    let w = wc / big * 0.5;
    if even && ((qx < arm && qz < w) || (qz < arm && qx < w)) {
        a = cross;
    }
    a
}

fn fresnel(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let sin_t = eta_i / eta_t * (1.0 - cos_i * cos_i).max(0.0).sqrt();
    if sin_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let rs = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    let rp = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    0.5 * (rs * rs + rp * rp)
}

fn refract(d: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let c = -n.dot(d);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    (k >= 0.0).then(|| d * eta + n * (eta * c - k.sqrt()))
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

enum Hit {
    Floor(f32),
    Lamp,
    Glass(f32, usize, Vec3),
}

/// The nearest thing a ray from outside the glass meets.
fn nearest(bodies: &[Body], o: Vec3, d: Vec3) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    let mut best_t = f32::INFINITY;
    if d.y < -1e-6 {
        let t = -o.y / d.y;
        if t > 1e-4 && t < best_t {
            best_t = t;
            best = Some(Hit::Floor(t));
        }
    }
    if d.y > 1e-6 {
        let t = (LIGHT_POS.y + 0.03 - o.y) / d.y;
        let p = o + d * t;
        if t > 1e-4 && t < best_t && p.x * p.x + p.z * p.z < LAMP_RADIUS * LAMP_RADIUS {
            best_t = t;
            best = Some(Hit::Lamp);
        }
    }
    for (k, b) in bodies.iter().enumerate() {
        if let Some((t0, n0, _, _)) = span(b, o, d) {
            if t0 > 1e-4 && t0 < best_t {
                best_t = t0;
                best = Some(Hit::Glass(t0, k, n0));
            }
        }
    }
    best
}

/// The spot light's illuminance at floor point `p` (spot_light_types.wgsl's kansei_spot_sample
/// on a surface facing up), shadowed by the glass.
fn direct(bodies: &[Body], p: Vec3) -> Vec3 {
    let outer = CONE_DEG.to_radians();
    let (cos_outer, cos_inner) = (outer.cos(), (outer * 0.6).cos());
    let d = LIGHT_POS - p;
    let dist2 = d.length_squared().max(1e-4);
    let l = d / dist2.sqrt();
    let r = dist2 / (14.0 * 14.0);
    let window = (1.0 - r * r).clamp(0.0, 1.0);
    let cone = ((l.y - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
    if l.y <= 0.0 || cone <= 0.0 {
        return Vec3::ZERO;
    }
    let len = dist2.sqrt();
    if bodies.iter().any(|b| span(b, p + Vec3::Y * 1e-3, l).is_some_and(|(t0, _, _, _)| t0.max(0.0) < len)) {
        return Vec3::ZERO;
    }
    LIGHT_COLOUR * LIGHT * (window * window * cone * cone / dist2 * l.y)
}

/// The light arriving at `o` from `d` (from outside the glass), `depth` bounces deep: one path,
/// the reflection or the refraction chosen by Fresnel at each glass surface.
fn radiance(bodies: &[Body], o: Vec3, d: Vec3, depth: u32, rng: &mut Rng) -> Vec3 {
    if depth > 16 {
        return sky(d);
    }
    match nearest(bodies, o, d) {
        None => sky(d),
        Some(Hit::Lamp) => LIGHT_COLOUR * (LIGHT / (std::f32::consts::PI * LAMP_RADIUS * LAMP_RADIUS)),
        Some(Hit::Floor(t)) => {
            let p = o + d * t;
            let albedo = floor_albedo(p.x, p.z);
            let diffuse = albedo * (direct(bodies, p) / std::f32::consts::PI + sky_rgb(SKY_UP));
            // polished: Schlick's Fresnel of F0 0.04, the reflection jittered over a lobe of
            // roughness 0.22 (alpha 0.05)
            let cos = (-d.y).clamp(0.0, 1.0);
            let f = 0.04 + 0.96 * (1.0 - cos).powi(5);
            let (u, v) = (rng.next(), rng.next());
            let alpha = 0.22f32 * 0.22;
            let theta = (alpha * (u / (1.0 - u).max(1e-6)).sqrt()).atan();
            let phi = v * std::f32::consts::TAU;
            let h = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            let mut r = d - 2.0 * d.dot(h) * h;
            if r.y <= 0.0 {
                r = Vec3::new(d.x, -d.y, d.z);
            }
            diffuse * (1.0 - f) + f * radiance(bodies, p + Vec3::Y * 1e-4, r, depth + 1, rng)
        }
        Some(Hit::Glass(t, k, n)) => {
            let p = o + d * t;
            let cos = (-n.dot(d)).clamp(0.0, 1.0);
            let f = fresnel(cos, 1.0, IOR);
            if rng.next() < f {
                let r = d - 2.0 * d.dot(n) * n;
                return radiance(bodies, p + n * 1e-4, r, depth + 1, rng);
            }
            let Some(mut dir) = refract(d, n, 1.0 / IOR) else { return Vec3::ZERO };
            // inside: to the far side, refract out or reflect back in
            let mut q = p - n * 1e-4;
            let mut throughput = Vec3::ONE;
            for bounce in 0..32 {
                let Some((_, _, t1, n1)) = span(&bodies[k], q, dir) else { return Vec3::ZERO };
                let exit = q + dir * t1;
                throughput *= TINT.powf(t1);
                // the outward normal; leaving from glass to air
                let cos = n1.dot(dir).clamp(0.0, 1.0);
                let f = fresnel(cos, IOR, 1.0);
                if rng.next() >= f {
                    if let Some(out) = refract(dir, -n1, IOR) {
                        return throughput * radiance(bodies, exit + n1 * 1e-4, out, depth + bounce + 1, rng);
                    }
                }
                dir = dir - 2.0 * dir.dot(n1) * n1;
                q = exit - n1 * 1e-4;
            }
            Vec3::ZERO
        }
    }
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
    let out = args.get(1).cloned().unwrap_or_else(|| "glass_particles_reference.ppm".into());
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let w: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(960);
    let h: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(540);
    let bodies = bodies();
    // the demo's default orbit: target (0, 1.5, 0), 7.2 m, azimuth 0.35, elevation 0.16
    let (target, radius, az, el) = (Vec3::new(0.0, 1.5, 0.0), 7.2f32, 0.35f32, 0.16f32);
    let eye = target + radius * Vec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos());
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    let tan_half = (38f32.to_radians() * 0.5).tan();
    let aspect = w as f32 / h as f32;
    let exposure = 1.0 / 2f32.powf(EV100);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let rows: Vec<(usize, Vec<u8>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let bodies = &bodies;
                s.spawn(move || {
                    let mut out = Vec::new();
                    for y in (k..h).step_by(threads) {
                        let mut row = Vec::with_capacity(w * 3);
                        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (y as u64 * 0x2545_f491));
                        for x in 0..w {
                            let mut sum = Vec3::ZERO;
                            for _ in 0..samples {
                                let (jx, jy) = (rng.next(), rng.next());
                                let u = ((x as f32 + jx) / w as f32 * 2.0 - 1.0) * tan_half * aspect;
                                let v = (1.0 - (y as f32 + jy) / h as f32 * 2.0) * tan_half;
                                let d = (forward + right * u + up * v).normalize();
                                sum += radiance(bodies, eye, d, 0, &mut rng);
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
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut image = vec![0u8; w * h * 3];
    for (y, row) in rows {
        image[y * w * 3..(y + 1) * w * 3].copy_from_slice(&row);
    }
    let mut file = format!("P6\n{w} {h}\n255\n").into_bytes();
    file.extend(image);
    std::fs::write(&out, file).expect("write the image");
    println!("{out}: {w} x {h}, {samples} samples a pixel");
}
