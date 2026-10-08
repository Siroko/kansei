//! Procedural tree meshes, 1 m tall (instances scale them), with the trunk along +Y from the
//! origin. Each tree is two meshes: bark (opaque) and foliage (alpha-tested cards on the atlas of
//! `textures.rs`). `position.w` carries an ambient-occlusion factor (1 = open, lower = inside the
//! crown); foliage normals are bent toward the crown's outward direction for soft shading.
//!
//! Norway spruce (Picea abies), forest-grown: a straight tapering trunk, bare with dead stubs for
//! the lower third, then a narrow conical crown of whorled branches that rise near the top, run
//! level in the middle and droop lower down, carrying flat sprays with pendulous branchlets
//! hanging beneath ("comb" spruce), and a vertical leader at the tip.
//!
//! Silver birch (Betula pendula): a slender, slightly wavering white trunk, steep ascending limbs
//! from about a third of the height, weeping leafy twigs hanging from them and a few leafy fans
//! at the limb ends and the top.
//!
//! From the Raggare intro's renderer (raggare-web's crates/intro/src/trees/geometry.rs, the same author's).

use glam::{Vec2, Vec3};
use kansei_core::geometries::Vertex;

use crate::canvas::Rng;
use crate::tree_textures::{Region, BIRCH_HANGING, BIRCH_SPRAY, SPRUCE_CURTAIN, SPRUCE_SPRAY};

#[derive(Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub struct TreeMesh {
    pub bark: Mesh,
    pub foliage: Mesh,
}

impl Mesh {
    fn vertex(&mut self, p: Vec3, ao: f32, n: Vec3, uv: Vec2) -> u32 {
        self.vertices.push(Vertex { position: [p.x, p.y, p.z, ao], normal: n.to_array(), uv: uv.to_array() });
        (self.vertices.len() - 1) as u32
    }

    /// A card along a polyline `spine` (base first), `half_widths` across it along `across`
    /// (one per spine point), textured with `region` (s across, t from `t0` at the base to `t1`
    /// at the end of the spine). Normals blend the card normal with `outward(p)`.
    #[allow(clippy::too_many_arguments)]
    fn ribbon(&mut self, spine: &[Vec3], across: Vec3, half_widths: &[f32], region: Region, t0: f32, t1: f32, ao: &dyn Fn(f32) -> f32, outward: &dyn Fn(Vec3) -> Vec3, bend: f32) {
        let base = self.vertices.len() as u32;
        let n = spine.len();
        for (i, (&p, &hw)) in spine.iter().zip(half_widths).enumerate() {
            let f = i as f32 / (n - 1) as f32;
            let along = if i + 1 < n { spine[i + 1] - p } else { p - spine[i - 1] };
            let card_n = along.cross(across).normalize_or_zero();
            let card_n = if card_n.dot(outward(p)) < 0.0 { -card_n } else { card_n };
            let normal = card_n.lerp(outward(p), bend).normalize_or_zero();
            let t = t0 + (t1 - t0) * f;
            self.vertex(p - across * hw, ao(f), normal, region.uv(0.0, t));
            self.vertex(p + across * hw, ao(f), normal, region.uv(1.0, t));
        }
        for i in 0..(n as u32 - 1) {
            let (a, b, c, d) = (base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3);
            self.indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }

    /// A tube along `spine` with radii `radii`, `sides` around; UV u around (`u_offset` + 0..1),
    /// v = `v_scale` times the distance along the spine (tiled bark).
    #[allow(clippy::too_many_arguments)]
    fn tube(&mut self, spine: &[Vec3], radii: &[f32], sides: u32, v_scale: f32, u_offset: f32, ao: &dyn Fn(f32) -> f32, cap_end: bool) {
        let base = self.vertices.len() as u32;
        let n = spine.len();
        let mut dist = 0.0;
        for i in 0..n {
            let p = spine[i];
            if i > 0 {
                dist += (p - spine[i - 1]).length();
            }
            let dir = if i + 1 < n { spine[i + 1] - p } else { p - spine[i - 1] }.normalize();
            let side = if dir.y.abs() > 0.95 { Vec3::X } else { Vec3::Y }.cross(dir).normalize();
            let up = dir.cross(side);
            for k in 0..=sides {
                let a = k as f32 / sides as f32 * std::f32::consts::TAU;
                let (s, c) = a.sin_cos();
                let radial = side * c + up * s;
                self.vertex(p + radial * radii[i], ao(i as f32 / (n - 1) as f32), radial, Vec2::new(u_offset + k as f32 / sides as f32, dist * v_scale));
            }
        }
        let ring = sides + 1;
        for i in 0..(n as u32 - 1) {
            for k in 0..sides {
                let a = base + i * ring + k;
                let b = a + ring;
                // winds outward (counter-clockwise seen from outside)
                self.indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
            }
        }
        if cap_end {
            let last = spine[n - 1];
            let tip = self.vertex(last, ao(1.0), (last - spine[n - 2]).normalize(), Vec2::new(u_offset + 0.5, dist * v_scale));
            let r0 = base + (n as u32 - 1) * ring;
            for k in 0..sides {
                self.indices.extend_from_slice(&[r0 + k, r0 + k + 1, tip]);
            }
        }
    }

    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

struct Lod {
    whorls: usize,
    per_whorl: usize,
    segments: usize,
    curtains: bool,
    trunk_sides: u32,
    stubs: bool,
    card_scale: f32,
}

const SPRUCE_LODS: [Lod; 3] = [
    Lod { whorls: 30, per_whorl: 6, segments: 3, curtains: true, trunk_sides: 8, stubs: true, card_scale: 1.1 },
    Lod { whorls: 15, per_whorl: 5, segments: 2, curtains: true, trunk_sides: 6, stubs: false, card_scale: 1.35 },
    Lod { whorls: 11, per_whorl: 5, segments: 1, curtains: false, trunk_sides: 4, stubs: false, card_scale: 1.6 },
];

/// Crown shape of a spruce archetype (all as fractions of the tree's height).
pub struct SpruceStyle {
    /// Where the live crown starts.
    pub crown_base: f32,
    /// Widest crown radius, at the crown base.
    pub crown_radius: f32,
    /// Extra droop of the lower branches.
    pub droop: f32,
    pub seed: u32,
}

/// Interior trees (slim and full, mixed at random) and forest-edge trees, which keep live
/// branches almost to the ground where the road clearing lets light in.
pub const SPRUCE_STYLES: [SpruceStyle; 3] = [
    SpruceStyle { crown_base: 0.3, crown_radius: 0.14, droop: 0.2, seed: 7 },
    SpruceStyle { crown_base: 0.22, crown_radius: 0.165, droop: 0.25, seed: 17 },
    SpruceStyle { crown_base: 0.07, crown_radius: 0.19, droop: 0.32, seed: 27 },
];
pub const SPRUCE_EDGE_STYLE: usize = 2;

fn spruce_trunk_radius(y: f32) -> f32 {
    let flare = 1.0 + 0.7 * (-y * 45.0).exp();
    (0.0105 * (1.0 - y).powf(0.9) + 0.0012) * flare
}

pub fn spruce(lod: usize, style: &SpruceStyle) -> TreeMesh {
    let l = &SPRUCE_LODS[lod.min(2)];
    let mut rng = Rng::new(style.seed);
    let mut bark = Mesh::default();
    let mut foliage = Mesh::default();
    let cb = style.crown_base;

    // trunk: denser rings near the flared base
    let rings = if lod == 0 { 24 } else if lod == 1 { 10 } else { 4 };
    let top = if lod == 2 { cb + 0.15 } else { 0.985 };
    let ys: Vec<f32> = (0..=rings).map(|i| top * (i as f32 / rings as f32).powf(1.3)).collect();
    let spine: Vec<Vec3> = ys.iter().map(|&y| Vec3::new(0.0, y, 0.0)).collect();
    let radii: Vec<f32> = ys.iter().map(|&y| spruce_trunk_radius(y)).collect();
    let trunk_ao = |f: f32| if f * top < cb { 1.0 } else { 0.55 };
    bark.tube(&spine, &radii, l.trunk_sides, 12.0, 0.0, &trunk_ao, lod < 2);

    // dead stubs on the bare bole
    if l.stubs {
        for _ in 0..14 {
            let y = rng.range(0.05, cb.max(0.1));
            let a = rng.range(0.0, std::f32::consts::TAU);
            let dir = Vec3::new(a.cos(), rng.range(-0.5, 0.1), a.sin()).normalize();
            let r = spruce_trunk_radius(y);
            let base = Vec3::new(0.0, y, 0.0) + dir * r * 0.8;
            // snapped short: 10-30 cm on a 20 m tree (long ones read as black spikes against the
            // lit fog in the trunks shot, which Unreal's boles don't have)
            let len = rng.range(0.005, 0.015);
            bark.tube(&[base, base + dir * len], &[0.0012, 0.0005], 3, 12.0, 0.0, &|_| 0.8, false);
        }
    }

    let crown_centre = Vec3::new(0.0, cb + (1.0 - cb) * 0.3, 0.0);
    let outward = move |p: Vec3| {
        let d = p - crown_centre;
        Vec3::new(d.x, d.y * 0.35 + 0.15, d.z).normalize_or_zero()
    };

    // whorls of branches: rising near the top, level mid-crown, drooping low down
    let golden = 2.399_963_2_f32;
    let mut azimuth = rng.range(0.0, std::f32::consts::TAU);
    for w in 0..l.whorls {
        let f = (w as f32 + rng.range(0.0, 0.6)) / l.whorls as f32;
        let y = cb + (0.965 - cb) * f;
        let t = (1.0 - y) / (1.0 - cb); // 0 at the top, 1 at the crown base
        let reach = style.crown_radius * t.powf(0.8) + 0.02;
        for b in 0..l.per_whorl {
            azimuth += golden + rng.range(-0.3, 0.3);
            let len = reach * rng.range(0.8, 1.15);
            let h = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
            let rise = (25.0 - 35.0 * t + rng.range(-6.0, 6.0)).to_radians().tan();
            let droop = 0.06 + style.droop * t;
            let start = Vec3::new(0.0, y, 0.0) + h * spruce_trunk_radius(y) * 0.8;
            let segs = l.segments;
            let spine: Vec<Vec3> = (0..=segs)
                .map(|i| {
                    let s = i as f32 / segs as f32;
                    let p = start + h * (len * s) + Vec3::Y * (len * (rise * s - droop * s * s));
                    Vec3::new(p.x, p.y.max(0.03), p.z) // low edge branches rest just above the ground
                })
                .collect();
            // the spray lies roughly flat, rolled a little around the branch
            let roll = rng.range(-0.35, 0.35) + if b % 2 == 0 { 0.1 } else { -0.1 };
            let lateral = Vec3::Y.cross(h).normalize();
            let across = (lateral * roll.cos() + Vec3::Y * roll.sin()).normalize();
            // a floor on the width keeps the short top branches from leaving gaps between whorls
            let hw: Vec<f32> = (0..=segs).map(|i| (len * 0.5).max(0.022) * l.card_scale * (1.0 - 0.25 * i as f32 / segs as f32)).collect();
            let crown_ao = 0.55 + 0.45 * (1.0 - t * 0.5);
            let ao = move |s: f32| (0.45 + 0.55 * s) * crown_ao;
            foliage.ribbon(&spine, across, &hw, SPRUCE_SPRAY, 1.0, 0.03, &ao, &outward, 0.55);
            // a second spray rolled steeply about the branch: a real branch's sprays fan in 3D, and
            // seen from the ground (most of the film's cameras) the flat one alone is edge-on
            let hw2: Vec<f32> = hw.iter().map(|w| w * 0.8).collect();
            let clear_of_ground = spine.iter().zip(&hw2).all(|(p, w)| p.y > w + 0.005);
            if lod < 2 && clear_of_ground {
                let steep = (lateral * (roll + 0.95).cos() + Vec3::Y * (roll + 0.95).sin()).normalize();
                foliage.ribbon(&spine, steep, &hw2, SPRUCE_SPRAY, 1.0, 0.03, &ao, &outward, 0.55);
            }

            // pendulous branchlets hanging from the outer part of the lower and middle branches
            // (short, and on every other branch: long ones read as a weeping willow's strands)
            if l.curtains && t > 0.3 && b % 2 == 0 {
                let from = 1usize.min(segs);
                let hang = len * (0.1 + 0.15 * t) * l.card_scale;
                let top_line: Vec<Vec3> = spine[from..].to_vec();
                if top_line.len() >= 2 {
                    // a vertical curtain: the ribbon runs down from the branch
                    let mid = top_line[top_line.len() / 2];
                    let bottom_line: Vec<Vec3> = top_line.iter().map(|p| Vec3::new(p.x, (p.y - hang).max(0.005), p.z)).collect();
                    let base_idx = foliage.vertices.len() as u32;
                    let n = top_line.len();
                    for (i, (&p, &q)) in top_line.iter().zip(&bottom_line).enumerate() {
                        let s = i as f32 / (n - 1) as f32;
                        let face = h.cross(Vec3::Y).normalize();
                        let normal = face.lerp(outward(mid), 0.7).normalize();
                        foliage.vertex(p, crown_ao * 0.8, normal, SPRUCE_CURTAIN.uv(0.08 + 0.84 * s, 0.0));
                        foliage.vertex(q, crown_ao * 0.55, normal, SPRUCE_CURTAIN.uv(0.08 + 0.84 * s, 1.0));
                    }
                    for i in 0..(n as u32 - 1) {
                        let (a, b2, c, d) = (base_idx + 2 * i, base_idx + 2 * i + 1, base_idx + 2 * i + 2, base_idx + 2 * i + 3);
                        foliage.indices.extend_from_slice(&[a, b2, c, c, b2, d]);
                    }
                }
            }
        }
    }

    // the leader: two crossed vertical sprays at the tip
    let leader_base = 0.93;
    for k in 0..2 {
        let a = azimuth + k as f32 * std::f32::consts::FRAC_PI_2;
        let across = Vec3::new(a.cos(), 0.0, a.sin());
        let spine = [Vec3::new(0.0, leader_base, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        foliage.ribbon(&spine, across, &[0.02 * l.card_scale, 0.012 * l.card_scale], SPRUCE_SPRAY, 1.0, 0.03, &|_| 1.0, &outward, 0.5);
    }
    TreeMesh { bark, foliage }
}

struct BirchLod {
    limbs: usize,
    clusters: usize,
    trunk_sides: u32,
    limb_sides: u32,
    card_scale: f32,
}

const BIRCH_LODS: [BirchLod; 3] = [
    BirchLod { limbs: 16, clusters: 9, trunk_sides: 8, limb_sides: 4, card_scale: 1.15 },
    BirchLod { limbs: 10, clusters: 4, trunk_sides: 6, limb_sides: 3, card_scale: 1.4 },
    BirchLod { limbs: 5, clusters: 1, trunk_sides: 4, limb_sides: 0, card_scale: 1.7 },
];

pub fn birch(lod: usize, seed: u32) -> TreeMesh {
    let l = &BIRCH_LODS[lod.min(2)];
    let mut rng = Rng::new(seed);
    let mut bark = Mesh::default();
    let mut foliage = Mesh::default();
    let cb = 0.34;
    let phase = rng.range(0.0, 6.0);
    let axis = move |y: f32| Vec3::new(0.018 * (y * 2.7 + phase).sin() * y, y, 0.012 * (y * 3.3 + phase).cos() * y);
    let radius = |y: f32| (0.0115 * (1.0 - y).powf(1.1) + 0.0012) * (1.0 + 0.5 * (-y * 40.0).exp());

    let rings = if lod == 0 { 16 } else if lod == 1 { 8 } else { 4 };
    let ys: Vec<f32> = (0..=rings).map(|i| 0.93 * i as f32 / rings as f32).collect();
    let spine: Vec<Vec3> = ys.iter().map(|&y| axis(y)).collect();
    let radii: Vec<f32> = ys.iter().map(|&y| radius(y)).collect();
    bark.tube(&spine, &radii, l.trunk_sides, 12.0, 0.0, &|f| if f < 0.4 { 1.0 } else { 0.7 }, lod < 2);

    let crown_centre = Vec3::new(0.0, cb + (1.0 - cb) * 0.45, 0.0);
    let outward = move |p: Vec3| {
        let d = p - crown_centre;
        Vec3::new(d.x, d.y * 0.6 + 0.2, d.z).normalize_or_zero()
    };

    let mut azimuth = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..l.limbs {
        azimuth += 2.399_963_2 + rng.range(-0.25, 0.25);
        let f = (i as f32 + rng.range(0.0, 0.8)) / l.limbs as f32;
        let y = cb + (0.86 - cb) * f;
        let len = 0.24 * (1.0 - 0.45 * f) * rng.range(0.8, 1.15);
        let h = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        let elev = (58.0 - 20.0 * (1.0 - f) + rng.range(-8.0, 8.0)).to_radians();
        let start = axis(y);
        // the limb arcs upward and out, then bends back toward the horizontal at its end
        let limb: Vec<Vec3> = (0..=4)
            .map(|k| {
                let s = k as f32 / 4.0;
                start + h * (len * s * elev.cos()) + Vec3::Y * (len * s * elev.sin() * (1.0 - 0.45 * s))
            })
            .collect();
        if l.limb_sides > 0 {
            let r0 = radius(y) * 0.45;
            let radii: Vec<f32> = (0..=4).map(|k| r0 * (1.0 - 0.8 * k as f32 / 4.0)).collect();
            // limbs: u in 1..2 marks them for the shader's dark twig bark
            bark.tube(&limb, &radii, l.limb_sides, 12.0, 1.0, &|_| 0.75, false);
        }
        // along the limb: short weeping twigs alternating with leafy fans; a fan at its end
        for c in 0..l.clusters {
            let s = 0.35 + 0.6 * (c as f32 + 0.5) / l.clusters as f32;
            let k = (s * 4.0).min(3.999);
            let p = limb[k as usize].lerp(limb[k as usize + 1], k.fract());
            let face = rng.range(0.0, std::f32::consts::TAU);
            let across = Vec3::new(face.cos(), 0.0, face.sin());
            // mostly leafy fans (a birch's crown is a broadleaf mass), a hanging twig now and then
            if c % 5 == 0 {
                let hang = 0.1 * rng.range(0.8, 1.2) * l.card_scale;
                let width = 0.06 * l.card_scale;
                let spine = [p, p - Vec3::Y * hang * 0.5 + h * 0.01, p - Vec3::Y * hang + h * 0.02];
                foliage.ribbon(&spine, across, &[width, width * 1.1, width * 1.2], BIRCH_HANGING, 0.0, 1.0, &|f| 0.95 - 0.35 * f, &outward, 0.6);
            } else {
                let out = (h * 0.7 + across * rng.range(-0.5, 0.5) + Vec3::Y * rng.range(0.1, 0.5)).normalize();
                let side = out.cross(Vec3::Y).normalize_or_zero();
                let len = 0.13 * rng.range(0.85, 1.2) * l.card_scale;
                foliage.ribbon(&[p, p + out * len], side, &[len * 0.45, len * 0.6], BIRCH_SPRAY, 1.0, 0.0, &|_| 0.9, &outward, 0.6);
            }
        }
        let tip = *limb.last().unwrap();
        let across = Vec3::Y.cross(h).normalize();
        let fan_len = 0.14 * l.card_scale;
        let spine = [tip - h * fan_len * 0.2, tip + h * fan_len * 0.8 + Vec3::Y * 0.02];
        foliage.ribbon(&spine, across, &[fan_len * 0.45, fan_len * 0.6], BIRCH_SPRAY, 1.0, 0.0, &|_| 1.0, &outward, 0.6);
    }
    // a couple of fans at the top
    for k in 0..2 {
        let a = azimuth + k as f32 * std::f32::consts::FRAC_PI_2;
        let across = Vec3::new(a.cos(), 0.0, a.sin());
        let top = axis(0.86);
        foliage.ribbon(&[top, top + Vec3::Y * 0.1 * l.card_scale], across, &[0.06, 0.08], BIRCH_SPRAY, 1.0, 0.0, &|_| 1.0, &outward, 0.5);
    }
    TreeMesh { bark, foliage }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(t: &TreeMesh, name: &str) {
        for (part, m) in [("bark", &t.bark), ("foliage", &t.foliage)] {
            assert!(!m.indices.is_empty(), "{name} {part} empty");
            assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()), "{name} {part} index out of range");
            for v in &m.vertices {
                let p = v.position;
                assert!(p.iter().all(|c| c.is_finite()), "{name} {part} non-finite vertex");
                assert!(p[1] >= -0.01 && p[1] <= 1.05, "{name} {part} y {}", p[1]);
                assert!((p[0] * p[0] + p[2] * p[2]).sqrt() < 0.45, "{name} {part} too wide");
                assert!(v.uv[0] >= 0.0 && v.uv[0] <= 1.0 || part == "bark");
            }
        }
    }

    #[test]
    fn bark_winds_outward() {
        for t in [spruce(0, &SPRUCE_STYLES[0]), birch(0, 1)] {
            let m = &t.bark;
            for tri in m.indices.chunks_exact(3) {
                let p = |i: u32| Vec3::from_slice(&m.vertices[i as usize].position[..3]);
                let n = |i: u32| Vec3::from(m.vertices[i as usize].normal);
                let face = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0]));
                if face.length() > 1e-10 {
                    assert!(face.dot(n(tri[0]) + n(tri[1]) + n(tri[2])) > 0.0, "inward bark triangle");
                }
            }
        }
    }

    #[test]
    fn spruce_and_birch_lods_are_valid_and_get_lighter() {
        for lod in 0..3 {
            for style in &SPRUCE_STYLES {
                check(&spruce(lod, style), "spruce");
            }
            check(&birch(lod, 1), "birch");
        }
        let tris = |t: TreeMesh| t.bark.triangles() + t.foliage.triangles();
        let s: Vec<usize> = (0..3).map(|l| tris(spruce(l, &SPRUCE_STYLES[1]))).collect();
        let b: Vec<usize> = (0..3).map(|l| tris(birch(l, 1))).collect();
        assert!(s[0] > s[1] && s[1] > s[2], "spruce LOD triangles {s:?}");
        assert!(b[0] > b[1] && b[1] > b[2], "birch LOD triangles {b:?}");
        assert!(s[0] < 4000 && s[2] < 250, "spruce budget {s:?}");
    }

    #[test]
    fn spruce_crown_is_conical() {
        let t = spruce(0, &SPRUCE_STYLES[0]);
        let width_at = |y0: f32, y1: f32| {
            t.foliage.vertices.iter().filter(|v| v.position[1] >= y0 && v.position[1] < y1)
                .map(|v| (v.position[0] * v.position[0] + v.position[2] * v.position[2]).sqrt())
                .fold(0.0f32, f32::max)
        };
        let (low, mid, high) = (width_at(0.4, 0.5), width_at(0.6, 0.7), width_at(0.85, 0.95));
        assert!(low > mid && mid > high, "crown radii {low} {mid} {high}");
        assert!(width_at(0.0, 0.06) == 0.0, "no foliage on the bare lower bole");
    }
}
