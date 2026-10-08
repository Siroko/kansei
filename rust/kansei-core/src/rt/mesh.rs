//! Meshes as the grid's gather reads them, and miaumiau.cat/?p=1457's midpoint split.

use glam::{Mat4, Vec3};

use crate::geometries::{Geometry, Vertex};

/// Words before an `RtMesh`'s vertices: where its vertices and indices start, how many
/// vertices and triangles it has.
const HEADER_WORDS: usize = 4;

/// A mesh for `RtGrid::gather`: its vertices' positions, uvs and normals, its triangles, and its
/// bounds.
#[derive(Clone, Debug, Default)]
pub struct RtMesh {
    pub positions: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// (what `RtSurface::with_smooth_normals` interpolates at a hit)
    pub normals: Vec<[f32; 3]>,
    /// Three vertex indices a triangle.
    pub indices: Vec<u32>,
    pub min: Vec3,
    pub max: Vec3,
}

impl RtMesh {
    /// `geometry`'s CPU vertices and indices.
    pub fn from_geometry(geometry: &Geometry) -> Self {
        let positions: Vec<[f32; 3]> = geometry.vertices.iter().map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &positions {
            min = min.min(Vec3::from(*p));
            max = max.max(Vec3::from(*p));
        }
        if positions.is_empty() {
            (min, max) = (Vec3::ZERO, Vec3::ZERO);
        }
        Self { uvs: geometry.vertices.iter().map(|v| v.uv).collect(), normals: geometry.vertices.iter().map(|v| v.normal).collect(), positions, indices: geometry.indices.clone(), min, max }
    }

    pub fn triangle_count(&self) -> u32 {
        (self.indices.len() / 3) as u32
    }

    /// The mesh as the gather reads it: a header (where the vertices and indices start, the
    /// vertex and triangle counts), the vertices (x, y, z, the uv as two f16 and the normal
    /// octahedral, as two snorm16), the indices.
    pub fn gpu_words(&self) -> Vec<u32> {
        let vertices = HEADER_WORDS;
        let indices = vertices + self.positions.len() * 5;
        let mut words = Vec::with_capacity(indices + self.indices.len());
        words.extend([vertices as u32, indices as u32, self.positions.len() as u32, self.triangle_count()]);
        for (k, (p, uv)) in self.positions.iter().zip(&self.uvs).enumerate() {
            let n = self.normals.get(k).copied().unwrap_or([0.0, 1.0, 0.0]);
            words.extend([p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), pack_half2(*uv), pack_octahedral(n)]);
        }
        words.extend_from_slice(&self.indices);
        words
    }

    /// A STORAGE buffer of `gpu_words`.
    pub fn create_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("RtMesh"), contents: bytemuck::cast_slice(&self.gpu_words()), usage: wgpu::BufferUsages::STORAGE })
    }
}

/// A unit vector, octahedral, as WGSL's `pack2x16snorm` packs the two coordinates
/// (`kansei_rt_unpack_normal` reads it).
fn pack_octahedral(n: [f32; 3]) -> u32 {
    let v = Vec3::from(n);
    let v = if v.length_squared() > 0.0 { v / (v.x.abs() + v.y.abs() + v.z.abs()) } else { Vec3::Y };
    let (mut x, mut y) = (v.x, v.y);
    if v.z < 0.0 {
        (x, y) = ((1.0 - v.y.abs()) * v.x.signum(), (1.0 - v.x.abs()) * v.y.signum());
    }
    let snorm = |f: f32| ((f.clamp(-1.0, 1.0) * 32767.0).round() as i32 as u32) & 0xffff;
    snorm(x) | snorm(y) << 16
}

/// Two f32 as WGSL's `pack2x16float` packs them (round to nearest even).
pub(crate) fn pack_half2(v: [f32; 2]) -> u32 {
    half_bits(v[0]) as u32 | (half_bits(v[1]) as u32) << 16
}

/// An f32 as IEEE half bits, rounded to nearest even (overflow to infinity, small values to
/// subnormals or zero).
pub(crate) fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let abs = b & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        // infinity, NaN
        return sign | 0x7c00 | if abs > 0x7f80_0000 { 0x200 } else { 0 };
    }
    let exp = (abs >> 23) as i32 - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        // a subnormal half
        let mant = (abs & 0x7f_ffff) | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = mant >> shift;
        let rest = mant & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = (rest > mid || (rest == mid && half & 1 == 1)) as u32;
        return sign | (half + round) as u16;
    }
    let mant = abs & 0x7f_ffff;
    let half = ((exp as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    let round = (rest > 0x1000 || (rest == 0x1000 && half & 1 == 1)) as u32;
    sign | (half + round) as u16
}

/// miaumiau.cat/?p=1457's split, at load: every triangle with an edge longer than `max_edge` cut
/// into four at its edges' midpoints, again until none is (midpoints shared, so no cracks). A
/// grid's build then scatters each triangle into a few cells, at the cost of more triangles.
pub fn split_large_triangles(geometry: &Geometry, max_edge: f32) -> Geometry {
    use std::collections::HashMap;
    let mut vertices: Vec<Vertex> = geometry.vertices.clone();
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
    let mut mid = |vertices: &mut Vec<Vertex>, a: u32, b: u32| -> u32 {
        *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
            let (va, vb) = (vertices[a as usize], vertices[b as usize]);
            let lerp = |x: &[f32], y: &[f32], out: &mut [f32]| out.iter_mut().zip(x.iter().zip(y)).for_each(|(o, (p, q))| *o = (p + q) * 0.5);
            let mut v = va;
            lerp(&va.position, &vb.position, &mut v.position);
            lerp(&va.normal, &vb.normal, &mut v.normal);
            lerp(&va.uv, &vb.uv, &mut v.uv);
            vertices.push(v);
            (vertices.len() - 1) as u32
        })
    };
    let pos = |vertices: &[Vertex], i: u32| Vec3::from_slice(&vertices[i as usize].position[..3]);
    let mut pending: Vec<[u32; 3]> = geometry.indices.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
    pending.reverse();
    let mut indices = Vec::with_capacity(geometry.indices.len());
    while let Some([a, b, c]) = pending.pop() {
        let (pa, pb, pc) = (pos(&vertices, a), pos(&vertices, b), pos(&vertices, c));
        if pa.distance(pb).max(pb.distance(pc)).max(pc.distance(pa)) <= max_edge {
            indices.extend_from_slice(&[a, b, c]);
            continue;
        }
        let (ab, bc, ca) = (mid(&mut vertices, a, b), mid(&mut vertices, b, c), mid(&mut vertices, c, a));
        pending.extend_from_slice(&[[ab, bc, ca], [ca, bc, c], [ab, b, bc], [a, ab, ca]]);
    }
    Geometry::new(&geometry.label, vertices, indices)
}

/// The world box of the box `min..max` under `m`.
pub fn transform_box(m: &Mat4, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    let centre = m.transform_point3((min + max) * 0.5);
    let half = (max - min) * 0.5;
    let reach = m.x_axis.truncate().abs() * half.x + m.y_axis.truncate().abs() * half.y + m.z_axis.truncate().abs() * half.z;
    (centre - reach, centre + reach)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::{BoxGeometry, PlaneGeometry};

    #[test]
    fn normals_pack_as_kansei_rt_unpack_normal_reads_them() {
        // a CPU twin of rt_types.wgsl's kansei_rt_unpack_normal
        let unpack = |w: u32| {
            let snorm = |h: u32| ((h & 0xffff) as u16 as i16 as f32 / 32767.0).max(-1.0);
            let (x, y) = (snorm(w), snorm(w >> 16));
            let mut v = Vec3::new(x, y, 1.0 - x.abs() - y.abs());
            if v.z < 0.0 {
                v = Vec3::new((1.0 - v.y.abs()) * if v.x >= 0.0 { 1.0 } else { -1.0 }, (1.0 - v.x.abs()) * if v.y >= 0.0 { 1.0 } else { -1.0 }, v.z);
            }
            v.normalize()
        };
        for n in [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.3, -0.5, 0.81], [-0.7, 0.1, -0.7], [1.0, 1.0, 1.0]] {
            let n = Vec3::from(n).normalize();
            let back = unpack(pack_octahedral(n.into()));
            assert!(back.dot(n) > 0.9999, "{n} came back {back}");
        }
    }

    #[test]
    fn half_bits_round_like_pack2x16float() {
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(0.5), 0x3800);
        assert_eq!(half_bits(2.0), 0x4000);
        assert_eq!(half_bits(-2.0), 0xc000);
        assert_eq!(half_bits(0.0), 0);
        assert_eq!(half_bits(65504.0), 0x7bff);
        assert_eq!(half_bits(1e6), 0x7c00);
        // 1 + 2^-11 is halfway between 1 and the next half: to even (1)
        assert_eq!(half_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(half_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        // the smallest subnormal half
        assert_eq!(half_bits(2f32.powi(-24)), 1);
    }

    #[test]
    fn the_split_keeps_the_surface_and_bounds_the_edges() {
        let plane = PlaneGeometry::new(4.0, 4.0);
        let split = split_large_triangles(&plane, 0.5);
        let area = |g: &Geometry| {
            g.indices
                .chunks(3)
                .map(|t| {
                    let p = |i: u32| Vec3::from_slice(&g.vertices[i as usize].position[..3]);
                    (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5
                })
                .sum::<f32>()
        };
        assert!((area(&split) - area(&plane)).abs() < 1e-3);
        for t in split.indices.chunks(3) {
            let p = |i: u32| Vec3::from_slice(&split.vertices[t[i as usize] as usize].position[..3]);
            assert!(p(0).distance(p(1)).max(p(1).distance(p(2))).max(p(2).distance(p(0))) <= 0.5 + 1e-5);
        }
        // every child keeps its parent's winding
        let normal = |g: &Geometry, t: &[u32]| {
            let p = |i: usize| Vec3::from_slice(&g.vertices[t[i] as usize].position[..3]);
            (p(1) - p(0)).cross(p(2) - p(0)).normalize()
        };
        let up = normal(&plane, &plane.indices[..3]);
        assert!(split.indices.chunks(3).all(|t| normal(&split, t).dot(up) > 0.99));
        // shared midpoints: no more vertices than a crack-free split needs
        let cube = split_large_triangles(&BoxGeometry::new(1.0, 1.0, 1.0), 0.3);
        assert!(cube.vertices.len() < cube.indices.len());
    }

    #[test]
    fn transformed_boxes_hold_the_transformed_corners() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)) * Mat4::from_rotation_y(0.7) * Mat4::from_scale(Vec3::new(2.0, 1.0, 0.5));
        let (lo, hi) = transform_box(&m, Vec3::new(-1.0, 0.0, -2.0), Vec3::new(1.0, 3.0, 2.0));
        for k in 0..8 {
            let c = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { 0.0 } else { 3.0 }, if k & 4 == 0 { -2.0 } else { 2.0 });
            let w = m.transform_point3(c);
            assert!(w.cmpge(lo - 1e-4).all() && w.cmple(hi + 1e-4).all());
        }
    }
}
