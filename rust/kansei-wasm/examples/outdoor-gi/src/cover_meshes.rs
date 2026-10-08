//! Ground-cover meshes, in metres at instance scale 1, standing on the origin (Y up). Cards
//! carry region-local UVs (0..1; the shader maps them into the instance's atlas region, v = 1 at
//! the ground), `position.w` is ambient occlusion (dark at the roots), and normals lean up so
//! the cover is lit like a field rather than like flat cards.
//!
//! From the Raggare intro's renderer (raggare-web's crates/intro/src/undergrowth/geometry.rs, the same author's).

use glam::{Vec2, Vec3};
use kansei_core::geometries::Vertex;

use crate::canvas::Rng;
use crate::tree_meshes::Mesh;

#[allow(clippy::too_many_arguments)]
fn push_card(m: &mut Mesh, centre: Vec3, across: Vec3, up: Vec3, width: f32, height: f32, flip: bool, ao_base: f32) {
    let base = m.vertices.len() as u32;
    let n = across.cross(up).normalize_or_zero();
    // mostly upward, a little of the card's own facing
    let normal = (Vec3::Y * 0.75 + n * 0.25).normalize();
    let (u0, u1) = if flip { (1.0, 0.0) } else { (0.0, 1.0) };
    for (s, t, u, v) in [(-0.5, 0.0, u0, 1.0), (0.5, 0.0, u1, 1.0), (-0.5, 1.0, u0, 0.0), (0.5, 1.0, u1, 0.0)] {
        let p = centre + across * (s * width) + up * (t * height);
        let ao = ao_base + (1.0 - ao_base) * t;
        m.vertices.push(Vertex { position: [p.x, p.y, p.z, ao], normal: normal.to_array(), uv: Vec2::new(u, v).to_array() });
    }
    m.indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
}

/// Crossed vertical cards through the centre, fanned around Y, each leaning a little outward.
pub fn crossed(cards: usize, width: f32, height: f32, lean: f32, seed: u32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut m = Mesh::default();
    for i in 0..cards {
        let a = i as f32 / cards as f32 * std::f32::consts::PI + rng.range(-0.15, 0.15);
        let across = Vec3::new(a.cos(), 0.0, a.sin());
        let out = Vec3::new(-a.sin(), 0.0, a.cos()) * rng.range(-lean, lean);
        let up = (Vec3::Y + out).normalize();
        push_card(&mut m, Vec3::ZERO, across, up, width, height * rng.range(0.9, 1.1), i % 2 == 1, 0.35);
    }
    m
}

/// A low shrub: leafy cards round an ellipsoid, facing out and up.
pub fn shrub(cards: usize, radius: f32, height: f32, seed: u32) -> Mesh {
    let mut rng = Rng::new(seed);
    let mut m = Mesh::default();
    let golden = 2.399_963_2_f32;
    for i in 0..cards {
        let f = (i as f32 + 0.5) / cards as f32;
        let a = i as f32 * golden;
        let y = height * (0.15 + 0.6 * f);
        let r = radius * (1.0 - (f - 0.35).abs()).clamp(0.4, 1.0) * rng.range(0.7, 1.0);
        let dir = Vec3::new(a.cos(), 0.0, a.sin());
        let centre = dir * r * 0.4 + Vec3::Y * y * 0.6;
        let up = (Vec3::Y * 0.8 + dir * 0.6).normalize();
        let across = Vec3::Y.cross(dir).normalize();
        push_card(&mut m, centre, across, up, radius * 1.1, height * 0.75, i % 2 == 0, 0.5);
    }
    m
}

/// Grass tuft, flowers and shrub meshes: (near, far).
pub struct CoverMeshes {
    pub grass: [Mesh; 2],
    pub flowers: [Mesh; 2],
    pub shrub: [Mesh; 2],
}

pub fn meshes() -> CoverMeshes {
    CoverMeshes {
        // Light Foliage's verge clumps, which the Unreal frames show as a continuous thicket along
        // the ditch: wide, dense and up to about 0.9 m tall
        grass: [crossed(8, 1.35, 0.9, 0.25, 1), crossed(4, 1.35, 0.86, 0.15, 2)],
        flowers: [crossed(4, 0.95, 0.62, 0.15, 3), crossed(3, 0.95, 0.6, 0.1, 4)],
        shrub: [shrub(18, 0.75, 1.0, 5), shrub(7, 0.8, 1.0, 6)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_meshes_are_valid_and_stand_on_the_ground() {
        let m = meshes();
        for (name, pair) in [("grass", &m.grass), ("flowers", &m.flowers), ("shrub", &m.shrub)] {
            assert!(pair[0].triangles() > pair[1].triangles(), "{name}: far LOD lighter");
            for mesh in pair.iter() {
                assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.vertices.len()));
                let min_y = mesh.vertices.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
                let max_y = mesh.vertices.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);
                assert!(min_y > -1e-3 && min_y < 0.2 && max_y > 0.3 && max_y < 1.3, "{name}: y {min_y}..{max_y}");
                assert!(mesh.vertices.iter().all(|v| v.uv[0] >= 0.0 && v.uv[0] <= 1.0 && v.uv[1] >= 0.0 && v.uv[1] <= 1.0));
            }
        }
    }
}
