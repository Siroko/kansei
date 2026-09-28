use super::cards::*;
use super::tests::rock;
use super::*;
use crate::geometries::Geometry;
/// A quad (two triangles, its own four vertices) centred at `c`, in the plane of `u` and `v`
/// (half extents), normal u × v.
pub(super) fn quad(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, c: Vec3, u: Vec3, v: Vec3) {
    let base = vertices.len() as u32;
    let n = u.cross(v).normalize();
    for (i, (su, sv)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
        let p = c + u * su + v * sv;
        vertices.push(Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: [(i == 1 || i == 2) as u32 as f32, (i >= 2) as u32 as f32] });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// A tree's crown of `count` cards: quads over a cone 10 m tall and 3 m wide at its base, each
/// tilted at random (deterministic).
pub(super) fn crown(count: u32) -> Geometry {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let mut seed = 11u32;
    let mut r = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    for _ in 0..count {
        let (h, a) = (r(), r() * std::f32::consts::TAU);
        let radius = 3.0 * (1.0 - h) * (0.6 + 0.4 * r());
        let c = Vec3::new(radius * a.cos(), 10.0 * h, radius * a.sin());
        let out = Vec3::new(a.cos(), 0.3, a.sin()).normalize();
        let side = Vec3::Y.cross(out).normalize();
        let tilt = r() - 0.5;
        let (u, v) = (side * 0.35, (out * tilt + Vec3::Y * (1.0 - tilt.abs())).normalize() * 0.25);
        quad(&mut vertices, &mut indices, c, u, v);
    }
    Geometry::new("crown", vertices, indices)
}

fn welded(geometry: &Geometry) -> (Vec<f32>, Vec<u32>) {
    let positions: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
    let ids = super::build::position_ids_for_tests(&positions);
    (positions, ids)
}

#[test]
fn small_open_flat_components_are_cards() {
    // 50 cards, a closed rock, an open tube (24 triangles: its normals cancel) and a flat grid
    // too large to be a card
    let mut g = crown(50);
    let rock = rock(2, false);
    let base = g.vertices.len() as u32;
    g.vertices.extend(rock.vertices.iter().cloned());
    g.indices.extend(rock.indices.iter().map(|i| i + base));
    let base = g.vertices.len() as u32;
    for k in 0..=12u32 {
        let a = k as f32 / 12.0 * std::f32::consts::TAU;
        for y in [0.0, 2.0] {
            g.vertices.push(Vertex { position: [20.0 + a.cos(), y, a.sin(), 1.0], normal: [a.cos(), 0.0, a.sin()], uv: [0.0; 2] });
        }
    }
    for k in 0..12u32 {
        let (a, b, c, d) = (base + 2 * k, base + 2 * k + 1, base + 2 * k + 2, base + 2 * k + 3);
        g.indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
    let base = g.vertices.len() as u32;
    for j in 0..=10u32 {
        for i in 0..=10u32 {
            g.vertices.push(Vertex { position: [-20.0 + i as f32, 0.0, j as f32, 1.0], normal: [0.0, 1.0, 0.0], uv: [0.0; 2] });
        }
    }
    for j in 0..10u32 {
        for i in 0..10u32 {
            let (a, b, c, d) = (base + j * 11 + i, base + j * 11 + i + 1, base + (j + 1) * 11 + i, base + (j + 1) * 11 + i + 1);
            g.indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let (positions, ids) = welded(&g);
    let (cards, rest) = find_cards(&g.indices, &positions, &ids, &ClusterOptions { cards: true, ..Default::default() });
    assert_eq!(cards.len(), 50);
    assert!(cards.iter().all(|c| c.triangles.len() == 2 && c.vertices.len() == 4 && (c.area - 0.35).abs() < 1e-3 && c.radius > 0.3), "{:?}", cards.iter().map(|c| (c.triangles.len(), c.area)).collect::<Vec<_>>());
    assert_eq!(rest.len() / 3, rock.indices.len() / 3 + 24 + 200);
    // a zero-area triangle doesn't break a card
    let mut g = crown(3);
    g.indices.extend_from_slice(&[0, 0, 1]);
    let (positions, ids) = welded(&g);
    let (cards, _) = find_cards(&g.indices, &positions, &ids, &ClusterOptions { cards: true, ..Default::default() });
    assert_eq!(cards.len(), 3);
}
