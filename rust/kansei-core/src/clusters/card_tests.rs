use super::cards::*;
use super::tests::{bad_edges_keyed, eyes, position_keys, rock, view};
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

/// The drawn cards' area (their triangles', scaled copies included) per cell of a `cells`³ grid
/// over `bounds` (lo, hi), and in all.
fn area_per_cell(mesh: &ClusterMesh, cut: &[usize], (lo, hi): (Vec3, Vec3), cells: usize) -> (Vec<f32>, f32) {
    let mut per = vec![0.0f32; cells * cells * cells];
    let mut total = 0.0;
    for &c in cut {
        for [a, b, d] in mesh.triangles(c) {
            let p = |v: u32| Vec3::from_slice(&mesh.vertices[v as usize].position[..3]);
            let area = 0.5 * (p(b) - p(a)).cross(p(d) - p(a)).length();
            let centre = (p(a) + p(b) + p(d)) / 3.0;
            let cell = ((centre - lo) / (hi - lo) * cells as f32).clamp(Vec3::ZERO, Vec3::splat(cells as f32 - 1.0)).as_uvec3();
            per[(cell.z as usize * cells + cell.y as usize) * cells + cell.x as usize] += area;
            total += area;
        }
    }
    (per, total)
}

fn card_options() -> ClusterOptions {
    ClusterOptions { cards: true, ..Default::default() }
}

#[test]
fn pruned_levels_hold_the_cards_area() {
    let mesh = ClusterMesh::build(&crown(800), &card_options());
    assert!(mesh.clusters.iter().all(|c| c.card));
    let levels = mesh.levels();
    assert!(levels.len() >= 4, "{} levels", levels.len());
    let bounds = (Vec3::new(-3.5, -0.5, -3.5), Vec3::new(3.5, 10.5, 3.5));
    let level0: Vec<usize> = (0..mesh.clusters.len()).filter(|&i| mesh.clusters[i].level == 0).collect();
    let (cells0, total0) = area_per_cell(&mesh, &level0, bounds, 3);
    let mut seed = 5;
    let mut triangles = Vec::new();
    for eye in eyes(30, 5.0, 3000.0, &mut seed).into_iter().chain([Vec3::new(0.0, 5.0, 40.0), Vec3::new(0.0, 5.0, 4000.0)]) {
        for threshold in [0.5, 1.0, 4.0] {
            let cut = mesh.select(&view(eye, threshold));
            let (cells, total) = area_per_cell(&mesh, &cut, bounds, 3);
            assert!((total / total0 - 1.0).abs() < 0.1, "eye {eye}, {threshold} px: {total} of {total0}");
            // (a cell is held to it where it keeps enough cards for the share to mean something:
            // eight at the cut's reduction)
            let kept = cut.iter().map(|&c| mesh.clusters[c].triangle_count).sum::<u32>() as f32 / 1600.0;
            for (k, (&a, &a0)) in cells.iter().zip(&cells0).enumerate() {
                if a0 / 0.35 * kept >= 8.0 {
                    assert!((a / a0 - 1.0).abs() < 0.35, "eye {eye}, {threshold} px: cell {k} holds {a} of {a0}");
                }
            }
            triangles.push(cut.iter().map(|&c| mesh.clusters[c].triangle_count).sum::<u32>());
        }
    }
    // from 4 km at 1 px, a small share of the cards
    let far = triangles[triangles.len() - 2];
    assert!(far * 8 < 1600, "{far} triangles from 4 km");
    // and there each part of the crown keeps about its area: a kept card is paired with its
    // nearest and drawn at the centre of the area it stands for
    let cut = mesh.select(&view(Vec3::new(0.0, 5.0, 4000.0), 1.0));
    let (cells, _) = area_per_cell(&mesh, &cut, bounds, 3);
    for (k, (&a, &a0)) in cells.iter().zip(&cells0).enumerate() {
        if a0 > 0.05 * total0 {
            assert!((a / a0 - 1.0).abs() <= 0.2, "from 4 km, cell {k} holds {a} of {a0}");
        }
    }
}

#[test]
fn levels_of_cards_nest_like_simplified_levels() {
    let mesh = ClusterMesh::build(&crown(800), &card_options());
    for c in &mesh.clusters {
        if c.parent_error.is_finite() {
            assert!(c.parent_error >= c.error, "{} < {}", c.parent_error, c.error);
            assert!(c.parent_bounds.contains(&c.lod_bounds));
        }
    }
    let roots = mesh.clusters.iter().filter(|c| !c.parent_error.is_finite()).count();
    assert!(roots <= 4, "{roots} roots");
}

#[test]
fn a_mixed_mesh_prunes_its_cards_and_simplifies_the_rest() {
    // a rock under a crown of cards, as one mesh
    let mut g = crown(400);
    let rock = rock(4, false);
    let base = g.vertices.len() as u32;
    g.vertices.extend(rock.vertices.iter().map(|v| Vertex { position: [v.position[0], v.position[1] - 2.0, v.position[2], 1.0], ..*v }));
    g.indices.extend(rock.indices.iter().map(|i| i + base));
    let mesh = ClusterMesh::build(&g, &card_options());
    assert!(mesh.clusters.iter().any(|c| c.card) && mesh.clusters.iter().any(|c| !c.card));
    let keys = position_keys(&mesh, 1e-5);
    let mut seed = 9;
    for eye in eyes(20, 4.0, 1000.0, &mut seed) {
        for threshold in [0.5, 2.0] {
            let cut = mesh.select(&view(eye, threshold));
            let solid: Vec<usize> = cut.iter().copied().filter(|&c| !mesh.clusters[c].card).collect();
            assert_eq!(bad_edges_keyed(&mesh, &keys, &solid), 0, "eye {eye}: the rock's cut is open");
        }
    }
    // without `cards`, the same mesh is all solid
    let plain = ClusterMesh::build(&g, &ClusterOptions::default());
    assert!(plain.clusters.iter().all(|c| !c.card));
}

#[test]
fn the_card_error_scale_moves_the_switch_nearer() {
    let eye = Vec3::new(0.0, 5.0, 150.0);
    let triangles = |scale: f32| {
        let mesh = ClusterMesh::build(&crown(800), &ClusterOptions { card_error_scale: scale, ..card_options() });
        mesh.select(&view(eye, 1.0)).iter().map(|&c| mesh.clusters[c].triangle_count).sum::<u32>()
    };
    let (full, quarter) = (triangles(1.0), triangles(0.25));
    assert!(quarter < full, "{quarter} with a quarter of the error, {full} with all of it");
}
