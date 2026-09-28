use super::*;
use glam::Vec3;

#[test]
fn an_enclosing_sphere_contains_every_sphere() {
    let spheres = [
        Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
        Sphere { center: Vec3::new(3.0, 0.0, 0.0), radius: 0.5 },
        Sphere { center: Vec3::new(0.0, -2.0, 1.0), radius: 2.0 },
        Sphere { center: Vec3::new(0.5, 0.0, 0.0), radius: 0.1 },
    ];
    let s = Sphere::enclosing(spheres);
    for o in &spheres {
        assert!(s.contains(o), "{s:?} misses {o:?}");
    }
    // one inside another: the outer one
    let inner = Sphere { center: Vec3::ZERO, radius: 0.5 };
    let outer = Sphere { center: Vec3::new(0.1, 0.0, 0.0), radius: 2.0 };
    assert_eq!(Sphere::enclosing([inner, outer]), outer);
}

use crate::geometries::{Geometry, Vertex};
use std::collections::HashMap;

/// A noisy icosphere (a rock) with `subdivisions`. With `seam`, the triangles on the x < 0 side
/// get their own vertices with other uvs, so a uv seam runs round the rock (split vertices at
/// one position).
fn rock(subdivisions: u32, seam: bool) -> Geometry {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut p: Vec<Vec3> = [[-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0], [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t], [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]]
        .iter()
        .map(|v| Vec3::from(*v).normalize())
        .collect();
    let mut f: Vec<[u32; 3]> = vec![[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for _ in 0..subdivisions {
        let mut mid = HashMap::new();
        let mut next = Vec::new();
        for [a, b, c] in f {
            let mut m = |x: u32, y: u32| {
                *mid.entry((x.min(y), x.max(y))).or_insert_with(|| {
                    p.push(((p[x as usize] + p[y as usize]) * 0.5).normalize());
                    p.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (m(a, b), m(b, c), m(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = next;
    }
    let bump = |v: Vec3| v * (1.0 + 0.08 * (7.0 * v.x).sin() * (5.0 * v.y).cos() + 0.05 * (11.0 * v.z).sin());
    let vertex = |v: Vec3, u: f32| {
        let b = bump(v);
        Vertex { position: [b.x, b.y, b.z, 1.0], normal: v.to_array(), uv: [v.x * 0.5 + u, v.y * 0.5 + 0.5] }
    };
    let mut vertices: Vec<Vertex> = p.iter().map(|&v| vertex(v, 0.5)).collect();
    if seam {
        // the x < 0 side's own copies, with uvs a whole unit over
        let copies: Vec<u32> = p
            .iter()
            .map(|&v| {
                vertices.push(vertex(v, 1.5));
                vertices.len() as u32 - 1
            })
            .collect();
        for tri in f.iter_mut() {
            if tri.iter().map(|&i| p[i as usize]).sum::<Vec3>().x < 0.0 {
                for i in tri.iter_mut() {
                    *i = copies[*i as usize];
                }
            }
        }
    }
    Geometry::new("rock", vertices, f.into_iter().flatten().collect())
}

#[test]
fn level_zero_is_the_mesh_in_clusters() {
    let rock = rock(4, false);
    let mesh = ClusterMesh::build(&rock, &ClusterOptions::default());
    let level0: Vec<usize> = (0..mesh.clusters.len()).filter(|&i| mesh.clusters[i].level == 0).collect();
    let triangles: usize = level0.iter().map(|&i| mesh.clusters[i].triangle_count as usize).sum();
    assert_eq!(triangles, rock.indices.len() / 3);
    for &i in &level0 {
        let c = &mesh.clusters[i];
        assert!(c.triangle_count <= 124 && c.vertex_count <= 64);
        assert_eq!(c.error, 0.0);
        for t in mesh.triangles(i) {
            for v in t {
                let p = mesh.vertices[v as usize].position;
                assert!(c.bounds.center.distance(Vec3::new(p[0], p[1], p[2])) <= c.bounds.radius * 1.0001);
            }
        }
    }
}

#[test]
fn tiny_and_empty_meshes() {
    let empty = ClusterMesh::build(&Geometry::new("empty", Vec::new(), Vec::new()), &ClusterOptions::default());
    assert!(empty.clusters.is_empty());
    let v = |x: f32, y: f32| Vertex { position: [x, y, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [x, y] };
    let one = ClusterMesh::build(&Geometry::new("one", vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0)], vec![0, 1, 2]), &ClusterOptions::default());
    assert_eq!(one.clusters.len(), 1);
    assert_eq!(one.triangles(0).collect::<Vec<_>>(), vec![[0, 1, 2]]);
}

#[test]
fn levels_shrink_to_a_root_and_errors_and_bounds_nest() {
    let mesh = ClusterMesh::build(&rock(5, false), &ClusterOptions::default());
    let levels = mesh.clusters.iter().map(|c| c.level).max().unwrap();
    let per_level: Vec<u32> = (0..=levels).map(|l| mesh.clusters.iter().filter(|c| c.level == l).map(|c| c.triangle_count).sum()).collect();
    assert!(levels >= 4, "{per_level:?}");
    assert!(per_level.windows(2).all(|w| w[1] < w[0]), "{per_level:?}");
    let root_triangles: u32 = mesh.clusters.iter().filter(|c| c.parent_error.is_infinite()).map(|c| c.triangle_count).sum();
    assert!(root_triangles <= 4 * 124, "{root_triangles} root triangles");
    for c in &mesh.clusters {
        assert!(c.parent_error >= c.error);
        assert!(c.parent_bounds.contains(&c.lod_bounds));
    }
}

#[test]
fn open_and_disjoint_meshes_terminate() {
    // a field of disjoint quads (cards): nothing to collapse, and the build still ends
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for i in 0..2000u32 {
        let (x, z) = ((i % 50) as f32, (i / 50) as f32);
        let base = vertices.len() as u32;
        for (dx, dy) in [(0.0, 0.0), (0.8, 0.0), (0.8, 0.8), (0.0, 0.8)] {
            vertices.push(Vertex { position: [x + dx, dy, z, 1.0], normal: [0.0, 0.0, 1.0], uv: [dx, dy] });
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let cards = ClusterMesh::build(&Geometry::new("cards", vertices, indices), &ClusterOptions::default());
    assert!(!cards.clusters.is_empty());
    // an open grid reduces: its outline isn't locked
    let n = 120u32;
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for z in 0..=n {
        for x in 0..=n {
            let (fx, fz) = (x as f32 / n as f32, z as f32 / n as f32);
            vertices.push(Vertex { position: [fx, 0.05 * (fx * 9.0).sin() * (fz * 7.0).cos(), fz, 1.0], normal: [0.0, 1.0, 0.0], uv: [fx, fz] });
        }
    }
    for z in 0..n {
        for x in 0..n {
            let i = z * (n + 1) + x;
            indices.extend([i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
        }
    }
    let grid = ClusterMesh::build(&Geometry::new("grid", vertices, indices), &ClusterOptions::default());
    assert!(grid.clusters.iter().any(|c| c.level >= 3));
}

#[test]
fn flat_shaded_meshes_keep_one_level() {
    // every triangle its own vertices with its face normal: every edge is a seam, and the
    // attribute-preserving simplifier moves none of them
    let smooth = rock(3, false);
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for t in smooth.indices.chunks(3) {
        let p: Vec<Vec3> = t.iter().map(|&i| Vec3::from_slice(&smooth.vertices[i as usize].position[..3])).collect();
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
        for q in p {
            indices.push(vertices.len() as u32);
            vertices.push(Vertex { position: [q.x, q.y, q.z, 1.0], normal: n.to_array(), uv: [0.0, 0.0] });
        }
    }
    let mesh = ClusterMesh::build(&Geometry::new("flat", vertices, indices), &ClusterOptions::default());
    assert!(mesh.clusters.iter().all(|c| c.level == 0 && c.parent_error.is_infinite()));
}
