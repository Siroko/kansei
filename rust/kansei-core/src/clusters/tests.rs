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
pub(super) fn rock(subdivisions: u32, seam: bool) -> Geometry {
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
        let options = ClusterOptions::default();
        assert!(c.triangle_count <= options.max_triangles as u32 && c.vertex_count <= options.max_vertices as u32);
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

/// One id per point of `mesh.vertices`: positions within `tolerance` of each other share one (a
/// seam's copies that differ by rounding or by the sign of zero are one point to a viewer).
pub(super) fn position_keys(mesh: &ClusterMesh, tolerance: f32) -> Vec<u32> {
    let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let mut points: Vec<Vec3> = Vec::new();
    mesh.vertices
        .iter()
        .map(|v| {
            let p = Vec3::from_slice(&v.position[..3]);
            let c = (p / tolerance).floor();
            let c = [c.x as i64, c.y as i64, c.z as i64];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(near) = cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            if let Some(&k) = near.iter().find(|&&k| points[k as usize].distance(p) <= tolerance) {
                                return k;
                            }
                        }
                    }
                }
            }
            points.push(p);
            let k = points.len() as u32 - 1;
            cells.entry(c).or_default().push(k);
            k
        })
        .collect()
}

/// Edges of `clusters`' triangles that betray a hole or an overlap: used an odd number of times
/// (a hole's rim), or shared by other than exactly two clusters once each (an overlap). Points
/// are compared within 1e-5 (`position_keys`), and triangles with two corners at one point
/// (zero area, as a UV sphere has at its poles) cover nothing and are skipped. A fold the
/// simplifier left inside one cluster (an edge used 4 times, all in that cluster) is neither.
fn bad_edges(mesh: &ClusterMesh, clusters: &[usize]) -> usize {
    bad_edges_keyed(mesh, &position_keys(mesh, 1e-5), clusters)
}

/// `bad_edges` with the mesh's `position_keys` at hand.
pub(super) fn bad_edges_keyed(mesh: &ClusterMesh, keys: &[u32], clusters: &[usize]) -> usize {
    let mut uses: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for &c in clusters {
        for t in mesh.triangles(c) {
            let k = t.map(|v| keys[v as usize]);
            if k[0] == k[1] || k[1] == k[2] || k[2] == k[0] {
                continue;
            }
            for (a, b) in [(k[0], k[1]), (k[1], k[2]), (k[2], k[0])] {
                uses.entry((a.min(b), a.max(b))).or_default().push(c);
            }
        }
    }
    uses.values()
        .filter(|u| {
            let crossing = u.iter().any(|&c| c != u[0]);
            u.len() % 2 == 1 || (crossing && u.len() != 2)
        })
        .count()
}

/// The total area of `clusters`' triangles.
fn area(mesh: &ClusterMesh, clusters: &[usize]) -> f32 {
    let p = |v: u32| Vec3::from_slice(&mesh.vertices[v as usize].position[..3]);
    clusters.iter().flat_map(|&c| mesh.triangles(c)).map(|t| (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5).sum()
}

/// A seeded sequence in [0, 1).
fn random(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

/// `count` eyes round the origin, in random directions, from `near` to `far` (log-uniform).
pub(super) fn eyes(count: usize, near: f32, far: f32, seed: &mut u32) -> Vec<Vec3> {
    (0..count)
        .map(|_| {
            let (z, a) = (random(seed) * 2.0 - 1.0, random(seed) * std::f32::consts::TAU);
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * a.cos(), z, r * a.sin()) * near * (far / near).powf(random(seed))
        })
        .collect()
}

/// Asserts every cut of `mesh` seen from `eyes` at `budgets` is non-empty, covers about the
/// mesh's area, and has no hole or overlap.
fn assert_cuts_closed(name: &str, mesh: &ClusterMesh, eyes: &[Vec3], budgets: &[f32]) {
    let level0: Vec<usize> = (0..mesh.clusters.len()).filter(|&i| mesh.clusters[i].level == 0).collect();
    let whole = area(mesh, &level0);
    let keys = position_keys(mesh, 1e-5);
    for &eye in eyes {
        for &threshold in budgets {
            let cut = mesh.select(&view(eye, threshold));
            assert!(!cut.is_empty(), "{name}: eye {eye}, budget {threshold}: an empty cut");
            let covered = area(mesh, &cut) / whole;
            assert!((0.8..1.2).contains(&covered), "{name}: eye {eye}, budget {threshold}: the cut covers {covered} of the mesh");
            let bad = bad_edges_keyed(mesh, &keys, &cut);
            assert_eq!(bad, 0, "{name}: eye {eye}, budget {threshold}: {bad} edges open or overlapping in a cut of {} clusters", cut.len());
        }
    }
}

pub(super) fn view(eye: Vec3, threshold: f32) -> LodView {
    LodView { eye, pixels_per_radian: 1080.0 / 0.8, near: 0.1, threshold }
}

#[test]
fn every_cut_is_closed() {
    for seam in [false, true] {
        let mesh = ClusterMesh::build(&rock(5, seam), &ClusterOptions::default());
        let mut cuts = Vec::new();
        for eye in [Vec3::new(0.0, 0.0, 3.0), Vec3::new(2.0, 1.0, 1.5), Vec3::new(0.0, 0.0, 40.0), Vec3::new(-300.0, 20.0, 0.0)] {
            for threshold in [0.0, 0.5, 1.0, 4.0, 1e9] {
                assert_cuts_closed(&format!("seam {seam}"), &mesh, &[eye], &[threshold]);
                let cut = mesh.select(&view(eye, threshold));
                cuts.push(cut.iter().map(|&i| mesh.clusters[i].triangle_count).sum::<u32>());
            }
        }
        // at a pixel's budget, the cut from 300 m is far coarser than the one from 3 m
        assert!(cuts[17] > 0 && cuts[17] * 20 < cuts[2], "seam {seam}: {cuts:?}");
        // and from anywhere, 1.3 m to 1.1 km
        let mut seed = 7;
        assert_cuts_closed(&format!("seam {seam}"), &mesh, &eyes(30, 1.3, 1100.0, &mut seed), &[0.5, 2.0]);
    }
}

#[test]
fn an_eye_inside_the_mesh_gets_the_finest_cut() {
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let cut = mesh.select(&view(Vec3::ZERO, 1.0));
    assert!(cut.iter().all(|&i| mesh.clusters[i].level == 0));
    assert_eq!(bad_edges(&mesh, &cut), 0);
}

#[test]
fn degenerate_triangles_never_break_a_cut() {
    let mut g = rock(4, false);
    // zero-area triangles: repeated vertices, and three vertices on one line
    g.indices.extend([0, 0, 1, 5, 5, 5]);
    let a = g.vertices.len() as u32;
    for k in 0..3 {
        let mut q = g.vertices[0];
        q.position[0] += 0.001 * k as f32;
        g.vertices.push(q);
    }
    g.indices.extend([a, a + 1, a + 2]);
    let mesh = ClusterMesh::build(&g, &ClusterOptions::default());
    for threshold in [0.0, 1.0, 1e9] {
        assert!(!mesh.select(&view(Vec3::new(0.0, 0.0, 30.0), threshold)).is_empty());
    }
}

/// `cargo test -p kansei-core --release --lib build_time -- --ignored --nocapture`
#[test]
#[ignore]
fn build_time() {
    let rock = rock(7, false);
    let start = std::time::Instant::now();
    let mesh = ClusterMesh::build(&rock, &ClusterOptions::default());
    let elapsed = start.elapsed();
    let fill = mesh.clusters.iter().map(|c| c.triangle_count as f32).sum::<f32>() / (mesh.clusters.len() * ClusterOptions::default().max_triangles) as f32;
    println!("{} triangles -> {} clusters ({fill:.2} full) in {elapsed:?}", rock.indices.len() / 3, mesh.clusters.len());
}

/// The rock cut into `islands` uv islands by longitude: each island's triangles get their own
/// vertices (uvs a unit over per island), as a textured asset's are.
fn rock_islands(subdivisions: u32, islands: u32) -> Geometry {
    let rock = rock(subdivisions, false);
    let mut copies: HashMap<(u32, u32), u32> = HashMap::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for t in rock.indices.chunks(3) {
        let centroid: Vec3 = t.iter().map(|&i| Vec3::from_slice(&rock.vertices[i as usize].position[..3])).sum();
        let longitude = centroid.z.atan2(centroid.x) + std::f32::consts::PI;
        let island = ((longitude / std::f32::consts::TAU * islands as f32) as u32).min(islands - 1);
        for &i in t {
            let v = *copies.entry((i, island)).or_insert_with(|| {
                let mut v = rock.vertices[i as usize];
                v.uv[0] += island as f32;
                vertices.push(v);
                vertices.len() as u32 - 1
            });
            indices.push(v);
        }
    }
    Geometry::new("rock islands", vertices, indices)
}

#[test]
fn seams_that_differ_by_rounding_or_the_sign_of_zero_stay_closed() {
    let mut seed = 11;
    let eyes = [vec![Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 40.0, 0.0)], eyes(10, 1.5, 500.0, &mut seed)].concat();
    // the engine's UV sphere: its last column repeats the first, off by rounding (sin(TAU)), and
    // its poles' copies differ in the sign of zero
    let sphere = ClusterMesh::build(&crate::geometries::SphereGeometry::new(1.0, 128, 64), &ClusterOptions::default());
    assert!(sphere.clusters.iter().filter(|c| c.parent_error.is_infinite()).map(|c| c.triangle_count).sum::<u32>() <= 4 * 124);
    assert_cuts_closed("sphere", &sphere, &eyes, &[0.5, 1.0, 1e9]);
    // the seamed rock, its seam copies' zeros written as -0.0
    let mut rock = rock(5, true);
    let copies = rock.vertices.len() / 2;
    for v in rock.vertices.iter_mut().skip(copies) {
        for x in v.position.iter_mut().take(3) {
            if *x == 0.0 {
                *x = -0.0;
            }
        }
    }
    let rock = ClusterMesh::build(&rock, &ClusterOptions::default());
    assert_cuts_closed("rock with -0.0", &rock, &eyes, &[0.5, 1.0, 1e9]);
}

#[test]
fn uv_islands_reduce_like_one_piece() {
    // the grouping finds neighbours across seams: a rock in 16 uv islands is cut from 40 m about
    // as coarsely as in one piece. In 64 islands the seams themselves limit it (the simplifier
    // keeps each seam a polyline): about 1,200 root triangles, against 3,100 grouped by vertex
    let triangles = |mesh: &ClusterMesh, cut: &[usize]| cut.iter().map(|&i| mesh.clusters[i].triangle_count).sum::<u32>();
    let far = view(Vec3::new(0.0, 0.0, 40.0), 1.0);
    let whole = ClusterMesh::build(&rock_islands(5, 1), &ClusterOptions::default());
    let islands = ClusterMesh::build(&rock_islands(5, 16), &ClusterOptions::default());
    let (one, sixteen) = (triangles(&whole, &whole.select(&far)), triangles(&islands, &islands.select(&far)));
    assert!(sixteen <= 2 * one, "from 40 m: {sixteen} triangles in 16 islands, {one} in one piece");
    let many = ClusterMesh::build(&rock_islands(5, 64), &ClusterOptions::default());
    let root: u32 = many.clusters.iter().filter(|c| c.parent_error.is_infinite()).map(|c| c.triangle_count).sum();
    assert!(root <= 1600, "{root} root triangles in 64 islands");
    let mut seed = 3;
    assert_cuts_closed("islands", &islands, &eyes(8, 1.5, 500.0, &mut seed), &[1.0]);
}

#[test]
fn the_cone_test_never_culls_a_cluster_facing_the_eye() {
    let mesh = ClusterMesh::build(&rock(5, false), &ClusterOptions::default());
    let p = |v: u32| Vec3::from_slice(&mesh.vertices[v as usize].position[..3]);
    let mut seed = 5;
    let mut culled = 0;
    for eye in eyes(40, 1.2, 200.0, &mut seed) {
        for (i, c) in mesh.clusters.iter().enumerate() {
            if !c.backfacing(eye) {
                continue;
            }
            culled += 1;
            for t in mesh.triangles(i) {
                let (a, b, d) = (p(t[0]), p(t[1]), p(t[2]));
                let n = (b - a).cross(d - a);
                assert!(n.dot(eye - a) <= 1e-5 * n.length() * (eye - a).length(), "cluster {i} culled from {eye} with a triangle facing it");
            }
        }
    }
    assert!(culled > 0, "nothing culled");
}

#[test]
fn clusters_are_well_filled() {
    // vertex pulling draws each cluster as max_triangles: what's short of it is wasted work
    let options = ClusterOptions::default();
    let mesh = ClusterMesh::build(&rock(5, false), &options);
    let fill = mesh.clusters.iter().map(|c| c.triangle_count as f32).sum::<f32>() / (mesh.clusters.len() * options.max_triangles) as f32;
    assert!(fill >= 0.9, "clusters {fill:.2} full");
}


#[test]
fn levels_partition_the_clusters_in_order() {
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let levels = mesh.levels();
    assert!(levels.len() > 3, "{} levels", levels.len());
    let mut next = 0;
    for (l, level) in levels.iter().enumerate() {
        assert_eq!(level.first, next, "level {l} starts where the one before ends");
        assert!(level.count > 0, "level {l} is empty");
        for c in &mesh.clusters[level.first as usize..(level.first + level.count) as usize] {
            assert_eq!(c.level as usize, l);
            assert!(c.error >= level.min_error);
            assert!(!c.parent_error.is_finite() || c.parent_error <= level.max_parent_error);
            assert!(c.lod_bounds.center.length() - c.lod_bounds.radius <= level.near_reach);
        }
        next += level.count;
    }
    assert_eq!(next as usize, mesh.clusters.len());
    assert!(!levels.last().unwrap().max_parent_error.is_finite(), "the root level has no parent");
}

#[test]
fn the_level_window_keeps_every_drawn_cluster() {
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let levels = mesh.levels();
    let mut seed = 7;
    let mut checked = 0;
    for eye in eyes(40, 0.5, 600.0, &mut seed).into_iter().chain([Vec3::ZERO, Vec3::new(0.0, 0.0, 1.02)]) {
        for threshold in [0.0, 0.25, 1.0, 4.0, 32.0] {
            let v = view(eye, threshold);
            for i in mesh.select(&v) {
                let c = &mesh.clusters[i];
                assert!(levels[c.level as usize].may_draw(&v), "cluster {i} (level {}) drawn from {eye} at {threshold} px, its level skipped", c.level);
                checked += 1;
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}

#[test]
fn the_level_window_skips_the_levels_a_view_cannot_reach() {
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let levels = mesh.levels();
    let skipped = |v: &LodView| levels.iter().map(|l| !l.may_draw(v)).collect::<Vec<_>>();
    // far away, the fine levels' parents are all under the budget
    let far = view(Vec3::new(0.0, 0.0, 400.0), 1.0);
    assert!(skipped(&far)[0] && skipped(&far)[1], "far: {:?}", skipped(&far));
    // at the surface with a tight budget, the coarsest level is over it
    let near = view(Vec3::new(0.0, 0.0, 1.02), 0.25);
    assert!(!skipped(&near)[0] && *skipped(&near).last().unwrap(), "near: {:?}", skipped(&near));
}
