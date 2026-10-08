//! optimesh's outputs on a test rock, called as `ClusterMesh::build` calls it, for the TS port's
//! differential tests (`tests/clusters/meshopt.test.ts`): the rock's positions, attributes and
//! indices, its meshlets, their bounds, their partition into groups of 4, and the first group
//! simplified to half with the vertices other groups share locked. Floats are written as their
//! bits, so the test compares them exactly.
//!
//! Usage: `cargo run --release -- <subdivisions> <out.json> [seam]` (`seam`: the x < 0 side's
//! triangles get their own vertices, a unit over in u). The committed fixtures are
//! `3 tests/fixtures/meshopt-rock3.json` and `3 tests/fixtures/meshopt-rock3-seam.json seam`.
//!
//! `graph <mesh> <out.json>` writes a whole graph instead, for `tests/clusters/build.test.ts`: the
//! mesh's vertices (as bits) and indices, and `ClusterMesh::build`'s `gpu_words()` with the
//! default options (`cards` on for meshes with cards). `<mesh>`: `rock<n>` (an n-times subdivided
//! rock), `rock<n>-seam` (with a uv seam), `crown<n>` (a tree crown of n cards) or `mixed` (cards,
//! a rock, an open tube and a flat grid, as `card_tests::small_open_flat_components_are_cards`).
use kansei_core::clusters::{ClusterMesh, ClusterOptions};
use kansei_core::geometries::{Geometry, IcosphereGeometry, Vertex};
use glam::Vec3;
use optimesh::clusterizer::{build_meshlets, build_meshlets_bound, Meshlet, MeshletBuffers, Positions};
use optimesh::meshletutils::compute_cluster_bounds;
use optimesh::partition::partition_clusters;
use optimesh::simplifier::{simplify_with_attributes, Attributes, SimplifyTarget, VertexData, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A JSON array of numbers.
fn list<T: std::fmt::Display>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

/// The test rock: an icosphere of `sub` subdivisions, bumped; with `seam`, the x < 0 side's
/// triangles on their own vertices a unit over in u.
fn rock(sub: u32, seam: bool) -> Geometry {
    let mut g = IcosphereGeometry::new(1.0, sub);
    for v in &mut g.vertices {
        let n = Vec3::from(v.normal);
        let h = 1.0 + 0.12 * (5.0 * n.x).sin() * (4.0 * n.y).cos() + 0.06 * (13.0 * n.z).sin();
        v.position = [n.x * h, n.y * h * 0.7, n.z * h, 1.0];
    }
    if seam {
        let n = g.vertices.len() as u32;
        let copies: Vec<_> = g.vertices.iter().map(|v| {
            let mut c = *v;
            c.uv[0] += 1.0;
            c
        }).collect();
        g.vertices.extend(copies);
        for t in g.indices.chunks_mut(3) {
            let x: f32 = t.iter().map(|&i| g.vertices[i as usize].position[0]).sum();
            if x < 0.0 {
                for i in t.iter_mut() {
                    *i += n;
                }
            }
        }
    }
    g
}

/// A quad (two triangles, its own four vertices) centred at `c`, in the plane of `u` and `v`
/// (half extents), normal u x v (`card_tests::quad`).
fn quad(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, c: Vec3, u: Vec3, v: Vec3) {
    let base = vertices.len() as u32;
    let n = u.cross(v).normalize();
    for (i, (su, sv)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
        let p = c + u * su + v * sv;
        vertices.push(Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: [(i == 1 || i == 2) as u32 as f32, (i >= 2) as u32 as f32] });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// A tree's crown of `count` cards (`card_tests::crown`).
fn crown(count: u32) -> Geometry {
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

/// Cards, a closed rock, an open tube (its normals cancel) and a flat grid too large to be a card.
fn mixed() -> Geometry {
    let mut g = crown(50);
    let r = rock(2, false);
    let base = g.vertices.len() as u32;
    g.vertices.extend(r.vertices.iter().cloned());
    g.indices.extend(r.indices.iter().map(|i| i + base));
    let base = g.vertices.len() as u32;
    for k in 0..=12u32 {
        let a = k as f32 / 12.0 * std::f32::consts::TAU;
        for y in [0.0, 2.0] {
            g.vertices.push(Vertex { position: [20.0 + a.cos(), y, a.sin(), 1.0], normal: [a.cos(), 0.0, a.sin()], uv: [0.0; 2] });
        }
    }
    for k in 0..12u32 {
        let (a, b) = (base + 2 * k, base + 2 * k + 2);
        g.indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
    }
    let base = g.vertices.len() as u32;
    for j in 0..=10u32 {
        for i in 0..=10u32 {
            g.vertices.push(Vertex { position: [-20.0 + i as f32, 0.0, j as f32, 1.0], normal: [0.0, 1.0, 0.0], uv: [0.0; 2] });
        }
    }
    for j in 0..10u32 {
        for i in 0..10u32 {
            let v = base + j * 11 + i;
            g.indices.extend_from_slice(&[v, v + 11, v + 1, v + 1, v + 11, v + 12]);
        }
    }
    g
}

/// `graph <mesh> <out.json>`: the mesh and its graph's words.
fn graph(name: &str, out: &str) {
    let (geometry, cards) = if let Some(n) = name.strip_prefix("crown") {
        (crown(n.parse().unwrap()), true)
    } else if name == "mixed" {
        (mixed(), true)
    } else {
        let rest = name.strip_prefix("rock").expect("rock<n>, rock<n>-seam, crown<n> or mixed");
        let (n, seam) = rest.strip_suffix("-seam").map_or((rest, false), |n| (n, true));
        (rock(n.parse().unwrap(), seam), false)
    };
    let options = ClusterOptions { cards, ..Default::default() };
    let t = std::time::Instant::now();
    let mesh = ClusterMesh::build(&geometry, &options);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let vertices: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2], v.position[3], v.normal[0], v.normal[1], v.normal[2], v.uv[0], v.uv[1]]).collect();
    let words = mesh.gpu_words();
    let json = format!("{{\"vertices\":{},\"indices\":{},\"cards\":{cards},\"words\":{}}}", list(&bits(&vertices)), list(&geometry.indices), list(&words));
    std::fs::write(out, json).unwrap();
    println!("{name}: {} triangles -> {} clusters over {} levels in {ms:.1} ms; {} words -> {out}", geometry.indices.len() / 3, mesh.clusters.len(), mesh.levels().len(), words.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("graph") {
        graph(&args[2], &args[3]);
        return;
    }
    let sub: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let out = args.get(2).cloned().unwrap_or_else(|| format!("meshopt{sub}.json"));
    let g = rock(sub, args.get(3).is_some_and(|s| s == "seam"));
    let positions: Vec<f32> = g.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
    let attributes: Vec<f32> = g.vertices.iter().flat_map(|v| [v.normal[0], v.normal[1], v.normal[2], v.uv[0], v.uv[1]]).collect();
    let indices = g.indices.clone();
    let vc = positions.len() / 3;

    let bound = build_meshlets_bound(indices.len(), 128, 124);
    let mut meshlets = vec![Meshlet::default(); bound];
    let mut mv = vec![0u32; bound * 128];
    let mut mt = vec![0u8; bound * 124 * 3];
    let count = build_meshlets(&mut MeshletBuffers { meshlets: &mut meshlets, vertices: &mut mv, triangles: &mut mt }, &indices, &Positions { data: &positions, count: vc, stride: 12 }, 128, 124, 0.25);
    let meshlets = &meshlets[..count];
    let globals: Vec<Vec<u32>> = meshlets.iter().map(|m| (0..m.triangle_count as usize * 3).map(|k| mv[m.vertex_offset as usize + mt[m.triangle_offset as usize + k] as usize]).collect()).collect();
    let bounds: Vec<Vec<u32>> = globals.iter().map(|gl| {
        let b = compute_cluster_bounds(gl, &positions, vc, 12);
        bits(&[b.center[0], b.center[1], b.center[2], b.radius, b.cone_apex[0], b.cone_apex[1], b.cone_apex[2], b.cone_axis[0], b.cone_axis[1], b.cone_axis[2], b.cone_cutoff])
    }).collect();

    let flat: Vec<u32> = globals.iter().flatten().copied().collect();
    let counts: Vec<u32> = globals.iter().map(|g| g.len() as u32).collect();
    let mut dest = vec![0u32; count];
    let groups = partition_clusters(&mut dest, &flat, &counts, Some(&positions), vc, 12, 4);

    // simplify group 0, locking the vertices other groups use too
    let mut owner = vec![u32::MAX; vc];
    for (i, gl) in globals.iter().enumerate() { for &v in gl { owner[v as usize] = if owner[v as usize] == u32::MAX || owner[v as usize] == dest[i] { dest[i] } else { u32::MAX - 1 }; } }
    let lock: Vec<u8> = owner.iter().map(|&o| if o == u32::MAX - 1 { SIMPLIFY_VERTEX_LOCK } else { 0 }).collect();
    let merged: Vec<u32> = globals.iter().enumerate().filter(|(i, _)| dest[*i] == 0).flat_map(|(_, g)| g.iter().copied()).collect();
    let target = ((merged.len() as f32 * 0.5) as usize / 3) * 3;
    let weights = [0.5f32, 0.5, 0.5, 0.1, 0.1];
    let mut simplified = vec![0u32; merged.len()];
    let (scount, serror) = simplify_with_attributes(&mut simplified, &merged, &VertexData { positions: &positions, count: vc, stride: 12 }, &Attributes { data: &attributes, stride: 20, weights: &weights, count: 5 }, Some(&lock), &SimplifyTarget { target_index_count: target, target_error: f32::MAX, options: SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE });

    let last = meshlets.last().copied().unwrap_or_default();
    let fields = [
        ("positions", list(&bits(&positions))),
        ("attributes", list(&bits(&attributes))),
        ("indices", list(&indices)),
        ("meshlets", format!("[{}]", meshlets.iter().map(|m| list(&[m.vertex_offset, m.triangle_offset, m.vertex_count, m.triangle_count])).collect::<Vec<_>>().join(","))),
        ("meshletVertices", list(&mv[..(last.vertex_offset + last.vertex_count) as usize])),
        ("meshletTriangles", list(&mt[..(last.triangle_offset + last.triangle_count * 3) as usize])),
        ("bounds", format!("[{}]", bounds.iter().map(|b| list(b)).collect::<Vec<_>>().join(","))),
        ("partitions", list(&dest)),
        ("partitionCount", groups.to_string()),
        ("lock", list(&lock)),
        ("merged", list(&merged)),
        ("target", target.to_string()),
        ("simplified", list(&simplified[..scount])),
        ("simplifyError", serror.to_bits().to_string()),
    ];
    let json = format!("{{{}}}", fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(","));
    std::fs::write(&out, json).unwrap();
    println!("{} meshlets, {} partitions, simplified {} -> {} indices; {out}", count, groups, merged.len(), scount);
}
