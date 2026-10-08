//! optimesh's outputs on a test rock, called as `ClusterMesh::build` calls it, for the TS port's
//! differential tests (`tests/clusters/meshopt.test.ts`): the rock's positions, attributes and
//! indices, its meshlets, their bounds, their partition into groups of 4, and the first group
//! simplified to half with the vertices other groups share locked. Floats are written as their
//! bits, so the test compares them exactly.
//!
//! Usage: `cargo run --release -- <subdivisions> <out.json> [seam]` (`seam`: the x < 0 side's
//! triangles get their own vertices, a unit over in u). The committed fixtures are
//! `3 tests/fixtures/meshopt-rock3.json` and `3 tests/fixtures/meshopt-rock3-seam.json seam`.
use kansei_core::geometries::IcosphereGeometry;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sub: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let out = args.get(2).cloned().unwrap_or_else(|| format!("meshopt{sub}.json"));
    let mut g = IcosphereGeometry::new(1.0, sub);
    // a bumpy rock, as the example's, in f32
    for v in &mut g.vertices {
        let n = glam::Vec3::from(v.normal);
        let h = 1.0 + 0.12 * (5.0 * n.x).sin() * (4.0 * n.y).cos() + 0.06 * (13.0 * n.z).sin();
        v.position = [n.x * h, n.y * h * 0.7, n.z * h, 1.0];
    }
    // seam=1: the x < 0 side's triangles get their own vertices, uvs a unit over (a uv seam)
    if args.get(3).map_or(false, |s| s == "seam") {
        let n = g.vertices.len() as u32;
        let copies: Vec<_> = g.vertices.iter().map(|v| { let mut c = *v; c.uv[0] += 1.0; c }).collect();
        g.vertices.extend(copies);
        for t in g.indices.chunks_mut(3) {
            let x: f32 = t.iter().map(|&i| g.vertices[i as usize].position[0]).sum();
            if x < 0.0 { for i in t.iter_mut() { *i += n; } }
        }
    }
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
