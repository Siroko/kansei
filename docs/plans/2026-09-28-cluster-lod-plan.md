# Cluster LOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Nanite-style cluster LOD graph for any Kansei `Geometry` on the CPU, in native and in wasm, with the per-cluster cut rule the GPU will later apply. That is milestone M1 of `2026-09-28-cluster-lod-design.md`; M2-M6 are laid out at the end and get their own task plans when they start.

**Architecture:**
- A new `kansei_core::clusters` module. `ClusterMesh::build` splits a mesh into clusters, groups neighbours, simplifies each group with its shared vertices locked, and re-splits, level after level, with `optimesh` (the pure-Rust meshoptimizer).
- Each cluster keeps its error and LOD sphere, and its parent group's. `ClusterMesh::select` draws a cluster iff its projected error is within the pixel budget and its parent's is over it.
- The tests check the property everything later relies on: every cut is free of holes and overlaps.

**Tech Stack:** Rust 2021, `optimesh` 1.1 (pure Rust, no `unsafe` by default, no dependencies), `glam` 0.29, `cargo test`, `wasm32-unknown-unknown`.

**Spec:** `docs/plans/2026-09-28-cluster-lod-design.md`

## Global Constraints

- Everything runs in wasm (`cargo check -p kansei-core --target wasm32-unknown-unknown`): no threads, no filesystem, no C dependencies.
- Kansei's one vertex layout: `Vertex { position: [f32; 4], normal: [f32; 3], uv: [f32; 2] }` (`geometries/geometry.rs`); indices are `u32`.
- Cluster limits: at most 124 triangles and 64 vertices (`ClusterOptions::default()`).
- The graph shares the geometry's vertex buffer: simplification only drops and reuses vertices.
- Errors are absolute (object-space metres), and a cluster's parent error and parent sphere contain its own.
- PRs go against `development`. Commit messages follow the repo's style (`feat(clusters): ...`) with no AI attribution.

## Review Focus

- **Flat-shaded meshes** (every edge an attribute seam): the build succeeds with one level and every cut is the whole mesh. Test `flat_shaded_meshes_keep_one_level`, Task 3.
- **Empty and one-triangle geometry**: no panic; 0 and 1 clusters. Test `tiny_and_empty_meshes`, Task 2.
- **Degenerate triangles** (repeated vertex, collinear) in the input: no panic, and cuts are non-empty. Test `degenerate_triangles_never_break_a_cut`, Task 4.
- **An eye inside a cluster's LOD sphere**: the distance clamps to `near`, giving the finest cut, still closed. Test `an_eye_inside_the_mesh_gets_the_finest_cut`, Task 4.
- **A fold the simplifier leaves inside one coarse cluster** (an edge used 4 times, all in that cluster) is not a crack. The closure check in Task 4 counts holes (odd edge use) and overlaps (edges shared by other than two clusters once each), not folds.

---

## File Structure

| file | responsibility |
|---|---|
| `rust/Cargo.toml` | workspace dependency `optimesh = "1.1"` |
| `rust/kansei-core/Cargo.toml` | `optimesh = { workspace = true }` |
| `rust/kansei-core/src/lib.rs` | `pub mod clusters;` |
| `rust/kansei-core/src/clusters/mod.rs` | public types: `Sphere`, `ClusterOptions`, `Cluster`, `ClusterMesh`, `LodView`, `projected_error`; `ClusterMesh::triangles` and `ClusterMesh::select` |
| `rust/kansei-core/src/clusters/build.rs` | `ClusterMesh::build` and its helpers (split, partition, locks) |
| `rust/kansei-core/src/clusters/tests.rs` | test meshes (the rock), the closure check, all M1 tests |

---

### Task 1: The `clusters` module, its dependency, and `Sphere`

**Files:**
- Modify: `rust/Cargo.toml` (`[workspace.dependencies]`)
- Modify: `rust/kansei-core/Cargo.toml` (`[dependencies]`)
- Modify: `rust/kansei-core/src/lib.rs`
- Create: `rust/kansei-core/src/clusters/mod.rs`
- Create: `rust/kansei-core/src/clusters/tests.rs`

**Interfaces:**
- Produces: `clusters::Sphere { center: glam::Vec3, radius: f32 }`, with `Sphere::enclosing(impl IntoIterator<Item = Sphere>) -> Sphere` and `Sphere::contains(&self, &Sphere) -> bool`.

- [ ] **Step 1: Add the dependency and the module**

`rust/Cargo.toml`, under `[workspace.dependencies]`:
```toml
optimesh = "1.1"
```
`rust/kansei-core/Cargo.toml`, under `[dependencies]`:
```toml
optimesh = { workspace = true }
```
`rust/kansei-core/src/lib.rs`, after `pub mod impostors;`:
```rust
pub mod clusters;
```

- [ ] **Step 2: Write the failing test**

`rust/kansei-core/src/clusters/tests.rs`:
```rust
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
```

- [ ] **Step 3: Run it to see it fail**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: FAIL to compile, `Sphere` not found.

- [ ] **Step 4: Write `mod.rs` with `Sphere`**

`rust/kansei-core/src/clusters/mod.rs`:
```rust
//! Cluster LOD (meshlets, as Unreal's Nanite): a mesh split into clusters of about 124
//! triangles, with a graph of coarser versions over them, so a view can draw each part of the
//! mesh at the coarsest version whose error it doesn't see. See
//! `docs/plans/2026-09-28-cluster-lod-design.md`.

use glam::Vec3;

/// A sphere; clusters' bounds and the spheres their errors are measured from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    /// A sphere enclosing all of `spheres` (grown from the first, not the smallest).
    pub fn enclosing(spheres: impl IntoIterator<Item = Sphere>) -> Sphere {
        let mut spheres = spheres.into_iter();
        let mut s = spheres.next().expect("at least one sphere");
        for o in spheres {
            let d = o.center.distance(s.center);
            if d + o.radius <= s.radius {
                continue;
            }
            if d + s.radius <= o.radius {
                s = o;
                continue;
            }
            let radius = (d + s.radius + o.radius) * 0.5;
            let center = s.center + (o.center - s.center) * ((radius - s.radius) / d.max(1e-12));
            s = Sphere { center, radius };
        }
        s
    }

    /// Whether `other` lies inside (to float precision).
    pub fn contains(&self, other: &Sphere) -> bool {
        self.center.distance(other.center) + other.radius <= self.radius * (1.0 + 1e-5) + 1e-6
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 5: Run the test to see it pass**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/kansei-core/Cargo.toml rust/kansei-core/src/lib.rs rust/kansei-core/src/clusters
git commit -m "feat(clusters): the module, optimesh, and enclosing spheres"
```

---

### Task 2: Level 0, the mesh in clusters

**Files:**
- Modify: `rust/kansei-core/src/clusters/mod.rs`
- Create: `rust/kansei-core/src/clusters/build.rs`
- Modify: `rust/kansei-core/src/clusters/tests.rs`

**Interfaces:**
- Consumes: `Sphere` (Task 1).
- Produces:
  - `ClusterOptions { max_vertices, max_triangles, group_size, simplify_ratio, stall_ratio, cone_weight, normal_weight, uv_weight }` with `Default`;
  - `Cluster { vertex_offset, vertex_count, triangle_offset, triangle_count: u32, bounds: Sphere, cone_axis: Vec3, cone_cutoff: f32, error: f32, lod_bounds: Sphere, parent_error: f32, parent_bounds: Sphere, level: u32 }`;
  - `ClusterMesh { vertices: Vec<Vertex>, clusters: Vec<Cluster>, cluster_vertices: Vec<u32>, cluster_triangles: Vec<u8> }`;
  - `ClusterMesh::build(&Geometry, &ClusterOptions) -> ClusterMesh` (level 0 only in this task);
  - `ClusterMesh::triangles(&self, cluster: usize) -> impl Iterator<Item = [u32; 3]>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests.rs`:
```rust
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: FAIL to compile, `ClusterMesh` not found.

- [ ] **Step 3: Add the types to `mod.rs`**

Append to `mod.rs` (before `#[cfg(test)] mod tests;`), and add `mod build;` next to it:
```rust
use crate::geometries::Vertex;

mod build;

/// How `ClusterMesh::build` splits and simplifies.
#[derive(Clone, Copy, Debug)]
pub struct ClusterOptions {
    pub max_vertices: usize,
    pub max_triangles: usize,
    /// Clusters per group simplified together (about).
    pub group_size: usize,
    /// Triangles a group keeps when simplified.
    pub simplify_ratio: f32,
    /// A group keeping more than this share of its triangles has stalled: it isn't simplified.
    pub stall_ratio: f32,
    /// Weight of normal-cone tightness against compactness when splitting (backface culling).
    pub cone_weight: f32,
    /// Weights of the normals and uvs in the simplifier's error.
    pub normal_weight: f32,
    pub uv_weight: f32,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self { max_vertices: 64, max_triangles: 124, group_size: 8, simplify_ratio: 0.5, stall_ratio: 0.85, cone_weight: 0.25, normal_weight: 0.5, uv_weight: 0.1 }
    }
}

/// One cluster: a run of `cluster_vertices` and `cluster_triangles`, its bounds, and the errors
/// the cut rule compares (see `ClusterMesh::select`).
#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    /// In triangles (3 local indices each).
    pub triangle_offset: u32,
    pub triangle_count: u32,
    /// Of its triangles, for culling.
    pub bounds: Sphere,
    /// Its triangles' normal cone (backface culling): axis, and cos of the half-angle (1 = none).
    pub cone_axis: Vec3,
    pub cone_cutoff: f32,
    /// The error of the simplification that made it (0 at level 0), measured from `lod_bounds`.
    pub error: f32,
    pub lod_bounds: Sphere,
    /// The error of the group simplified from it and its siblings (∞ when none was), from
    /// `parent_bounds`.
    pub parent_error: f32,
    pub parent_bounds: Sphere,
    pub level: u32,
}

/// A mesh as a graph of clusters over its own vertices.
pub struct ClusterMesh {
    /// The geometry's vertices, shared by every level.
    pub vertices: Vec<Vertex>,
    pub clusters: Vec<Cluster>,
    /// Per cluster, its vertices as indices into `vertices`.
    pub cluster_vertices: Vec<u32>,
    /// Per cluster, 3 indices into its own vertices per triangle.
    pub cluster_triangles: Vec<u8>,
}

impl ClusterMesh {
    /// Cluster `cluster`'s triangles, as indices into `vertices`.
    pub fn triangles(&self, cluster: usize) -> impl Iterator<Item = [u32; 3]> + '_ {
        let c = &self.clusters[cluster];
        let vertices = &self.cluster_vertices[c.vertex_offset as usize..(c.vertex_offset + c.vertex_count) as usize];
        self.cluster_triangles[c.triangle_offset as usize * 3..(c.triangle_offset + c.triangle_count) as usize * 3]
            .chunks(3)
            .map(move |t| [vertices[t[0] as usize], vertices[t[1] as usize], vertices[t[2] as usize]])
    }
}
```

- [ ] **Step 4: Write `build.rs` with level 0**

`rust/kansei-core/src/clusters/build.rs`:
```rust
use glam::Vec3;
use optimesh::clusterizer::{build_meshlets, build_meshlets_bound, Meshlet, MeshletBuffers, Positions};
use optimesh::meshletutils::compute_cluster_bounds;

use super::{Cluster, ClusterMesh, ClusterOptions, Sphere};
use crate::geometries::Geometry;

impl ClusterMesh {
    /// Split `geometry` into clusters and build the graph of coarser versions over them.
    pub fn build(geometry: &Geometry, options: &ClusterOptions) -> ClusterMesh {
        let positions: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
        let mut mesh = ClusterMesh { vertices: geometry.vertices.clone(), clusters: Vec::new(), cluster_vertices: Vec::new(), cluster_triangles: Vec::new() };
        mesh.split(&geometry.indices, &positions, 0.0, None, 0, options);
        mesh
    }

    /// Clusters of `indices` (into `vertices`), carrying `error` and `lod_bounds` (their own
    /// bounds when `None`); per new cluster, its index and its triangles as indices into
    /// `vertices`.
    pub(super) fn split(&mut self, indices: &[u32], positions: &[f32], error: f32, lod_bounds: Option<Sphere>, level: u32, options: &ClusterOptions) -> Vec<(usize, Vec<u32>)> {
        if indices.is_empty() {
            return Vec::new();
        }
        let bound = build_meshlets_bound(indices.len(), options.max_vertices, options.max_triangles);
        let mut meshlets = vec![Meshlet::default(); bound];
        let mut vertices = vec![0u32; bound * options.max_vertices];
        let mut triangles = vec![0u8; bound * options.max_triangles * 3];
        let count = build_meshlets(
            &mut MeshletBuffers { meshlets: &mut meshlets, vertices: &mut vertices, triangles: &mut triangles },
            indices,
            &Positions { data: positions, count: positions.len() / 3, stride: 12 },
            options.max_vertices,
            options.max_triangles,
            options.cone_weight,
        );
        let mut out = Vec::with_capacity(count);
        for m in &meshlets[..count] {
            let local_vertices = &vertices[m.vertex_offset as usize..(m.vertex_offset + m.vertex_count) as usize];
            // (meshlet triangle offsets count indices, 3 per triangle)
            let local_triangles = &triangles[m.triangle_offset as usize..(m.triangle_offset + m.triangle_count * 3) as usize];
            let global: Vec<u32> = local_triangles.iter().map(|&t| local_vertices[t as usize]).collect();
            let b = compute_cluster_bounds(&global, positions, positions.len() / 3, 12);
            let bounds = Sphere { center: Vec3::from(b.center), radius: b.radius };
            let lod_bounds = lod_bounds.unwrap_or(bounds);
            out.push((self.clusters.len(), global));
            self.clusters.push(Cluster {
                vertex_offset: self.cluster_vertices.len() as u32,
                vertex_count: m.vertex_count,
                triangle_offset: (self.cluster_triangles.len() / 3) as u32,
                triangle_count: m.triangle_count,
                bounds,
                cone_axis: Vec3::from(b.cone_axis),
                cone_cutoff: b.cone_cutoff,
                error,
                lod_bounds,
                parent_error: f32::INFINITY,
                parent_bounds: lod_bounds,
                level,
            });
            self.cluster_vertices.extend_from_slice(local_vertices);
            self.cluster_triangles.extend_from_slice(local_triangles);
        }
        out
    }
}
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add rust/kansei-core/src/clusters
git commit -m "feat(clusters): a mesh split into clusters with their bounds and normal cones"
```

---

### Task 3: The graph, grouped, simplified and re-split level after level

**Files:**
- Modify: `rust/kansei-core/src/clusters/build.rs`
- Modify: `rust/kansei-core/src/clusters/tests.rs`

**Interfaces:**
- Consumes: `ClusterMesh::split` (Task 2).
- Produces: `ClusterMesh::build` now builds every level. Each child records its group's error and sphere as `parent_error` and `parent_bounds`, and the group's new clusters carry them as `error` and `lod_bounds`.

- [ ] **Step 1: Write the failing tests**

Append to `tests.rs`:
```rust
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: `levels_shrink_to_a_root_and_errors_and_bounds_nest` and `open_and_disjoint_meshes_terminate` FAIL (only level 0 exists). `flat_shaded_meshes_keep_one_level` passes already.

- [ ] **Step 3: Build the graph**

In `build.rs`, extend the imports:
```rust
use optimesh::partition::partition_clusters;
use optimesh::simplifier::{simplify_with_attributes, Attributes, SimplifyTarget, VertexData, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK};
use std::collections::HashMap;
```
Replace `build`'s body:
```rust
    pub fn build(geometry: &Geometry, options: &ClusterOptions) -> ClusterMesh {
        let positions: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
        let attributes: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.normal[0], v.normal[1], v.normal[2], v.uv[0], v.uv[1]]).collect();
        let weights = [options.normal_weight, options.normal_weight, options.normal_weight, options.uv_weight, options.uv_weight];
        let position_ids = position_ids(&positions);
        let mut mesh = ClusterMesh { vertices: geometry.vertices.clone(), clusters: Vec::new(), cluster_vertices: Vec::new(), cluster_triangles: Vec::new() };
        // the clusters still without a parent, with their triangles
        let mut pending = mesh.split(&geometry.indices, &positions, 0.0, None, 0, options);
        let mut level = 0;
        while pending.len() > 1 {
            level += 1;
            let groups = partition(&pending, &positions, options.group_size);
            let lock = shared_vertex_locks(&groups, &pending, &position_ids);
            let mut next = Vec::new();
            let mut progress = false;
            for group in &groups {
                let merged: Vec<u32> = group.iter().flat_map(|&i| pending[i].1.iter().copied()).collect();
                let target = ((merged.len() as f32 * options.simplify_ratio) as usize / 3) * 3;
                let mut simplified = vec![0u32; merged.len()];
                let (count, error) = simplify_with_attributes(
                    &mut simplified,
                    &merged,
                    &VertexData { positions: &positions, count: positions.len() / 3, stride: 12 },
                    &Attributes { data: &attributes, stride: 20, weights: &weights, count: 5 },
                    Some(&lock),
                    &SimplifyTarget { target_index_count: target, target_error: f32::MAX, options: SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE },
                );
                if count as f32 > merged.len() as f32 * options.stall_ratio {
                    // too little came off: next round, grouped with other neighbours
                    next.extend(group.iter().map(|&i| pending[i].clone()));
                    continue;
                }
                progress = true;
                // never less than a child's error, from a sphere round all of theirs
                let error = group.iter().map(|&i| mesh.clusters[pending[i].0].error).fold(error, f32::max);
                let bounds = Sphere::enclosing(group.iter().map(|&i| mesh.clusters[pending[i].0].lod_bounds));
                for &i in group {
                    let child = &mut mesh.clusters[pending[i].0];
                    child.parent_error = error;
                    child.parent_bounds = bounds;
                }
                next.extend(mesh.split(&simplified[..count], &positions, error, Some(bounds), level, options));
            }
            if !progress {
                break;
            }
            pending = next;
        }
        mesh
    }
```
Append the helpers to `build.rs`:
```rust
/// One id per distinct position (seams split vertices, not positions).
fn position_ids(positions: &[f32]) -> Vec<u32> {
    let mut ids = HashMap::new();
    positions
        .chunks(3)
        .map(|p| {
            let next = ids.len() as u32;
            *ids.entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]).or_insert(next)
        })
        .collect()
}

/// Groups of about `size` neighbouring clusters (indices into `pending`).
fn partition(pending: &[(usize, Vec<u32>)], positions: &[f32], size: usize) -> Vec<Vec<usize>> {
    let indices: Vec<u32> = pending.iter().flat_map(|(_, t)| t.iter().copied()).collect();
    let counts: Vec<u32> = pending.iter().map(|(_, t)| t.len() as u32).collect();
    let mut destination = vec![0u32; pending.len()];
    let groups = partition_clusters(&mut destination, &indices, &counts, Some(positions), positions.len() / 3, 12, size);
    let mut out = vec![Vec::new(); groups];
    for (i, &g) in destination.iter().enumerate() {
        out[g as usize].push(i);
    }
    out
}

/// `SIMPLIFY_VERTEX_LOCK` on every vertex whose position more than one group uses: groups meet
/// at the same vertices whatever level each is drawn at.
fn shared_vertex_locks(groups: &[Vec<usize>], pending: &[(usize, Vec<u32>)], position_ids: &[u32]) -> Vec<u8> {
    const NONE: u32 = u32::MAX;
    const SHARED: u32 = u32::MAX - 1;
    let mut owner = vec![NONE; position_ids.len()];
    for (g, group) in groups.iter().enumerate() {
        for &i in group {
            for &v in &pending[i].1 {
                let p = position_ids[v as usize] as usize;
                owner[p] = match owner[p] {
                    NONE => g as u32,
                    o if o == g as u32 => o,
                    _ => SHARED,
                };
            }
        }
    }
    position_ids.iter().map(|&p| if owner[p as usize] == SHARED { SIMPLIFY_VERTEX_LOCK } else { 0 }).collect()
}
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/clusters
git commit -m "feat(clusters): the graph: groups simplified with their shared vertices locked, level after level"
```

---

### Task 4: The cut rule

**Files:**
- Modify: `rust/kansei-core/src/clusters/mod.rs`
- Modify: `rust/kansei-core/src/clusters/tests.rs`

**Interfaces:**
- Consumes: `ClusterMesh`, `Cluster` (Tasks 2-3).
- Produces:
  - `LodView { eye: Vec3, pixels_per_radian: f32, near: f32, threshold: f32 }` (all in the mesh's own space);
  - `projected_error(error: f32, sphere: Sphere, view: &LodView) -> f32`;
  - `ClusterMesh::select(&self, view: &LodView) -> Vec<usize>`.

  M2's GPU selection reproduces exactly these.

- [ ] **Step 1: Write the failing tests**

Append to `tests.rs`:
```rust
fn key(v: &Vertex) -> [u32; 3] {
    [v.position[0].to_bits(), v.position[1].to_bits(), v.position[2].to_bits()]
}

/// Edges (by position) of `clusters`' triangles that betray a hole or an overlap: used an odd
/// number of times (a hole's rim), or shared by other than exactly two clusters once each (an
/// overlap). A fold the simplifier left inside one cluster (an edge used 4 times, all in that
/// cluster) is neither.
fn bad_edges(mesh: &ClusterMesh, clusters: &[usize]) -> usize {
    let mut uses: HashMap<([u32; 3], [u32; 3]), Vec<usize>> = HashMap::new();
    for &c in clusters {
        for t in mesh.triangles(c) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (a, b) = (key(&mesh.vertices[a as usize]), key(&mesh.vertices[b as usize]));
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

fn view(eye: Vec3, threshold: f32) -> LodView {
    LodView { eye, pixels_per_radian: 1080.0 / 0.8, near: 0.1, threshold }
}

#[test]
fn every_cut_is_closed() {
    for seam in [false, true] {
        let mesh = ClusterMesh::build(&rock(5, seam), &ClusterOptions::default());
        let mut cuts = Vec::new();
        for eye in [Vec3::new(0.0, 0.0, 3.0), Vec3::new(2.0, 1.0, 1.5), Vec3::new(0.0, 0.0, 40.0), Vec3::new(-300.0, 20.0, 0.0)] {
            for threshold in [0.0, 0.5, 1.0, 4.0, 1e9] {
                let cut = mesh.select(&view(eye, threshold));
                let bad = bad_edges(&mesh, &cut);
                assert_eq!(bad, 0, "seam {seam}, eye {eye}, budget {threshold}: {bad} edges open or overlapping in a cut of {} clusters", cut.len());
                cuts.push(cut.iter().map(|&i| mesh.clusters[i].triangle_count).sum::<u32>());
            }
        }
        // at a pixel's budget, the cut from 300 m is far coarser than the one from 3 m
        assert!(cuts[17] * 20 < cuts[2], "seam {seam}: {cuts:?}");
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: FAIL to compile, `LodView` / `select` not found.

- [ ] **Step 3: Write the cut rule**

Append to `mod.rs` (before the test module):
```rust
/// Where a view sees a mesh from, in the mesh's own space, and how much error it tolerates.
#[derive(Clone, Copy, Debug)]
pub struct LodView {
    pub eye: Vec3,
    /// Pixels per radian at the view's centre: the viewport's height / (2 tan(fov_y / 2)).
    pub pixels_per_radian: f32,
    /// Distances are clamped to this (an eye inside a sphere).
    pub near: f32,
    /// The error budget, pixels.
    pub threshold: f32,
}

/// `error` (metres) seen from the view as pixels: over the distance to the nearest point of
/// `sphere`. A parent's sphere contains its children's and its error is at least theirs, so its
/// projected error is at least theirs from any eye.
pub fn projected_error(error: f32, sphere: Sphere, view: &LodView) -> f32 {
    if error == 0.0 {
        return 0.0;
    }
    if !error.is_finite() {
        return f32::INFINITY;
    }
    let distance = (sphere.center.distance(view.eye) - sphere.radius).max(view.near);
    error / distance * view.pixels_per_radian
}

impl ClusterMesh {
    /// The clusters `view` draws: each whose error is within the budget and whose parent's is
    /// over it. Every point of the mesh is in exactly one (no holes, no overlaps).
    pub fn select(&self, view: &LodView) -> Vec<usize> {
        (0..self.clusters.len())
            .filter(|&i| {
                let c = &self.clusters[i];
                projected_error(c.error, c.lod_bounds, view) <= view.threshold && projected_error(c.parent_error, c.parent_bounds, view) > view.threshold
            })
            .collect()
    }
}
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `cd rust && cargo test -p kansei-core --lib clusters`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/clusters
git commit -m "feat(clusters): the cut rule, and every cut closed at any budget and viewpoint"
```

---

### Task 5: Wasm, build time, and the PR

**Files:**
- Modify: `rust/kansei-core/src/clusters/tests.rs`
- Modify: `docs/plans/2026-09-28-cluster-lod-design.md` (measured numbers)

- [ ] **Step 1: Add the build-time benchmark**

Append to `tests.rs`:
```rust
/// `cargo test -p kansei-core --release --lib build_time -- --ignored --nocapture`
#[test]
#[ignore]
fn build_time() {
    let rock = rock(7, false);
    let start = std::time::Instant::now();
    let mesh = ClusterMesh::build(&rock, &ClusterOptions::default());
    println!("{} triangles -> {} clusters in {:?}", rock.indices.len() / 3, mesh.clusters.len(), start.elapsed());
}
```

- [ ] **Step 2: Run it and the wasm check**

Run:
```bash
cd rust
cargo test -p kansei-core --release --lib build_time -- --ignored --nocapture
cargo check -p kansei-core --target wasm32-unknown-unknown
cargo clippy -p kansei-core --all-targets 2>&1 | grep -A5 "clusters/"
```
Expected:
- about 327,680 triangles to about 7,600 clusters in about 0.5 s (the prototype measured 0.46 s on this Mac);
- the wasm check finishes;
- no clippy warnings in `clusters/`.

Write the measured time into the design doc's "Risks → build time" line.

- [ ] **Step 3: Run the whole suite**

Run: `cd rust && cargo test -p kansei-core`
Expected: every test passes (the new 9 among them).

- [ ] **Step 4: Commit and open the PR**

```bash
git add rust/kansei-core/src/clusters docs/plans/2026-09-28-cluster-lod-design.md
git commit -m "test(clusters): build time, and the design's numbers"
git push -u origin fm/kansei-cluster-lod-m1
```
Open the PR against `development`. Title: `feat(clusters): cluster LOD graph (M1)`. The body gives the per-level triangle counts of the rock, the cut tests, the build time and the wasm check.

---

## Milestones after M1

Each starts with its own task plan (written against the code M1 leaves), and ends in one PR.

### M2: GPU clusters for the camera
- **Upload:** `ClusterMesh` to storage buffers. Vertices stay the geometry's. Local triangles are packed 4 bytes per triangle. Cluster records are `#[repr(C)]` and checked against their WGSL size (`shaders_validate_*` pattern).
- **Selection per view:** an expansion pass (per visible instance, the graph levels its distance range can reach; a prefix sum), then a cluster pass (the M1 cut rule with a per-view `lod_error_scale`, the frustum, the backface cone). It appends `(instance slot, cluster)` to each view's draw list and writes `draw_indirect` args.
- **`InstanceCulling` gains `InstanceTransform`:** a position offset plus optional scale, yaw and quaternion fields, for moving spheres and cones.
- **Generated vertex stage** around the material's `vertex_main`: struct-parameter and located-parameter forms, and instance attributes decoded from the geometry's instance layouts. The cluster variant of group 2 has storage bindings 2-6, with the draw list at a dynamic offset. Pipelines are cached under a cluster flag.
- **Example `cluster-lod`:** a field of high-poly rocks, with `bench=1` alternating the cluster and ordinary paths in-page, as the occlusion-culling example does.
- **Done when:**
  - the GPU cut equals `ClusterMesh::select` for the same view (GPU test, read back);
  - the image at τ = 0 equals the ordinary path's;
  - the example shows a measured GPU-time win at τ = 1, or the PR reports why not before going on.

### M3: every view
- Spot shadows, cascades, the sky-occlusion top-down view, velocity, the rendered planar reflection, and the impostor bake draw clusters.
- Per-view `lod_error_scale` (shadow and sky views coarser), and `CullStats` with clusters and triangles per view.
- **Done when:** shadows and reflections match the ordinary path at τ = 0, and shadow-view triangle counts fall with the scale.

### M4: card clusters (foliage)
- Card detection per connected component, and stochastic-pruning levels (Cook et al. 2007) that scale the kept cards by the inverse kept fraction. Alpha test in every pass.
- **Done when:** the film's spruce is built as clusters, with stills against today's LOD0/1/2 at their switch distances, coverage held, and the triangle counts per shot measured.

### M5: occlusion per cluster
- Two phases with the existing depth pyramids, and a visibility bit per candidate slot.
- **Done when:** a dense forest scene culls hidden clusters (stats), with no popping at disocclusion (a turning camera, stills).

### M6: the film (app side, with the web lead)
- Trees, debris and terrain on clusters, the discrete LOD bands retired where clusters replace them, the impostor kept as the far band.
- **Done when:** per-shot triangles and GPU time are measured before and after with the in-page A/B, and stills are compared.
