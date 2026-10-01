# Cluster LOD, milestone 4: card clusters (foliage) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Foliage meshes of alpha-tested cards get cluster LOD too. Their coarser levels keep a spatially stratified half of each group's cards and enlarge the kept ones to hold the crown's card area (stochastic pruning), so the cut rule of M1–M2 draws a tree's crown with fewer cards where it's small on screen.

**Architecture:**
- **Detection.** `ClusterMesh::build` splits the mesh into cards (with `ClusterOptions::cards`): small, open, flat-ish connected components. The rest goes through M1's path unchanged.
- **Level 0.** Whole cards are packed into clusters in Morton order.
- **Each round.**
  - Card clusters are grouped by Morton order.
  - Each group keeps one card of each neighbouring pair. The kept card inherits its pair's area and is scaled about its centroid to cover it; the scaled copies are new vertices.
  - The group's error is the growth of its cards' typical size, times `card_error_scale`.
- **Output.** Card clusters are ordinary clusters, so the GPU cut, the generated vertex stage and the renderer need nothing new.
- **The film's trees.** They need an instance transform with a negated yaw and a stretch margin, both added here.

**Tech Stack:** Rust, optimesh 1.1 (bounds), wgpu 24 (GPU tests), the film clone for the evaluation.

**Spec:** `docs/plans/2026-09-28-cluster-lod-design.md` §4 and milestone M4 ("card clusters: stochastic pruning, alpha-tested in every pass; the film's spruce, stills against today's LODs, coverage held"). M1 plan: `docs/plans/2026-09-28-cluster-lod-plan.md`; M2 plan: `docs/plans/2026-09-28-cluster-lod-m2-plan.md`.

## Where M4 departs from the design (decided here)

1. **Cards are opt-in** (`ClusterOptions::cards`, off by default). Detection is per connected component, as designed, but a solid mesh built of separate flat panels (walls, signs) would otherwise have its panels pruned away at distance.
2. **The error has a scale** (`ClusterOptions::card_error_scale`, 1 by default). The error is the design's: the growth of the group's typical card size. At 1 pixel, though, the film's big sprays would never prune within the mesh bands, while its hand-made LOD1 drops 58% of the branches at 50–75 m. Task 5 calibrates the scale on the film's spruce against today's LOD distances.
3. **"Alpha-tested in every pass" is the camera's pass only.** Other views move to clusters in M3. Card materials cull nothing (`CullMode::None`), so M2's fix already keeps cone culling off for them.
4. **No colour or contrast adjustment** of kept cards (Cook et al. adjust both). Area is held; the stills in Task 5 show whether more is needed.

## Global Constraints

- M1's properties hold unchanged for solid meshes. Every existing test in `clusters::tests` and `clusters::gpu_tests` passes untouched.
- Clusters stay stored by build round (`ClusterMesh::levels` relies on it). Card and solid clusters of one round are appended together.
- A card fits one cluster: `card_max_triangles` ≤ `max_triangles`, and its vertices ≤ `max_vertices` (it is left solid otherwise).
- Scaled cards are new vertices in `ClusterMesh::vertices`. The renderable's own geometry is untouched: the other views still draw it.
- The GPU never sees an infinity. Every `#[repr(C)]` struct is checked against its WGSL size.
- PRs go against `development`, with no AI attribution. Film measurements are in-page A/B, headless, with `about:blank` after.

## Review Focus

1. **A cut drawing a crown region twice, or not at all** (a level-0 card and the pruned copy that stands for it both drawn): drawn area per spatial cell stays near level 0's, at every eye and budget. Pinned by Task 2's `pruned_levels_hold_the_cards_area`.
2. **A mixed mesh (bark and cards in one) cracking its solid part:** cuts of the non-card clusters stay closed. Pinned by `a_mixed_mesh_prunes_its_cards_and_simplifies_the_rest`.
3. **Degenerate or odd cards** (zero-area triangles, a lone card in a group, cards over the vertex limit): no panic, and the card is kept or left solid. Pinned by Task 1 and 2 tests.
4. **The film's instance transform** (a yaw of minus the bearing, and widths up to 1.1× the height plus sway): the cull's spheres cover the drawn instance. Pinned by Task 4's GPU test.
5. **Coverage at a distance** (Cook's concern: pruned crowns look sparser or denser): measured on the film's spruce in Task 5 against today's LODs.

## File Structure

- Create `rust/kansei-core/src/clusters/cards.rs`: `Card`, `find_cards`, and the card rounds (pack, group, prune, scale).
- Modify `rust/kansei-core/src/clusters/build.rs`: the build routes cards, and each round runs solid and card groups. A shared `push_cluster` replaces `split`'s push.
- Modify `rust/kansei-core/src/clusters/mod.rs`:
  - `ClusterOptions { cards, card_max_triangles, card_flatness, card_error_scale }`;
  - `Cluster::card`;
  - `mod cards`;
  - `#[cfg(test)] mod card_tests`.
- Create `rust/kansei-core/src/clusters/card_tests.rs`.
- Modify `rust/kansei-core/src/clusters/gpu.rs` and `shaders/cluster_cull.wgsl`: `InstanceTransform::Placement::yaw_scale`, `ClusterLod::stretch`.
- Modify `rust/kansei-core/src/clusters/gpu_tests.rs`: the transform test.
- Modify `docs/plans/2026-09-28-cluster-lod-design.md`: §4 as built, and the evaluation.
- The evaluation (Task 5) is scratch, in a clone of the film, not in this PR. Its stills and numbers go in the PR and in `docs/plans/…-design.md`.

---

### Task 1: Finding cards

**Files:**
- Create: `rust/kansei-core/src/clusters/cards.rs`, `rust/kansei-core/src/clusters/card_tests.rs`
- Modify: `rust/kansei-core/src/clusters/mod.rs`

**Interfaces:**
- Produces:
  - `pub(super) struct Card { pub vertices: Vec<u32>, pub triangles: Vec<[u16; 3]>, pub centroid: Vec3, pub area: f32, pub radius: f32 }`, where `triangles` index into `vertices`
  - `pub(super) fn find_cards(indices: &[u32], positions: &[f32], position_ids: &[u32], options: &ClusterOptions) -> (Vec<Card>, Vec<u32>)`: the cards, and the indices of every other triangle
  - `ClusterOptions { cards: bool (false), card_max_triangles: usize (64), card_flatness: f32 (0.5), card_error_scale: f32 (1.0) }`

A component is a card when all of these hold:
- it has at most `card_max_triangles` triangles and at most `max_vertices` vertices;
- it is open (an edge used once, by position);
- its area-weighted normal keeps at least `card_flatness` of its area: |Σ nᵢAᵢ| ≥ flatness × ΣAᵢ. A card, even curved, keeps most; a tube or a closed shape cancels out.

- [ ] **Step 1: Failing tests** (`card_tests.rs`: `use super::*; use super::cards::*; use super::tests::rock;`)

```rust
use glam::Vec3;

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
```

- [ ] **Step 2: Run to see it fail** — `cd rust && cargo test -p kansei-core --lib clusters::card_tests`. Expected: FAIL to compile (`find_cards`, `ClusterOptions::cards` missing).

- [ ] **Step 3: Implement** (`cards.rs`; options in `mod.rs`, with their defaults and docs; `build.rs` exposes `#[cfg(test)] pub(super) fn position_ids_for_tests(positions) -> Vec<u32>` over its `position_ids`)

```rust
//! Cards: foliage's small, open, flat-ish pieces (sprays, leaves, ribbons), which edge collapse
//! can't reduce (every vertex is on a border). Their coarser levels are pruned instead: see
//! `ClusterMesh::build` and the card rounds below.

use glam::Vec3;
use std::collections::HashMap;

use super::ClusterOptions;

/// A card: its vertices (into the mesh's) and its triangles (into those), its area-weighted
/// centroid, its area, and its radius from the centroid.
#[derive(Clone, Debug)]
pub(super) struct Card {
    pub vertices: Vec<u32>,
    pub triangles: Vec<[u16; 3]>,
    pub centroid: Vec3,
    pub area: f32,
    pub radius: f32,
}

/// The triangles of `indices` split into cards and the rest (as indices), by connected component
/// over `position_ids` (welded positions): a card has at most `card_max_triangles` triangles and
/// `max_vertices` vertices, is open (an edge used once), and keeps `card_flatness` of its area in
/// its area-weighted normal.
pub(super) fn find_cards(indices: &[u32], positions: &[f32], position_ids: &[u32], options: &ClusterOptions) -> (Vec<Card>, Vec<u32>) {
    let p = |v: u32| Vec3::from_slice(&positions[v as usize * 3..v as usize * 3 + 3]);
    // components: union-find over positions
    let mut parent: Vec<u32> = (0..position_ids.len() as u32).collect();
    fn root(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for t in indices.chunks(3) {
        let a = root(&mut parent, position_ids[t[0] as usize]);
        for &v in &t[1..] {
            let b = root(&mut parent, position_ids[v as usize]);
            parent[b as usize] = a;
        }
    }
    let mut components: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, t) in indices.chunks(3).enumerate() {
        components.entry(root(&mut parent, position_ids[t[0] as usize])).or_default().push(i);
    }
    let mut roots: Vec<u32> = components.keys().copied().collect();
    roots.sort_unstable();
    let (mut cards, mut rest) = (Vec::new(), Vec::new());
    for r in roots {
        let triangles = &components[&r];
        let tri = |i: usize| [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
        let mut vertices: Vec<u32> = triangles.iter().flat_map(|&i| tri(i)).collect();
        vertices.sort_unstable();
        vertices.dedup();
        let (mut area, mut normal, mut centroid) = (0.0f32, Vec3::ZERO, Vec3::ZERO);
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for &i in triangles {
            let [a, b, c] = tri(i);
            let n = (p(b) - p(a)).cross(p(c) - p(a));
            let ta = 0.5 * n.length();
            area += ta;
            normal += n * 0.5;
            centroid += (p(a) + p(b) + p(c)) / 3.0 * ta;
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let (x, y) = (position_ids[x as usize], position_ids[y as usize]);
                *edges.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
        let open = edges.values().any(|&n| n == 1);
        let is_card = triangles.len() <= options.card_max_triangles && vertices.len() <= options.max_vertices && open && area > 0.0 && normal.length() >= options.card_flatness * area;
        if !is_card {
            rest.extend(triangles.iter().flat_map(|&i| tri(i)));
            continue;
        }
        let centroid = centroid / area;
        let local: HashMap<u32, u16> = vertices.iter().enumerate().map(|(j, &v)| (v, j as u16)).collect();
        let radius = vertices.iter().map(|&v| p(v).distance(centroid)).fold(0.0, f32::max);
        cards.push(Card { triangles: triangles.iter().map(|&i| tri(i).map(|v| local[&v])).collect(), vertices, centroid, area, radius });
    }
    (cards, rest)
}
```

- [ ] **Step 4: Pass** — same command. Expected: PASS.
- [ ] **Step 5: Commit** — `feat(clusters): finding cards: small, open, flat-ish components`.

---

### Task 2: Pruned levels of cards

**Files:**
- Modify: `rust/kansei-core/src/clusters/cards.rs`, `rust/kansei-core/src/clusters/build.rs`, `rust/kansei-core/src/clusters/mod.rs` (`Cluster::card`)
- Test: `rust/kansei-core/src/clusters/card_tests.rs`

**Interfaces:**
- Consumes: `Card`, `find_cards` (Task 1); M1's `ClusterMesh`, `Cluster`, `Sphere::enclosing`, `compute_cluster_bounds`.
- Produces:
  - `Cluster::card: bool`;
  - `ClusterMesh::push_cluster(vertices: &[u32], triangles: &[u8], positions: &[f32], error, lod_bounds, level, card) -> usize`, which `split` now uses;
  - in `cards.rs`: `Placed { card: u32, first_vertex: u32 (u32::MAX: the card's own vertices), represents: u32, area: f32 }`, `pack(mesh, cards, placed, positions, error, lod_bounds, level, options) -> Vec<(usize, Vec<Placed>)>` and `prune_round(...)`.

How a round works:
- **Grouping.** Sort the pending card clusters by the Morton code of their bounds' centre and chunk them into groups of `group_size`.
- **Pruning.** In each group, sort the placed cards by the Morton code of their centroid and pair neighbours. Keep one of each pair: the lower `hash(card, level)`, so it's deterministic. The kept card takes on its pair's `represents` and `area`, and is scaled about its centroid by √(area / card.area): the crown's card area is held. An odd card stays as it is.
- **Stalls.** A group with fewer than two cards stalls and is carried forward, as M1's stalls are.
- **Error.** The group's error is `card_error_scale × (√(A / k) − √(A / n₀))`, where A is the area the group represents, k its cards after pruning and n₀ the level-0 cards it stands for. It is never less than a child's.
- **LOD bounds.** A sphere around the children's, as in M1.

- [ ] **Step 1: Failing tests**

```rust
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
            for (k, (&a, &a0)) in cells.iter().zip(&cells0).enumerate() {
                if a0 > 0.03 * total0 {
                    assert!((a / a0 - 1.0).abs() < 0.35, "eye {eye}, {threshold} px: cell {k} holds {a} of {a0}");
                }
            }
            triangles.push(cut.iter().map(|&c| mesh.clusters[c].triangle_count).sum::<u32>());
        }
    }
    // from 4 km at 1 px, a small share of the cards
    let far = triangles[triangles.len() - 2];
    assert!(far * 8 < 1600, "{far} triangles from 4 km");
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
    assert!(quarter * 2 < full, "{quarter} with a quarter of the error, {full} with all of it");
}
```

(`eyes`, `view`, `position_keys` and `bad_edges_keyed` become `pub(super)` in `tests.rs`.)

- [ ] **Step 2: Run to see them fail** — `cargo test -p kansei-core --lib clusters::card_tests`. Expected: FAIL (compile: `Cluster::card`; or, once it compiles, no pruning: all clusters level 0).

- [ ] **Step 3: Implement**

`mod.rs`: `Cluster { …, pub card: bool }`, documented as "one of pruned cards (foliage), not simplified triangles".

`build.rs`:
- `ClusterMesh::build` computes positions, attributes, `position_ids` and `canonical` as today.
- With `options.cards`, it calls `find_cards`. The solid path gets `rest`, the card path the cards.
- The round loop keeps two pending sets. Each round runs M1's solid groups over `pending`, then `cards::prune_round` over `pending_cards`, appending each set's new clusters, all with `level = round`. It ends when neither progressed.
- `positions` becomes a `Vec<f32>` the card rounds append to, so bounds see the scaled copies.
- `split` pushes through `push_cluster(…, card: false)`.

`cards.rs` (continued):

```rust
use super::{ClusterMesh, Sphere};
use crate::geometries::Vertex;
use optimesh::meshletutils::compute_cluster_bounds;

/// A card at some level: which card, where its vertices are (u32::MAX: the card's own; else the
/// first of its scaled copy's, in the card's vertex order), how many level-0 cards it stands for,
/// and the area it covers.
#[derive(Clone, Copy, Debug)]
pub(super) struct Placed {
    pub card: u32,
    pub first_vertex: u32,
    pub represents: u32,
    pub area: f32,
}

impl Placed {
    fn vertex(&self, card: &Card, j: usize) -> u32 {
        if self.first_vertex == u32::MAX { card.vertices[j] } else { self.first_vertex + j as u32 }
    }
}

/// A 30-bit Morton code of `p` in `(lo, hi)`.
pub(super) fn morton(p: Vec3, lo: Vec3, hi: Vec3) -> u32 {
    let q = ((p - lo) / (hi - lo).max(Vec3::splat(1e-9)) * 1023.0).clamp(Vec3::ZERO, Vec3::splat(1023.0)).as_uvec3();
    let spread = |mut x: u32| {
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        (x | (x << 2)) & 0x0924_9249
    };
    spread(q.x) | spread(q.y) << 1 | spread(q.z) << 2
}

fn hash(a: u32, b: u32) -> u32 {
    let x = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77);
    (x ^ (x >> 15)).wrapping_mul(0x2C1B_3C6D)
}

/// Whole cards packed into clusters, in the order given (neighbours: Morton order), each up to
/// the options' vertex and triangle limits, with `error`, `lod_bounds` (their own when None) and
/// `level`; per new cluster, its index and its cards.
#[allow(clippy::too_many_arguments)]
pub(super) fn pack(mesh: &mut ClusterMesh, cards: &[Card], placed: &[Placed], positions: &[f32], error: f32, lod_bounds: Option<Sphere>, level: u32, options: &super::ClusterOptions) -> Vec<(usize, Vec<Placed>)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < placed.len() {
        let (mut v, mut t, mut end) = (0, 0, start);
        while end < placed.len() {
            let card = &cards[placed[end].card as usize];
            if end > start && (v + card.vertices.len() > options.max_vertices || t + card.triangles.len() > options.max_triangles) {
                break;
            }
            v += card.vertices.len();
            t += card.triangles.len();
            end += 1;
        }
        let (mut vertices, mut triangles) = (Vec::with_capacity(v), Vec::with_capacity(t * 3));
        for p in &placed[start..end] {
            let card = &cards[p.card as usize];
            let base = vertices.len() as u8;
            vertices.extend((0..card.vertices.len()).map(|j| p.vertex(card, j)));
            triangles.extend(card.triangles.iter().flat_map(|t| t.map(|j| base + j as u8)));
        }
        let index = mesh.push_cluster(&vertices, &triangles, positions, error, lod_bounds, level, true);
        out.push((index, placed[start..end].to_vec()));
        start = end;
    }
    out
}

/// One round over the card clusters still without a parent (`pending`): grouped in Morton order,
/// each group's cards pruned to one of each neighbouring pair (the kept one scaled to cover its
/// pair's area: new vertices, appended to the mesh and to `positions`), the children given their
/// parent's error and sphere, and the kept cards packed into this level's clusters. Returns the
/// next pending set and whether any group was pruned.
pub(super) fn prune_round(mesh: &mut ClusterMesh, cards: &[Card], pending: Vec<(usize, Vec<Placed>)>, positions: &mut Vec<f32>, level: u32, options: &super::ClusterOptions) -> (Vec<(usize, Vec<Placed>)>, bool) {
    let (lo, hi) = bounds_of(positions);
    let mut pending = pending;
    pending.sort_by_key(|(c, _)| morton(mesh.clusters[*c].bounds.center, lo, hi));
    let mut next = Vec::new();
    let mut progress = false;
    for group in pending.chunks(options.group_size.max(2)) {
        let mut placed: Vec<Placed> = group.iter().flat_map(|(_, p)| p.iter().copied()).collect();
        if placed.len() < 2 {
            next.extend(group.iter().cloned());
            continue;
        }
        progress = true;
        placed.sort_by_key(|p| morton(cards[p.card as usize].centroid, lo, hi));
        let (area, n0) = placed.iter().fold((0.0, 0), |(a, n), p| (a + p.area, n + p.represents));
        let mut kept = Vec::with_capacity(placed.len().div_ceil(2));
        for pair in placed.chunks(2) {
            let (a, b) = match pair {
                [a, b] => if hash(a.card, level) <= hash(b.card, level) { (*a, Some(*b)) } else { (*b, Some(*a)) },
                [a] => (*a, None),
                _ => unreachable!(),
            };
            let Some(b) = b else {
                kept.push(a);
                continue;
            };
            let card = &cards[a.card as usize];
            let (represents, covered) = (a.represents + b.represents, a.area + b.area);
            let scale = (covered / card.area).sqrt();
            let first_vertex = mesh.vertices.len() as u32;
            for &v in &card.vertices {
                let mut vertex: Vertex = mesh.vertices[v as usize];
                let p = card.centroid + (Vec3::from_slice(&vertex.position[..3]) - card.centroid) * scale;
                vertex.position[..3].copy_from_slice(&p.to_array());
                mesh.vertices.push(vertex);
                positions.extend_from_slice(&p.to_array());
            }
            kept.push(Placed { card: a.card, first_vertex, represents, area: covered });
        }
        let k = kept.len() as f32;
        let own = options.card_error_scale * ((area / k).sqrt() - (area / n0 as f32).sqrt());
        let error = group.iter().map(|(c, _)| mesh.clusters[*c].error).fold(own.max(0.0), f32::max);
        let bounds = Sphere::enclosing(group.iter().map(|(c, _)| mesh.clusters[*c].lod_bounds));
        for (c, _) in group {
            mesh.clusters[*c].parent_error = error;
            mesh.clusters[*c].parent_bounds = bounds;
        }
        next.extend(pack(mesh, cards, &kept, positions, error, Some(bounds), level, options));
    }
    (next, progress)
}

fn bounds_of(positions: &[f32]) -> (Vec3, Vec3) {
    positions.chunks(3).fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(Vec3::from_slice(p)), hi.max(Vec3::from_slice(p))))
}
```

Level 0 of the cards: sort all cards by the Morton code of their centroid, place each as its own (`first_vertex: u32::MAX, represents: 1, area: card.area`), and `pack` with error 0 and no LOD bounds.

`push_cluster` factors `split`'s push: `compute_cluster_bounds` over the cluster's global triangles, then the cone, bounds, error, LOD bounds, parent ∞, level and card.

- [ ] **Step 4: Pass, then the module** — `cargo test -p kansei-core --lib clusters::`. Expected: every card test and every M1/M2 test PASS.
- [ ] **Step 5: Commit** — `feat(clusters): pruned levels of cards: one of each pair kept, scaled to hold the area`.

---

### Task 3: The GPU draws card clusters

**Files:** Test: `rust/kansei-core/src/clusters/gpu_tests.rs`

- [ ] **Step 1: Test** `card_clusters_draw_through_the_camera_path`.
  - A crown of 800 cards, instanced 3 times, with an alpha-tested material (`CullMode::None`, the fragment discarding outside a disc in uv).
  - At τ = 0 the cluster image equals the mesh's (≤ 0.5% of covered texels differ): cards drawn from both sides, no cone culling.
  - At 2 px from 120 m: far fewer triangles, and the covered texel count within 15% of the full mesh's (coverage held).
- [ ] **Step 2: Run.** Expected: PASS with no new code, since card clusters are ordinary clusters and M2's cone switch already honours `CullMode::None`. If it fails, the failure is a defect in Tasks 1–2 or M2: debug it there.
- [ ] **Step 3: Commit** — `test(clusters): card clusters through the camera path, coverage held`.

---

### Task 4: Instance transforms for the film's trees

**Files:**
- Modify: `rust/kansei-core/src/clusters/gpu.rs`, `rust/kansei-core/src/shaders/cluster_cull.wgsl`
- Test: `rust/kansei-core/src/clusters/gpu_tests.rs`

**Interfaces:**
- `InstanceTransform::Placement { position, scale, yaw, yaw_scale: f32, rotation }`. `yaw_scale` multiplies the record's yaw: −1 for a bearing that turns a mesh by minus itself, as the film's trees are placed.
- `ClusterLod::stretch: f32` (1 by default), with `with_stretch`: how much further than `transform` the material may stretch or sway an instance (the film: widths up to 1.1× the height, a sway of 0.6% of it). The cull's scale bound, and so spheres and errors, grow by it.
- `ClusterCullGpu` gains `yaw_scale` and `stretch`, padded to 128 bytes.

- [ ] **Step 1: Failing test.** `a_negated_yaw_and_a_stretch_bound_the_film_s_trees`:
  - Placement records with a bearing, `yaw_scale: -1.0`, `stretch: 1.2`.
  - `expected_world` (M2's) takes the model with the negated yaw and the scale bound × 1.2.
  - The GPU cut equals it (`assert_cut`).
  - Mutation: with `yaw_scale` 1 instead, the test fails.
- [ ] **Step 2: Run to see it fail** (the fields don't exist).
- [ ] **Step 3: Implement** the fields, the params, and the WGSL: `a = record_f32(…) * params.yaw_scale`, `scale = … * params.stretch`. Update the struct-size test.
- [ ] **Step 4: Pass**, then `cargo test -p kansei-core`.
- [ ] **Step 5: Commit** — `feat(clusters): a yaw factor and a stretch margin for placed instances`.

---

### Task 5: The film's spruce: stills, coverage, calibration (evaluation, scratch)

In a clone of midsommar-web (read-only source; the clone and its changes stay in the scratchpad), with this branch's engine:

- [ ] **Step 1: Wire it** behind `?cards=1&ces=<scale>`.
  - For each spruce style, build a `ClusterMesh` from its LOD0 foliage (`cards: true`, `card_error_scale` from `?ces`).
  - Replace that style's three foliage LOD renderables with one cluster renderable over the same instances, in LOD0's near edge up to the impostors' band:
    `InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: -1.0, rotation: None }`, stretch 1.15, the same material.
  - Bark keeps today's LODs.
  - Log each mesh's levels, cards and build time.
- [ ] **Step 2: Stills** at the forest shots (6, 15.5, 39, 48, 59 s), today against cards at `ces` 1, 0.5, 0.25, 0.125. Crop the forest regions and compare side by side.
- [ ] **Step 3: Coverage.**
  - Coverage is the share of each forest region's pixels darker than the sky behind them (the crowns' silhouette coverage), per shot, today against cards.
  - Held means within 5% relative, with no visible thinning or thickening of crowns at the LOD distances.
- [ ] **Step 4: Cost.**
  - Triangles drawn: CPU `select` sums over the instances in view, per shot.
  - GPU time: in-page A/B (today against cards), with the camera held per pair.
- [ ] **Step 5: Calibrate.**
  - Pick the `card_error_scale` default at which crowns keep coverage at today's LOD distances (or nearer).
  - If no value holds coverage, record it: colour and contrast adjustment (departure 4), or a different error, is the next step, and the PR says so.
- [ ] **Step 6: Record** the stills (three pairs, jpg) in `docs/plans/cluster-lod-m4/` and the numbers in the design doc's §4.

---

### Task 6: Docs, review, PR

- [ ] **Step 1:** Design doc §4 as built: detection, the pairing, area held, the error and its scale, departures 1–4, and Task 5's numbers.
- [ ] **Step 2:** Whole suite, clippy on the files touched, wasm32 build.
- [ ] **Step 3:** Fresh whole-branch review, fix pass (each fix test-first), then the PR against `development` with the stills (absolute pinned URLs) and the numbers.
