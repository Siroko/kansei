# Cluster LOD (meshlets) design

## Goal

Make the cost of geometry follow the pixels it covers rather than the triangles it was modelled with. A renderable opts into **cluster LOD**. Its mesh is split into clusters of about 128 triangles, and a graph of coarser versions is built over them. Each frame, and for each view (the camera, every shadow map, the sky occlusion's top-down view), the GPU picks per cluster the coarsest version whose error stays under a pixel, culls what that view can't see, and draws the rest. Nothing to author: no hand-made LOD meshes, no distance thresholds, no pops at LOD switches, and a mesh can be half near and half far.

This is the idea behind Unreal's Nanite, within what WebGPU allows (below). It is a general Kansei feature. The Midsommar film is the first user and the yardstick.

## Non-goals

- A software rasterizer for sub-pixel triangles. Nanite rasterizes tiny triangles in compute with 64-bit atomics into a visibility buffer. WebGPU has no 64-bit atomics, so every triangle goes through the hardware rasterizer. Keeping triangles at about a pixel is the LOD selection's job.
- Streaming cluster pages from disk. Every level stays resident; the film's meshes are small.
- Replacing discrete LODs everywhere. Meshes below a cluster's size (the film's 16-triangle grass cards) gain nothing and keep `InstanceCulling`'s LOD bands.

## What the film tells us

Measured in the lake shot, the frame drew 11.8M triangles for 0.87M pixels: 5.2M for the camera, 4.1M for the mirror (gone since screen-space reflections, #64), and 1.2M for each headlamp's shadow map. Moving the trees' LOD switches 1.5× nearer halved those counts and saved 1-3.6 ms per shot. So geometry is the cost, and LOD distance is the lever.

The film's geometry is mostly **alpha-tested cards**:

| mesh | triangles per LOD | cards | instances |
|---|---|---|---|
| spruce | ~2,900 / 800 / 150 + impostor | ~84% of each LOD | 24,263 |
| birch | ~1,160 / 470 / 65 + impostor | ~33% | 2,127 |
| grass, flowers, shrubs | 6-36 | all | ~32,000 |
| debris (rocks, stumps) | two LODs | none (solid) | 1,160 |

Two consequences:
1. Cluster LOD on solid meshes (bark, debris, the car, terrain) is the general win. In this film, though, solid meshes are the minority of triangles.
2. The film's gain depends on **card-aware LOD** for foliage. Edge-collapse simplification can't reduce cards: every card vertex is a border vertex. Card clusters instead get stochastic pruning (below), which also fixes the film's thinning coarse LODs, where crowns go lighter. It is a milestone of its own, not an afterthought.

Expected for the film, to be confirmed by measurement: an error-driven cut instead of hand-tuned screen-height thresholds, cluster-level frustum and backface culling inside a tree, and coarser cuts in the shadow views. Together those should bring the camera's 5.2M triangles to roughly 1.5-2.5M, with the pops of LOD switches gone. Cluster-level occlusion inside the forest (a later milestone) is on top of that.

## WebGPU constraints and what they decide

| constraint | consequence |
|---|---|
| no mesh or task shaders | clusters are drawn by an ordinary vertex shader that fetches its own vertices (vertex pulling) |
| no multi-draw-indirect | one indirect draw per renderable per view, as `InstanceCulling` does today: its instance count is the number of clusters drawn |
| no 64-bit atomics | no packed depth+id atomics; the selection writes a compacted list with a 32-bit counter |
| at most 4 bind groups, all in use | the cluster buffers go into a cluster variant of group 2 (the vertex-only mesh group) |
| indirect `firstInstance` needs an optional feature | each view's list of drawn clusters is bound with a dynamic storage offset |
| vertex-stage storage buffers are read-only | fine: the vertex shader only reads |

This keeps the renderer's structure: every pass still issues one indirect draw per renderable per view. Render bundles, pass order and the views are unchanged.

## 1. Building the cluster graph (CPU, at load, in wasm too)

`clusters::ClusterMesh::build(&Geometry, &ClusterOptions)`, on the CPU. The film generates its trees in wasm at load, so the builder runs in the browser. It uses **`optimesh`**, a pure-Rust port of meshoptimizer 1.1: bit-exact against the C++ (differential-tested), no `unsafe` by default, builds for `wasm32-unknown-unknown`, no dependencies. From it we use `build_meshlets`, `partition_clusters`, `simplify_with_attributes` and `compute_cluster_bounds`. The graph itself follows meshoptimizer's cluster-LOD demo (`clusterlod.h`) and Nanite:

0. **Weld.** Positions within a millionth of the mesh's size are made bit-identical (and `-0.0` becomes `0.0`). The simplifier and the locks match positions exactly, so a seam whose copies differ by rounding (the engine's own UV sphere's last column) or by the sign of zero would otherwise open cracks at coarser levels.
1. **Level 0.** Split the mesh into clusters of at most 124 triangles and 128 vertices (`build_meshlets`, cone weight 0.25 so clusters face one way and backface culling works). At 128 vertices clusters fill to ~96% of their triangles; at 64 they reach only ~70%, and every cluster is drawn as 124 triangles. Each gets its bounding sphere, its normal cone (apex, axis, cutoff; `Cluster::backfacing` is the reference test), and error 0.
2. **Group.** Partition the current clusters into groups of about 8 neighbours (`partition_clusters`, spatial). It is given one vertex per position, so clusters on either side of a uv seam are neighbours; by vertex index a rock in 16 uv islands was cut from 40 m with 3,870 triangles against 632 in one piece.
3. **Simplify each group** to half its triangles (`simplify_with_attributes`, weighted on normals and uvs, absolute error). Vertices the group shares with clusters outside it are locked (`SIMPLIFY_VERTEX_LOCK` in the per-vertex lock array, found by position so uv seams don't count as boundaries). A neighbouring group's clusters therefore meet this one's at the same vertices, whatever level each is drawn at: no cracks. The mesh's own open borders are *not* locked by default (unlike `SIMPLIFY_LOCK_BORDER`), so open meshes still reduce.
4. **Error and bounds.**
   - The group's error is the simplifier's error plus nothing less than its children's: `max(simplify_error, max(child.error))`.
   - Its LOD sphere is the union of its children's LOD spheres.
   - Every child records the group's error and sphere as its *parent*. Errors grow and spheres contain each other up the graph, which is what makes the selection below consistent.
5. **Re-split** the simplified group into clusters (step 1), which carry the group's error and sphere as their own. They form the next level.
6. **Stalls.** A group that won't lose 15% of its triangles is not simplified. Its clusters go into the next round unchanged, to be grouped with other neighbours. A round with no progress ends the build. Clusters that never get a parent have parent error ∞.

The vertex buffer is the geometry's own. Simplification only drops and reuses vertices (no new positions), so every level shares one buffer. The graph adds only indices: the coarser levels together hold about as many triangles again as the mesh (each halves the one below). Clusters store 8-bit local indices and a list of global vertex indices.

A prototype of this build, the exact code of the M1 plan, was run against `optimesh` 1.1. On a noisy icosphere of 20,480 triangles it gave 9 levels (20,480 → 10,216 → 5,098 → … → 78). Every cut, over 4 viewpoints × 5 budgets, with and without a uv seam, was free of holes and overlaps, and it compiles for wasm32. The M1 build (after its review: welding, grouping by position, 128 vertices) turns 327,680 triangles into 5,513 clusters, 96% full, in 0.53 s (release, native). Two behaviours to know:
- **Meshes whose every edge is an attribute seam** (flat-shaded, a normal per face) don't reduce. The attribute-preserving simplifier moves no seam vertex, so they keep one level. Welding normals before the build fixes that, if such content appears.
- **At extreme reductions the simplifier can fold an edge inside one coarse cluster** (four triangles on one edge). That's a local artifact, not a crack. The closure test tells the two apart.

**The cut rule.** For a view, draw cluster `c` iff `projected(c.error, c.lod_sphere) ≤ τ < projected(c.parent_error, c.parent_sphere)`. Here τ is the error budget in pixels (1 by default), and `projected(e, s) = e / max(distance(eye, s.center) − s.radius, near) × pixels_per_radian`. Because a parent's sphere contains its children's and its error is at least theirs, each point of the surface passes the rule in exactly one cluster: a crack-free, overlap-free cut, decided per cluster with no traversal. A CPU reference of this rule (`ClusterMesh::select`) is what the tests and the GPU are checked against.

## 2. Selecting and culling on the GPU, per view

A cluster renderable keeps `InstanceCulling` for its instances (non-instanced renderables are one instance). Per frame and view:

1. **Instances** are culled as today (frustum, casters-only, layers, and later occlusion). This gives each view its compacted list of visible instance records.
2. **Expansion.** Testing every cluster of every visible instance would be instances × clusters (20,000 trees × 25 clusters is 500,000 tests per view: acceptable). For big meshes that product explodes. So one pass computes, per visible instance, the range of graph levels its cut can reach, from the instance's nearest and farthest distance. It counts the clusters in those levels, and a prefix sum lays the candidates out.
3. **Cluster test**, one thread per candidate:
   - the cut rule (per view: `τ × lod_error_scale`, so shadow maps and the sky's top-down view take coarser cuts);
   - the view's frustum against the cluster's sphere, moved by the instance's transform;
   - the backface cone;
   - later, Hi-Z occlusion (milestone 5).

   Survivors append `(instance slot, cluster)` to the view's draw list and count their triangles into `CullStats`.
4. **Draw args.** `draw_indirect` with `vertex_count = 3 × max_triangles` and `instance_count = drawn clusters`. Triangles past a cluster's own count come out degenerate (all three corners at one vertex), so the rasterizer drops them. The clusters are filled to about 96% by the builder (at 128 vertices per cluster), so the padding is a few percent of vertex work. Compacting a triangle list instead is a later option if the measurements ask for it.

Instances need a transform the cull shader can read, to move cluster spheres and cones. `InstanceCulling` gains an `InstanceTransform` description of the record: position offset, optional uniform-scale field, optional yaw field (radians, about +y), optional quaternion. That covers the film's `X, Y, Z, height, bearing, …` records and the usual layouts. A missing rotation disables cone culling for that renderable, which stays correct.

**As built (M2).** The camera's pass only; the other views follow in M3.
- **No prefix sum.** Each visible instance is one workgroup. It walks only the levels its cut can reach and strides over their clusters. `LevelBounds::may_draw` decides this conservatively from the instance's distance, the level's smallest error and largest parent error, and how far its spheres reach from the mesh's origin.
- **One indirect dispatch per renderable.** A one-thread `prepare` pass sizes it from the visible-instance count.
- **One buffer for the mesh.** Vertices, cluster vertices, triangles, and cluster and level records share one storage buffer.
- **Backface culling is a switch** (`ClusterLod::cone_culling`, on by default), skipped for mirroring transforms.
- **The draw list has a capacity.** Clusters past it are counted and not drawn.
- **`CullStats`** comes with M3; the draw's arguments already count the triangles drawn.

## 3. Drawing: vertex pulling around the material's own `vertex_main`

Materials stay as they are. For a cluster renderable the engine generates the vertex stage from the material's WGSL:

- `@vertex fn vertex_main(...)` becomes a plain function. Its `@location` / `@builtin` parameter attributes are stripped when it takes located parameters. When it takes a struct, the struct is kept: naga accepts a struct with `@location` members as a plain parameter, but rejects located parameters on a plain function (probed with naga 24).
- A new `@vertex fn kansei_cluster_vertex_main(@builtin(vertex_index), @builtin(instance_index))` finds its draw-list entry, the cluster, the triangle and the vertex. It reads the vertex (position, normal, uv: Kansei has one vertex layout) and the instance record's attributes, decoded from the formats of the geometry's instance-buffer layout (`Float32xN`, `Uint32xN`, `Unorm8x4`, `Snorm16x2`, and so on). It fills the material's inputs and calls it.
- The same generated stage serves every pass. Shadow, reflection, velocity and sky-occlusion pipelines already reuse `vertex_main` (`Material::get_depth_pipeline`).
- Group 2 for cluster pipelines keeps bindings 0 and 1 (normal and world matrices, dynamic offsets) and adds read-only storage: vertices, local triangles, cluster records, instance records, and the view's draw list (dynamic offset). That's 5 storage buffers against WebGPU's 8 per stage.
- Pipelines are cached under a cluster flag next to the existing keys.

Materials whose vertex stage reads builtins other than its located inputs (`vertex_index`, `instance_index`) aren't supported on the cluster path, and the engine says so when the renderable is added. The transformation is validated by naga in tests on every material in the repository's examples.

## 4. Foliage: card clusters (milestone 4)

Cards (quads and ribbons, alpha-tested, `CullMode::None`) are detected per connected component: small, open, about planar. Their clusters don't simplify by edge collapse. Their levels are built by **stochastic pruning** (Cook, Halstead, Planck, Ryu 2007): each coarser level keeps a spatially stratified subset of the group's cards and scales each kept card's area by the inverse of the kept fraction, so the crown's coverage and silhouette density hold. A pruned level's error is the mean spacing its missing cards open up, measured like any other error, so the same cut rule applies.

Card clusters draw with the material's alpha test in every pass (`shadow_fragment_entry` as today). The impostor stays as the far band where even the root cut is too many triangles for its pixels. The handover is by projected size, as `InstanceCulling`'s bands work today.

**As built (M4).**
- **Opt-in.** Cards are opt-in (`ClusterOptions::cards`): a solid mesh made of separate flat panels would otherwise lose them at a distance.
- **Detection.** A card is a connected component (by welded position) with at most `card_max_triangles` (64) triangles and at most `max_vertices`, open, keeping `card_flatness` (0.5) of its area in its area-weighted normal. A tube or a closed shape cancels out. The rest of the mesh takes M1's path, and each build round runs both.
- **Level 0.** Whole cards are packed into clusters in Morton order.
- **Each round.**
  - Card clusters are grouped by Morton order.
  - In a group, each card is paired with its nearest free neighbour, and one of each pair is kept (by a hash).
  - The kept card stands for both. It is scaled about its centroid to cover their area and drawn at the area-weighted centre of what it stands for, as new vertices.
  - A group's error is `card_error_scale` × the growth of its typical card size, √(A/k) − √(A/n₀), and never less than a child's.
- **The cap.** Pruning stops at `card_max_scale` (4: at most a sixteenth of the cards). Below that, a few giant cards would carry the crown's area away from where it was; the impostor takes over there.
- **Tests.** A crown of 800 cards keeps its drawn area within 10% at every cut. From 4 km, each part of the crown stays within 20% of its own area (pairing along Morton order alone: 41%). A mixed mesh keeps its solid part crack-free.
- **The film's trees.** They needed `InstanceTransform::Placement::yaw_scale` (−1: a bearing turns the mesh by minus itself) and `ClusterLod::stretch` (1.15: widths up to 1.1× the height, and the sway). A stretch turns cones off.

**Evaluation on the film's spruce** (a scratch clone: LOD0 foliage on card clusters against today's three LODs; bark, shadows and the mirror unchanged):
- **Build.** The graphs build in 6–8 ms each in wasm: 68–71 clusters over 9–11 levels from ~2,400 foliage triangles.
- **Coverage: cards cover slightly more, and far crowns lose their spires.** Coverage is the share of the frame's upper 55% darker than the sky's 95th-percentile luma. From today's LODs to cards (`card_error_scale` 1 down to 0.25, which barely moved it):
  - at 6 s, +3.7%;
  - at 30 s, +2.5–4.3%;
  - at 39, 48 and 59 s, within ±0.2%.

  The stills show what the extra coverage is (`docs/plans/cluster-lod-m4/`, `card_error_scale` 0.25; the default of 1 was not compared by eye). Near crowns match. The far shore at 30 s loses its spires: kept cards move to their groups' area-weighted centres and grow up to 4x, so a distant crown fills out into a rounded, lumpy mass where today's LODs keep a spruce's point.

  Not measured: a reference with LOD0 forced everywhere, so whether today's LODs thin the crowns is an inference; and the triangles drawn per shot, so how deep the cut pruned at each shot is unknown. The test crown's pruning stops at 126 of 1,600 triangles (about 1/13), not the cap's 1/16.

  So cards are not ready for the film's far trees as they are. Worth trying next: a lower `card_max_scale` (2), keeping kept cards near their own positions at a crown's silhouette, or handing distant crowns to impostors.
- **Cost is not measured yet.** The whole film is GPU-bound on the shared Mac (35–72 ms a frame headless), and the runs were stopped for the machine's load. What to expect:
  - The cap would leave about 150 of 2,400 foliage triangles, roughly today's LOD2 (146); the test crown stops a little above it (1/13).
  - M2 measured vertex pulling about 15% slower than discrete LODs.

  So cards buy coverage and no LOD pops rather than triangles. M2's proposed compacted index buffer is what would change that.

## 5. Occlusion (milestone 5)

Two phases per cluster, with the existing depth pyramids (`DepthPyramid`, linear pyramids for oblique views):
- Early: clusters visible last frame (a bit per candidate slot, as `InstanceCulling` keeps per instance) are drawn.
- Late: the rest are tested against the pyramid of the early depth.

Per-instance occlusion didn't pay in the film (#37): a tree is rarely wholly hidden. A cluster is a much smaller thing to hide.

## 6. Stats and debugging

- `CullStats` gains clusters and triangles per view.
- A debug view colours clusters by id or by graph level.
- `ClusterMesh` reports its level count, triangles per level, and build time.

## Risks

- **Vertex pulling cost.** Fetching from storage instead of the vertex-input hardware, plus the padding, costs more per vertex. It pays only where selection removes more than it adds. Measured in M2 (example `cluster-lod`, in-page A/B, 1280 x 720, rocks of 81,920 triangles at a 1-pixel budget):
  - Against the full mesh, clusters are 4.5x faster at 4,096 rocks (2.47 vs 11.08 ms a frame) and 3x at 576.
  - Against four discrete LODs cut from the same graph at the same budget, they are about 15% slower (1.14-1.24 vs 1.01-1.04 ms a frame).
  - The cull costs 0.09 ms at 4,096 rocks. The rest is the draw: a non-indexed draw shades 3 vertices a triangle, where an indexed mesh reuses about 0.6, and every cluster is drawn as 124 triangles.
  - Small rocks are discrete LODs' best case: a whole rock is near or far. A compacted index buffer, with the cull writing each drawn cluster's triangles for one indexed indirect draw, would recover vertex reuse and drop the padding.
- **Rewriting material WGSL.** It's a text transformation over two known forms. It's covered by naga validation of every material in the repository, and it fails loudly (the renderable keeps the ordinary path) rather than drawing wrong.
- **Build time in wasm.** Natively, 327,680 triangles build into 5,513 clusters in 0.53 s (release, `clusters::tests::build_time`). The film builds ~30 tree meshes of 1-3K triangles, which is milliseconds even at wasm's speed. A mesh of millions of triangles would take seconds, which is the case for an offline build (the same code, run natively, serialised).
- **Foliage quality.** Stochastic pruning is proven for foliage, but the film's crowns are ribbons, not free cards. Milestone 4 compares stills against today's LODs before the film adopts it.

## Milestones

Each is one PR against `development`. Task-level plans are written when a milestone starts. The first is in `2026-09-28-cluster-lod-plan.md`.

| | scope | done when |
|---|---|---|
| M1 | `ClusterMesh::build` and the CPU cut rule (`select`), with `optimesh` | every cut of a closed test mesh is closed (each edge used twice) at any budget and viewpoint; errors and spheres nest; builds in wasm |
| M2 | GPU path for the camera: upload, per-view expansion and cluster test, generated vertex stage, instancing, example `cluster-lod` with an in-page A/B | image equal to the ordinary path at τ = 0; the GPU cut equals the CPU reference; a measured win on a dense mesh field |
| M3 | every view: spot shadows, cascades, sky occlusion, velocity, rendered planar reflection, impostor bake; per-view `lod_error_scale`; stats | shadows and reflections match the ordinary path at τ = 0; shadow-view triangle counts fall with the scale |
| M4 | card clusters: stochastic pruning, alpha-tested in every pass | the film's spruce, stills against today's LODs, coverage held |
| M5 | two-phase occlusion per cluster | a forest scene culls hidden clusters; no popping at disocclusion |
| M6 | the film (app side, with the web lead): trees, debris, terrain on clusters | per-shot triangles and GPU time before and after, stills |
