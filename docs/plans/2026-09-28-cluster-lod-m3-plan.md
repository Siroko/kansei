# Cluster LOD, milestone 3: every view — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every view that draws the scene (spot shadows, cascades, sky occlusion, rendered planar reflections, velocity) draws a clustered renderable's cut for *that* view, at a per-view error scale, and the renderer reports clusters and triangles drawn per view. Until now those views draw the full mesh.

**Architecture:**
- **One cull, many views.** `ClusterCulling` holds a storage array of views (`ClusterViewGpu`), written once a frame. Each renderable has one *cut* per view that draws it: its own parameters (with the view's index), draw list, indirect draw and dispatch. One compute pass runs every cut. (`queue.write_buffer` is immediate: a single view uniform rewritten per view would leave every cut with the last view.)
- **Orthographic views.** Cascades and the sky's top-down view are orthographic: an error there is `error × pixels per metre`, whatever the distance. `LodView` and `ClusterViewGpu` gain `orthographic`. `pixels_per_radian` (perspective) and pixels per metre (orthographic) are both `height / 2 × |P[1][1]|` of the view's projection.
- **Per-view error scale.** A view's budget is `cluster_threshold × lod_error_scale`. The camera's scale is 1. Shadows (spot and cascades) use `Renderer::set_shadow_cluster_error_scale`. Reflections use `PlanarReflection::lod_error_scale`, and the sky uses `SkyOcclusionOptions::lod_error_scale`. These mirror `lod_distance_scale`.
- **Drawing.** The shadow and sky passes use a cluster *depth* pipeline, the velocity pass a cluster *velocity* pipeline, and reflections the existing cluster pipeline (same key as the camera's). Each draws its view's cut with that cut's group 2.
- **Stats.** `CullStats` gains `clusters`. A clustered renderable's triangles come from its cut (the cull counts them) instead of its instance draw, which it no longer issues.

**Tech Stack:** Rust, wgpu 24 (headless GPU tests), naga.

**Spec:** `docs/plans/2026-09-28-cluster-lod-design.md` §2, §6 and milestone M3 ("every view: spot shadows, cascades, sky occlusion, velocity, rendered planar reflection, impostor bake; per-view `lod_error_scale`; stats — shadows and reflections match the ordinary path at τ = 0; shadow-view triangle counts fall with the scale"). M2 plan: `docs/plans/2026-09-28-cluster-lod-m2-plan.md`.

## Where M3 departs from the design (decided here)

1. **The impostor bake keeps the full mesh.** The bake draws each part once, into a few hundred pixels per frame, when the impostor is made (not per frame). A cut at the bake's resolution would bake coarse levels into the impostor, which is then seen at many sizes. The full mesh is the right reference, and the bake is not a per-frame cost.
2. **No dynamic storage offsets.** Each cut has its own buffers and bind groups, as `InstanceCulling`'s chunks do. A view that doesn't draw a renderable gets no cut for it (no buffers).
3. **No cone test in orthographic views.** A directional view has no eye point. Its clusters skip the backface cone, which stays correct and only culls less. Spot shadows keep cones, with the light's position as the eye.
4. **Clusters stay single-phase in reflections**, as on the camera (M2): the cut reads the view's compacted instances and draws nothing in a late phase.
5. **Point-light cubemap shadows are out of scope.** They are a legacy path outside the cull views (no `InstanceCulling`, one fixed pipeline) and draw the mesh as before.

## Global Constraints

- M1, M2 and M4 tests pass untouched, except where a signature they call changes (`LodView` gains a field; `ClusterViewGpu::new` gains arguments).
- `writeBuffer` is immediate: nothing per view is written into one shared buffer between encodes. One write of the whole view array per frame.
- Every `#[repr(C)]` struct is checked against its WGSL size (as `ClusterCullGpu`'s test does).
- The GPU never sees an infinity.
- Views that do not draw a renderable (not a caster in a shadow view, not on a reflection's layer mask) get no cut for it.
- PRs go against `development`, with no AI attribution. Headless GPU only; `about:blank` and `stop` after any browser capture.

## Review Focus

1. **Two views of one frame seeing each other's view data** (the last-write trap): each view's cut equals the CPU reference for that view, with several views in one pass. Pinned by Task 1's `every_view_gets_its_own_cut_in_one_pass`.
2. **An orthographic view taking a perspective cut** (distance shrinking an error that shouldn't shrink, or the level window skipping levels): the GPU equals the CPU reference for ortho views, and CPU ortho cuts are closed. Pinned by Task 1.
3. **A shadow with holes or doubled geometry at τ = 0** (a cut missing from a shadow view, or both mesh and cut drawn): the shadowed image equals the mesh path. Pinned by Task 3.
4. **A renderable in some views but not others** (not casting shadows, off a reflection's layer mask): no cut, and no draw in that view. Pinned by Task 2.
5. **Velocity of a clustered renderable** (drawn from the mesh while the GBuffer drew the cut: depth mismatch, no velocity): the velocity texture equals the mesh path's at τ = 0. Pinned by Task 4.

## File Structure

- `rust/kansei-core/src/clusters/mod.rs`: `LodView::orthographic`, `projected_error_at` for ortho.
- `rust/kansei-core/src/clusters/gpu.rs`: `ClusterViewGpu` (ortho, pixels), `ClusterCulling` views array, `ClusterGpu` per-view `Cut`s, `ClusterLod::prepare(view, ...)`.
- `rust/kansei-core/src/shaders/cluster_cull.wgsl`: `views` storage array; `params.view`; ortho projection; no cones in ortho views.
- `rust/kansei-core/src/renderers/renderer.rs`: `cluster_views`, `run_cluster_culling` over every view, cluster draws in the shadow, sky, reflection and velocity passes, `set_shadow_cluster_error_scale`, pipeline warm-up.
- `rust/kansei-core/src/materials/material.rs`: `get_cluster_depth_pipeline`, `get_cluster_velocity_pipeline`, and their lookups.
- `rust/kansei-core/src/culling/stats.rs`: `CullStats::clusters`; cluster readback entries.
- `rust/kansei-core/src/reflections/planar_reflection.rs`, `shadows/sky_occlusion.rs`: `lod_error_scale`.
- Tests: `clusters/tests.rs` (CPU ortho), `clusters/gpu_tests.rs` (GPU views, renderer passes).
- Docs: design §2 "As built (M3)", §6.

---

### Task 1: One cull pass, many views (core)

**Files:** `clusters/mod.rs`, `clusters/gpu.rs`, `shaders/cluster_cull.wgsl`, `clusters/tests.rs`, `clusters/gpu_tests.rs`.

**Interfaces:**
- Produces:
  - `LodView { eye, pixels_per_radian, near, threshold, orthographic: bool }`. With `orthographic`, `pixels_per_radian` is pixels per metre and `projected_error_at` ignores the distance.
  - `ClusterViewGpu::new(view_proj, eye, pixels_per_unit, near, threshold, orthographic)`.
  - `ClusterCulling::set_views(device, queue, &[ClusterViewGpu])`: one write.
  - `ClusterGpu::bind(device, queue, culling, view: u32, source, params) -> bool`.
  - `ClusterGpu::args(view)`, `draw_bind_group(view)`, `draws(view)` (test-only).
  - `ClusterCulling::encode(encoder, &[(&ClusterGpu, u32)])`: each pair is a cut to run.
  - `ClusterLod::prepare(..., view: u32, ...)`.
  - `ClusterCullGpu`: `pad0` becomes `view: u32` (still 128 bytes).

- [ ] **Step 1: failing tests.**
  - CPU (`tests.rs`), `orthographic_cuts_are_closed_and_ignore_distance`. With `orthographic: true`, the cuts of `rock(4, false)` are closed at each of `eyes()` and budgets 0.5/1/4. The cut from an eye 10 m away equals the cut from 100 m (same pixels per metre).
  - GPU (`gpu_tests.rs`), `every_view_gets_its_own_cut_in_one_pass`:
    - Three views in one `encode`: perspective near (threshold 0.5), perspective far (threshold 2), and orthographic top-down (`glam::Mat4::orthographic_rh`, 8 m wide, 512 px).
    - Each is placed on the three `placement_record` instances of the film test.
    - Each cut equals `expected_world` for its own view (`expected_world` gains orthographic handling: `error × scale × ppm`, no distance, no cones).
  - Mutation to check: drop the view index (always view 0), and the test fails.
- [ ] **Step 2: run and watch them fail** (compile errors first: the fields don't exist).
- [ ] **Step 3: implement.**
  - WGSL:
    - `@group(1) @binding(0) var<storage, read> views: array<ClusterView>`, with the view read as `views[params.view]`.
    - `ClusterView` gains `orthographic: u32`, in the place of one pad word.
    - `projected_at` returns `error × pixels` when orthographic.
    - The cone test only runs when the view isn't orthographic.
  - `ClusterCulling` grows the views buffer (STORAGE | COPY_DST, powers of two) and remakes its bind group.
  - `ClusterGpu`:
    - holds `mesh`, `vertex_count`, `cluster_count` and `empty` once;
    - holds `cuts: Vec<Option<Cut>>` indexed by view, where `Cut` holds `params`, `written`, `draws`, `capacity`, `args`, `dispatch`, `bound` and `draw` (M2's per-renderable state);
    - makes a cut on its first `bind` for that view.
  - Camera call sites pass view 0 (`MAIN_VIEW`), and the renderer compiles unchanged in behaviour.
- [ ] **Step 4: run** `cargo test -p kansei-core --lib clusters` (all green) and the whole lib suite.
- [ ] **Step 5: commit** `feat(clusters): one cull pass cuts many views, orthographic included`.

### Task 2: The renderer cuts every view; stats

**Files:** `renderers/renderer.rs`, `culling/stats.rs`, `reflections/planar_reflection.rs`, `shadows/sky_occlusion.rs`, `clusters/gpu_tests.rs`.

**Interfaces:**
- Consumes: Task 1's API.
- Produces:
  - `Renderer::set_shadow_cluster_error_scale(f32)` and `shadow_cluster_error_scale()`, 1 by default.
  - `PlanarReflection::lod_error_scale` and `SkyOcclusionOptions::lod_error_scale`, 1 by default.
  - `CullStats::clusters: u32`.
  - `fn cluster_views(&self, camera, main, target_height) -> Vec<Option<ClusterViewGpu>>`, in `cull_views` order.
  - `run_cluster_culling` makes cuts for every (renderable, view) where the view is `Some`, the renderable is drawn there (`cast_shadow` for casters-only views, the layer mask) and `c.view(view)` exists. It writes all the views once.
  - `fn cluster_cut(r, view) -> Option<(&ClusterGpu, u32)>`.

- [ ] **Step 1: failing tests** (`gpu_tests.rs`, renderer-level, `headless()`):
  - `shadow_views_cut_clustered_casters`:
    - Rocks from `rocks(.., clusters: true)`, lit by a spot light with shadows and a sun with cascades.
    - Stats are enabled with `enable_culling_stats` (or its equivalent). Draw ~4 frames until the stats arrive.
    - Assert that `stats.view(SpotShadow(0)).clusters > 0` and `stats.view(Cascade(0)).clusters > 0`.
  - `shadow_triangles_fall_with_the_scale`: at scale 1 and then 8, the spot view's `triangles` falls (strictly), and the camera's stays equal.
  - `renderables_off_a_view_get_no_cut_there`: with `cast_shadow = false`, no cut exists for the spot view.
- [ ] **Step 2: watch them fail.**
- [ ] **Step 3: implement.**
  - `cluster_views` per view:
    - camera: `main.lod_origin`, M2's pixels per radian;
    - spot slot: the light's position (from `slot.view.inverse()`), the atlas layer size × P[1][1] / 2, perspective;
    - cascade: the slot's projection, the map's size, orthographic;
    - reflection: the mirrored camera's position, the reflection's height, perspective;
    - sky: `sky.camera()`'s projection, its size, orthographic.
  - Threshold = `cluster_threshold × lod_error_scale`.
  - Stats:
    - `StatsReadback::record_clusters(view, args, 0)` copies the cut's 32 bytes;
    - the readback adds `clusters = min(claimed, capacity)` (word 5 capped via word 1: use word 1, the instance count drawn) and `triangles += word 6`;
    - a clustered renderable's instance-draw triangles are excluded from that view.
- [ ] **Step 4: run** the renderer tests and the full suite.
- [ ] **Step 5: commit** `feat(renderer): cluster cuts for every view, per-view error scale, cluster stats`.

### Task 3: Shadow and sky passes draw their cuts

**Files:** `materials/material.rs`, `renderers/renderer.rs`, `clusters/gpu_tests.rs`.

**Interfaces:**
- Produces: `Material::get_cluster_depth_pipeline(device, shared, instances, depth_format, bias) -> Result<&RenderPipeline, String>` and `cluster_depth_pipeline(&DepthPipelineKey)`.
  - Layout `[material, camera, cluster_mesh]`, with the generated vertex stage and the same fragment/cull/bias as `get_depth_pipeline`.
  - Warmed where the depth pipelines are (both render paths).

- [ ] **Step 1: failing tests:**
  - `spot_shadows_match_the_mesh_at_zero_error`: rocks on a ground plane, with the spot shadow falling on the ground. The final image with clusters (τ = 0) matches the mesh path within `compare`'s tolerance (differing × 200 < covered). Checked with at least one rock partly out of the light's frustum.
  - `cascade_shadows_match_the_mesh_at_zero_error`: the same under a sun with cascades.
  - `sky_occlusion_matches_the_mesh_at_zero_error`: the sky occlusion's depth (read back) matches the mesh path's.
  - Mutation to check: draw the camera's cut in the shadow pass, and the image differs.
- [ ] **Step 2: watch them fail** (the shadow pass still draws the mesh; the test must fail on *doubling* too: keep the mesh draw and add the cut draw, and it must fail. If it doesn't, the scene doesn't expose it; add a rock only the light sees).
- [ ] **Step 3: implement.** In `run_spot_shadow_pass`, `run_cascade_shadow_pass` and `run_sky_occlusion_pass`: when `cluster_cut(r, view)` and a cluster depth pipeline exist, set that pipeline, group 0, group 2 = the cut's draw group at the mesh offset, and `draw_indirect(args, 0)`; else `draw_geometry`.
- [ ] **Step 4: run** and the full suite.
- [ ] **Step 5: commit** `feat(renderer): shadow maps and sky occlusion draw cluster cuts`.

### Task 4: Reflections and velocity draw their cuts

**Files:** `materials/material.rs`, `renderers/renderer.rs`, `clusters/gpu_tests.rs`.

**Interfaces:** `Material::get_cluster_velocity_pipeline(device, shared, instances)` and `cluster_velocity_pipeline()`: the cluster layout (4 groups), velocity targets, and velocity depth state.

- [ ] **Step 1: failing tests:**
  - `planar_reflections_match_the_mesh_at_zero_error`: a rendered (not screen-space) planar reflection under rocks; the reflection texture (or final image) matches the mesh path's.
  - `velocity_of_clustered_renderables_matches_the_mesh`: a moving rock (world matrix changes between frames), with the velocity texture read back and compared against the mesh path's at τ = 0. Also at τ = 4, the velocity is written wherever the GBuffer drew the rock (no `NO_VELOCITY` holes where depth is the rock's).
- [ ] **Step 2: watch them fail.**
- [ ] **Step 3: implement.**
  - `draw_reflection` uses `CameraClusterDraw`-like substitution with the reflection view's cut (group 2 at the mesh offset; nothing in the late phase).
  - `draw_velocity` uses the camera cut (`MAIN_VIEW`) with the cluster velocity pipeline.
  - `CameraClusterDraw` becomes `ClusterDraw { pipeline, group, args }`, with `of(r, view, pipeline)`.
- [ ] **Step 4: run** and the full suite.
- [ ] **Step 5: commit** `feat(renderer): reflections and velocity draw cluster cuts`.

### Task 5: Docs, example, review, PR

- [ ] Design §2 "As built (M3)" (views, ortho, per-view scale, memory: a cut's draw list per view, so `ClusterLod::capacity` bounds each), §6 stats as built, and the milestone table's rulings (impostor bake, cubemap shadows).
- [ ] Example `cluster-lod`: a spot light with shadows and a shadow-scale slider (the in-page A/B shows shadow-view triangles). Measure one A/B on a quiet machine only if the machine is free; otherwise state it is unmeasured.
- [ ] Final whole-branch review, fix pass, PR against `development` with the numbers and no AI attribution.
