# Cluster LOD, milestone 2: GPU clusters for the camera — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A renderable with `clusters` is drawn for the camera as the GPU's cut of its cluster graph, recomputed every frame: its instances culled as today, then per visible instance the clusters of the levels its cut can reach tested (the cut rule, the frustum, the backface cone) into a draw list, drawn by one indirect draw through a vertex stage generated around the material's own `vertex_main`.

**Architecture:** `clusters::gpu` packs a `ClusterMesh` into one `u32` storage buffer (header, vertices, cluster vertices, packed triangles, cluster records, level records). After instance culling, the renderer runs three dispatches per cluster renderable: `prepare` resets the draw and sizes the cull's indirect dispatch from the visible-instance count, `cull` runs a workgroup per visible instance, and `finish` clamps the draw to the list's capacity. A material's cluster pipeline comes from generated WGSL: `vertex_main` becomes a plain function, and `kansei_cluster_vertex_main` fetches its inputs and calls it. Every other view (shadow maps, reflections, velocity, impostor bakes) keeps the ordinary path until M3.

**Tech Stack:** Rust, wgpu 24 (WebGPU), WGSL, naga 24 (tests), optimesh 1.1 (M1's build).

**Spec:** `docs/plans/2026-09-28-cluster-lod-design.md` (sections 2 and 3, milestone M2). M1's plan, whose graph this draws: `docs/plans/2026-09-28-cluster-lod-plan.md`.

## Where M2 departs from the design (decided here)

1. **Expansion without a prefix sum.** The design lays out each visible instance's candidates with a prefix sum. Here each visible instance is one workgroup. It walks only the levels its cut can reach (`LevelBounds::may_draw`: a conservative per-level test from the instance's distance) and strides over their clusters. The same work is skipped, with no candidate buffer and no scan, in one pass. If wrong, the cost is imbalance on a single huge mesh (one workgroup of 64 threads over thousands of clusters); the fix is several workgroups per instance.
2. **One mesh buffer instead of four.** Vertices, cluster vertices, triangles and records share one read-only storage buffer. Group 2 gains 3 storage bindings, not 5.
3. **Cone culling is a switch** (`ClusterLod::cone_culling`, on by default), not "off when the transform has no rotation": a transform without a rotation field also describes unrotated instances.
4. **Stats move to M3.** `CullStats`' clusters and triangles per view come with the other views. M2's draw arguments carry the triangle count, which the tests read.
5. **The draw list has a capacity.** By default it holds every cluster of every instance, up to 4M entries. Clusters past it are counted and not drawn.
6. **Representative material forms.** The design validates "every material in the repository's examples". Those embed WGSL in Rust strings with placeholders, so the tests validate the engine's materials and every form the examples and the film use.

## Global Constraints

- wgpu 24 / WebGPU only: no mesh shaders, no multi-draw-indirect, no 64-bit atomics, 4 bind groups, indirect `first_instance` always 0, vertex-stage storage read-only, at most 8 storage buffers per stage.
- Depth is `[0, 1]`, cleared to 1.0. Front faces are counter-clockwise; materials cull back faces by default.
- `queue.write_buffer` lands before the next submit: each renderable's parameters get their own buffer, written only when they change; the view is written once a frame.
- Vertex shaders must not read group 3. Group 3 is untouched.
- Every WGSL module is validated by naga in `cargo test -p kansei-core`, and every `#[repr(C)]` uniform struct is checked against its WGSL size.
- GPU tests pass without running when no adapter exists (`eprintln!("no GPU adapter: skipped"); return;`).
- The GPU never sees an infinity: an ∞ parent error is stored as `NO_PARENT` (−1).
- PRs go against `development`, with no AI attribution. Example measurements are in-page A/B (`bench=`), in headless Chrome with its own session, loading `about:blank` after.

## Review Focus

1. **Buffers recreated under the cluster bind groups.** Instance culling's `ensure_views`, growing matrix buffers, or a growing draw list: bind groups are rebuilt and bundles re-recorded, never a draw from a stale buffer. Pinned by Task 5's `grown_instances_rebind_the_clusters`.
2. **The level window skipping a level the cut needs** (rounding at the boundary, stalled clusters whose parent is several levels up, scaled instances): the GPU list equals the CPU reference except for clusters within rounding of a decision. Pinned by Task 1's `the_level_window_keeps_every_drawn_cluster` and Task 3's equality tests with scaled, rotated and mirrored instances.
3. **Materials the generated stage can't feed** (builtin inputs, unprovided locations, mismatched types, comments and trailing commas): either a valid module, or an `Err` naming the reason while the renderable keeps the ordinary path. Pinned by Task 4.
4. **Mirrored transforms (negative determinant):** cone culling must not remove visible clusters. Pinned by Task 3's `matrix_culled_and_single_instances_cut_as_the_cpu_does`.
5. **Zero visible instances, capacity overflow, and last frame's list:** nothing stale drawn, never past the list. Pinned by Task 3's `nothing_visible_draws_nothing_and_the_capacity_holds`.

## File Structure

- Modify `rust/kansei-core/src/clusters/mod.rs`: `LevelBounds`, `ClusterMesh::levels`, `projected_error_at`, `WINDOW_SLACK`; `mod gpu; mod vertex_stage;`, re-exports.
- Create `rust/kansei-core/src/clusters/gpu.rs`: the packing (`ClusterMesh::gpu_words`), `InstanceTransform`, `ClusterLod`, `ClusterGpu` (per renderable), `ClusterCulling` (pipelines and view), GPU structs.
- Create `rust/kansei-core/src/clusters/vertex_stage.rs`: `cluster_vertex_stage(code, instances) -> Result<String, String>`.
- Create `rust/kansei-core/src/shaders/cluster_mesh.wgsl`: accessors of the packed mesh, included by the cull shader and the generated stage.
- Create `rust/kansei-core/src/shaders/cluster_cull.wgsl`: `prepare`, `cull`, `finish`.
- Create `rust/kansei-core/src/clusters/gpu_tests.rs`: the packing, naga, GPU fetch, GPU cull against the CPU reference, generated stages, and renderer images.
- Modify `rust/kansei-core/src/clusters/tests.rs`: the level window's tests; `pub(super)` on `rock`, `eyes`, `view`.
- Modify `rust/kansei-core/src/materials/material.rs`: `get_cluster_pipeline`, `cluster_pipeline`, and a shared `create_pipeline`.
- Modify `rust/kansei-core/src/renderers/shared_layouts.rs`: `cluster_mesh_bgl`.
- Modify `rust/kansei-core/src/objects/renderable.rs`: `clusters: Option<ClusterLod>`.
- Modify `rust/kansei-core/src/culling/mod.rs`: re-export `MainView` to the crate.
- Modify `rust/kansei-core/src/renderers/renderer.rs`:
  - prepare cluster pipelines and add `run_cluster_culling`;
  - cluster draws in bundles and in the dynamic draws;
  - no occlusion phases for cluster renderables;
  - `set_cluster_error_threshold`;
  - `render_scene_to_gbuffer` becomes `pub(crate)`.
- Create `rust/kansei-wasm/examples/cluster-lod/` (`Cargo.toml`, `src/lib.rs`, `www/index.html`), and add it to `rust/Cargo.toml`'s `exclude`.
- Modify `docs/plans/2026-09-28-cluster-lod-design.md` (M2 as built) and `AGENTS.md` (one line).

---

### Task 1: The level window (CPU)

**Files:**
- Modify: `rust/kansei-core/src/clusters/mod.rs`
- Test: `rust/kansei-core/src/clusters/tests.rs`

**Interfaces:**
- Consumes: M1's `ClusterMesh`, `Cluster { error, lod_bounds, parent_error, parent_bounds, level }`, `LodView`, `projected_error`.
- Produces:
  - `pub struct LevelBounds { pub first: u32, pub count: u32, pub min_error: f32, pub max_parent_error: f32, pub near_reach: f32, pub far_reach: f32 }`
  - `LevelBounds::may_draw(&self, view: &LodView) -> bool`
  - `ClusterMesh::levels(&self) -> Vec<LevelBounds>`
  - `pub fn projected_error_at(error: f32, distance: f32, view: &LodView) -> f32`
  - `pub const WINDOW_SLACK: f32 = 1e-3`

- [ ] **Step 1: Write the failing tests** (append to `clusters/tests.rs`; mark `rock`, `eyes` and `view` `pub(super)` for Task 2's tests)

```rust
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cd rust && cargo test -p kansei-core clusters::tests::level 2>&1 | tail -5`
Expected: FAIL to compile: `no method named levels found for struct ClusterMesh`.

- [ ] **Step 3: Implement** (in `clusters/mod.rs`; `projected_error` now calls `projected_error_at`)

```rust
/// `error` seen from `distance` away (clamped to the view's `near`), in pixels.
pub fn projected_error_at(error: f32, distance: f32, view: &LodView) -> f32 {
    if error == 0.0 {
        return 0.0;
    }
    if !error.is_finite() {
        return f32::INFINITY;
    }
    error / distance.max(view.near) * view.pixels_per_radian
}

/// `error` (metres) seen from the view as pixels: over the distance to the nearest point of
/// `sphere`. A parent's sphere contains its children's and its error is at least theirs, so its
/// projected error is at least theirs from any eye.
pub fn projected_error(error: f32, sphere: Sphere, view: &LodView) -> f32 {
    projected_error_at(error, sphere.center.distance(view.eye) - sphere.radius, view)
}

/// A relative margin `LevelBounds::may_draw` gives the budget, so rounding (on the GPU, with
/// transformed spheres) never skips a level the cut rule would draw from.
pub const WINDOW_SLACK: f32 = 1e-3;

/// What one build round's clusters (`Cluster::level`) span, to skip a whole level: `may_draw`
/// is false only when none of them can pass the cut rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelBounds {
    /// Its clusters: `first..first + count` of `ClusterMesh::clusters`.
    pub first: u32,
    pub count: u32,
    /// The smallest of its clusters' errors.
    pub min_error: f32,
    /// The largest of their parents' errors (∞ when one has none).
    pub max_parent_error: f32,
    /// The farthest a cluster's LOD sphere's nearest point lies from the mesh's origin, beyond
    /// the eye's own distance: the largest `|lod_bounds.center| - lod_bounds.radius`.
    pub near_reach: f32,
    /// The farthest a parent's sphere reaches from the origin: the largest
    /// `|parent_bounds.center| + parent_bounds.radius` (of the finite parents).
    pub far_reach: f32,
}

impl LevelBounds {
    /// Whether some cluster of the level may pass the cut rule for `view`. With `d` the eye's
    /// distance to the mesh's origin, each cluster's own sphere is at most `d + near_reach` away
    /// (so its error projects to at least `min_error` from there) and each parent's at least
    /// `d - far_reach` (so its error projects to at most `max_parent_error` from there).
    pub fn may_draw(&self, view: &LodView) -> bool {
        let d = view.eye.length();
        let fine_enough = projected_error_at(self.min_error, d + self.near_reach, view) <= view.threshold * (1.0 + WINDOW_SLACK);
        let parent_over = !self.max_parent_error.is_finite() || projected_error_at(self.max_parent_error, d - self.far_reach, view) > view.threshold * (1.0 - WINDOW_SLACK);
        fine_enough && parent_over
    }
}

impl ClusterMesh {
    /// Each build round's clusters and what they span (clusters are stored by round).
    pub fn levels(&self) -> Vec<LevelBounds> {
        let mut levels: Vec<LevelBounds> = Vec::new();
        for (i, c) in self.clusters.iter().enumerate() {
            while levels.len() <= c.level as usize {
                levels.push(LevelBounds { first: i as u32, count: 0, min_error: f32::INFINITY, max_parent_error: 0.0, near_reach: f32::NEG_INFINITY, far_reach: f32::NEG_INFINITY });
            }
            let level = &mut levels[c.level as usize];
            debug_assert_eq!(level.first + level.count, i as u32, "clusters are stored by level");
            level.count += 1;
            level.min_error = level.min_error.min(c.error);
            level.max_parent_error = level.max_parent_error.max(c.parent_error);
            level.near_reach = level.near_reach.max(c.lod_bounds.center.length() - c.lod_bounds.radius);
            if c.parent_error.is_finite() {
                level.far_reach = level.far_reach.max(c.parent_bounds.center.length() + c.parent_bounds.radius);
            }
        }
        levels
    }
}
```

- [ ] **Step 4: Run the tests to see them pass, then the module's suite**

Run: `cd rust && cargo test -p kansei-core clusters:: 2>&1 | tail -5`
Expected: PASS: M1's 13 tests and these 3 (one ignored benchmark).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/clusters/mod.rs rust/kansei-core/src/clusters/tests.rs
git commit -m "feat(clusters): the level window: which levels a view's cut can reach"
```

---

### Task 2: The mesh on the GPU, and fetching a cluster's vertices

**Files:**
- Create: `rust/kansei-core/src/clusters/gpu.rs`, `rust/kansei-core/src/shaders/cluster_mesh.wgsl`, `rust/kansei-core/src/clusters/gpu_tests.rs`
- Modify: `rust/kansei-core/src/clusters/mod.rs` (`mod gpu;`, `#[cfg(test)] mod gpu_tests;`)

**Interfaces:**
- Consumes: `ClusterMesh::levels`, `LevelBounds` (Task 1).
- Produces:
  - `ClusterMesh::gpu_words(&self) -> Vec<u32>` and `ClusterMesh::max_triangles(&self) -> u32`
  - `pub(crate) const HEADER_WORDS = 8, VERTEX_WORDS = 9, CLUSTER_WORDS = 28, LEVEL_WORDS = 8; NO_PARENT: f32 = -1.0`
  - `pub(crate) const CLUSTER_MESH_WGSL: &str`
  - WGSL, over a binding named `kansei_cluster_mesh: array<u32>`: `kansei_cluster_word(cluster, word) -> u32`, `kansei_cluster_f32`, `kansei_cluster_vec4`, `kansei_cluster_vertex(cluster, vertex) -> u32`, `kansei_vertex_f32(vertex, word) -> f32`, `kansei_level_word(level, word) -> u32`
  - Test helpers in `gpu_tests.rs`: `validate`, `struct_size`, `device`, `read_words`

The layout (header words): `[vertices, cluster_vertices, triangles, clusters, levels, cluster count, level count, max_triangles]`, the first five being the word offsets of each section.
- **Cluster record (28 words):**
  - 0 vertex_offset, 1 triangle_offset, 2 triangle_count, 3 level
  - 4-7 bounds (xyz, r)
  - 8-10 cone apex, 11 cone cutoff
  - 12-14 cone axis, 15 error
  - 16-19 lod_bounds
  - 20-23 parent_bounds
  - 24 parent_error (`NO_PARENT` for ∞), 25-27 zero
- **Level record (8 words):** first, count, min_error, max_parent_error (`NO_PARENT` for ∞), near_reach, far_reach, 0, 0.

- [ ] **Step 1: Write the failing tests** (`clusters/gpu_tests.rs`)

```rust
use super::gpu::*;
use super::tests::rock;
use super::*;

pub(super) fn validate(name: &str, code: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    module
}

pub(super) fn struct_size(module: &naga::Module, name: &str) -> usize {
    module
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no struct {name}"))
}

/// A device, or None without an adapter.
pub(super) fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    pollster::block_on(adapter.request_device(&Default::default(), None)).ok()
}

pub(super) fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    words
}

#[test]
fn the_gpu_words_hold_the_clusters_and_levels() {
    let mesh = ClusterMesh::build(&rock(3, false), &ClusterOptions::default());
    let words = mesh.gpu_words();
    let section = |k: usize| words[k] as usize;
    let levels = mesh.levels();
    assert_eq!((section(5), section(6), words[7]), (mesh.clusters.len(), levels.len(), mesh.max_triangles()));
    assert_eq!(words.len(), section(4) + levels.len() * LEVEL_WORDS);
    assert_eq!(f32::from_bits(words[section(0) + 5 * VERTEX_WORDS + 4]), mesh.vertices[5].normal[0]);
    for (i, c) in mesh.clusters.iter().enumerate() {
        let record = &words[section(3) + i * CLUSTER_WORDS..][..CLUSTER_WORDS];
        let decoded: Vec<[u32; 3]> = (0..record[2])
            .map(|t| {
                let packed = words[section(2) + (record[1] + t) as usize];
                [0, 1, 2].map(|k| words[section(1) + record[0] as usize + ((packed >> (8 * k)) & 0xff) as usize])
            })
            .collect();
        assert_eq!(decoded, mesh.triangles(i).collect::<Vec<_>>(), "cluster {i}");
        assert_eq!(f32::from_bits(record[15]), c.error);
        assert_eq!(f32::from_bits(record[24]), if c.parent_error.is_finite() { c.parent_error } else { NO_PARENT });
    }
    let root = &words[section(4) + (levels.len() - 1) * LEVEL_WORDS..];
    assert_eq!((root[0], root[1], f32::from_bits(root[3])), (levels.last().unwrap().first, levels.last().unwrap().count, NO_PARENT));
    assert!(words.iter().all(|&w| f32::from_bits(w).is_finite() || w == u32::MAX || (w >> 23) & 0xff != 0xff), "no infinity or NaN reaches the GPU");
}

const FETCH_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(0) @binding(1) var<storage, read_write> fetched: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let per_cluster = 3u * kansei_cluster_mesh[7];
    if (id.x >= per_cluster || id.y >= kansei_cluster_mesh[5]) {
        return;
    }
    let vertex = kansei_cluster_vertex(id.y, id.x);
    fetched[id.y * per_cluster + id.x] = vec2<u32>(vertex, bitcast<u32>(kansei_vertex_f32(vertex, 1u)));
}
"#;

#[test]
fn the_gpu_fetches_every_clusters_vertices() {
    let code = format!("{FETCH_WGSL}\n{CLUSTER_MESH_WGSL}");
    validate("fetch", &code);
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    use wgpu::util::DeviceExt;
    let mesh = ClusterMesh::build(&rock(3, true), &ClusterOptions::default());
    let per_cluster = 3 * mesh.max_triangles() as usize;
    let words = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&mesh.gpu_words()), usage: wgpu::BufferUsages::STORAGE });
    let fetched = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (mesh.clusters.len() * per_cluster * 8) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: words.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: fetched.as_entire_binding() }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(per_cluster.div_ceil(64) as u32, mesh.clusters.len() as u32, 1);
    }
    queue.submit(Some(encoder.finish()));
    let fetched = read_words(&device, &queue, &fetched);
    for (i, c) in mesh.clusters.iter().enumerate() {
        let triangles: Vec<[u32; 3]> = mesh.triangles(i).collect();
        for k in 0..per_cluster {
            // past its triangles, the first corner: triangles of no area
            let expected = if k / 3 < c.triangle_count as usize { triangles[k / 3][k % 3] } else { triangles[0][0] };
            let at = 2 * (i * per_cluster + k);
            assert_eq!(fetched[at], expected, "cluster {i}, vertex {k}");
            assert_eq!(f32::from_bits(fetched[at + 1]), mesh.vertices[expected as usize].position[1]);
        }
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core clusters::gpu_tests 2>&1 | tail -5`
Expected: FAIL to compile: `no method named gpu_words` / `CLUSTER_MESH_WGSL` not found.

- [ ] **Step 3: Implement `shaders/cluster_mesh.wgsl`**

```wgsl
// A cluster mesh's words (clusters/gpu.rs, ClusterMesh::gpu_words) in `kansei_cluster_mesh`,
// which the including module binds. Header: where the vertices, the clusters' vertices, their
// triangles, the cluster records and the level records start (words), the cluster and level
// counts, and the triangles every cluster is drawn as.

const KANSEI_CLUSTER_WORDS: u32 = 28u;
const KANSEI_LEVEL_WORDS: u32 = 8u;
const KANSEI_VERTEX_WORDS: u32 = 9u;

fn kansei_cluster_word(cluster: u32, word: u32) -> u32 {
    return kansei_cluster_mesh[kansei_cluster_mesh[3] + cluster * KANSEI_CLUSTER_WORDS + word];
}

fn kansei_cluster_f32(cluster: u32, word: u32) -> f32 {
    return bitcast<f32>(kansei_cluster_word(cluster, word));
}

fn kansei_cluster_vec4(cluster: u32, word: u32) -> vec4<f32> {
    return vec4<f32>(kansei_cluster_f32(cluster, word), kansei_cluster_f32(cluster, word + 1u), kansei_cluster_f32(cluster, word + 2u), kansei_cluster_f32(cluster, word + 3u));
}

// The mesh vertex drawn as vertex `vertex` of `cluster` (3 per triangle). Past its triangles,
// the first triangle's first corner: the padding up to the draw's size has no area.
fn kansei_cluster_vertex(cluster: u32, vertex: u32) -> u32 {
    var triangle = vertex / 3u;
    var corner = vertex % 3u;
    if (triangle >= kansei_cluster_word(cluster, 2u)) {
        triangle = 0u;
        corner = 0u;
    }
    let packed = kansei_cluster_mesh[kansei_cluster_mesh[2] + kansei_cluster_word(cluster, 1u) + triangle];
    let local = (packed >> (corner * 8u)) & 0xffu;
    return kansei_cluster_mesh[kansei_cluster_mesh[1] + kansei_cluster_word(cluster, 0u) + local];
}

// Word `word` of a mesh vertex: position 0-3, normal 4-6, uv 7-8.
fn kansei_vertex_f32(vertex: u32, word: u32) -> f32 {
    return bitcast<f32>(kansei_cluster_mesh[kansei_cluster_mesh[0] + vertex * KANSEI_VERTEX_WORDS + word]);
}

fn kansei_level_word(level: u32, word: u32) -> u32 {
    return kansei_cluster_mesh[kansei_cluster_mesh[4] + level * KANSEI_LEVEL_WORDS + word];
}
```

- [ ] **Step 4: Implement the packing** (`clusters/gpu.rs`; in `mod.rs`: `mod gpu;` and `#[cfg(test)] mod gpu_tests;`)

```rust
//! A `ClusterMesh` on the GPU, and the camera's per-frame cut of it.

use super::{ClusterMesh, Sphere};
use crate::geometries::Vertex;

/// The packed mesh's layout (`ClusterMesh::gpu_words`, read by cluster_mesh.wgsl).
pub(crate) const HEADER_WORDS: usize = 8;
pub(crate) const VERTEX_WORDS: usize = 9;
pub(crate) const CLUSTER_WORDS: usize = 28;
pub(crate) const LEVEL_WORDS: usize = 8;
/// The error of a missing parent (∞): shaders never see infinities.
pub(crate) const NO_PARENT: f32 = -1.0;

pub(crate) const CLUSTER_MESH_WGSL: &str = include_str!("../shaders/cluster_mesh.wgsl");

const _: () = assert!(std::mem::size_of::<Vertex>() == VERTEX_WORDS * 4);

impl ClusterMesh {
    /// The most triangles in one cluster: what every cluster is drawn as.
    pub fn max_triangles(&self) -> u32 {
        self.clusters.iter().map(|c| c.triangle_count).max().unwrap_or(0)
    }

    /// The mesh as the GPU reads it, in one buffer. A header of where each section starts (in
    /// words), the cluster and level counts, and `max_triangles`; then the vertices (`Vertex` as
    /// is), the clusters' vertices, their triangles (3 local indices in a word's low 3 bytes),
    /// the cluster records and the level records (see cluster_mesh.wgsl).
    pub fn gpu_words(&self) -> Vec<u32> {
        let levels = self.levels();
        let vertices = HEADER_WORDS;
        let cluster_vertices = vertices + self.vertices.len() * VERTEX_WORDS;
        let triangles = cluster_vertices + self.cluster_vertices.len();
        let clusters = triangles + self.cluster_triangles.len() / 3;
        let level_records = clusters + self.clusters.len() * CLUSTER_WORDS;
        let mut words = Vec::with_capacity(level_records + levels.len() * LEVEL_WORDS);
        words.extend([vertices, cluster_vertices, triangles, clusters, level_records, self.clusters.len(), levels.len(), self.max_triangles() as usize].map(|w| w as u32));
        words.extend_from_slice(bytemuck::cast_slice(&self.vertices));
        words.extend_from_slice(&self.cluster_vertices);
        words.extend(self.cluster_triangles.chunks(3).map(|t| t[0] as u32 | (t[1] as u32) << 8 | (t[2] as u32) << 16));
        let parent = |e: f32| if e.is_finite() { e } else { NO_PARENT };
        let sphere = |s: Sphere| [s.center.x, s.center.y, s.center.z, s.radius].map(f32::to_bits);
        for c in &self.clusters {
            words.extend([c.vertex_offset, c.triangle_offset, c.triangle_count, c.level]);
            words.extend(sphere(c.bounds));
            words.extend([c.cone_apex.x, c.cone_apex.y, c.cone_apex.z, c.cone_cutoff].map(f32::to_bits));
            words.extend([c.cone_axis.x, c.cone_axis.y, c.cone_axis.z, c.error].map(f32::to_bits));
            words.extend(sphere(c.lod_bounds));
            words.extend(sphere(c.parent_bounds));
            words.extend([parent(c.parent_error).to_bits(), 0, 0, 0]);
        }
        let finite = |x: f32| if x.is_finite() { x } else { 0.0 };
        for l in &levels {
            words.extend([l.first, l.count, l.min_error.to_bits(), parent(l.max_parent_error).to_bits(), finite(l.near_reach).to_bits(), finite(l.far_reach).to_bits(), 0, 0]);
        }
        words
    }
}
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cd rust && cargo test -p kansei-core clusters:: 2>&1 | tail -5`
Expected: PASS: 18 tests (1 ignored). Without an adapter, `the_gpu_fetches_every_clusters_vertices` prints "skipped" and passes.

- [ ] **Step 6: Commit**

```bash
git add rust/kansei-core/src/clusters rust/kansei-core/src/shaders/cluster_mesh.wgsl
git commit -m "feat(clusters): the mesh as one GPU buffer, and fetching a cluster's vertices"
```

---

### Task 3: The cull: prepare, cull, finish

**Files:**
- Create: `rust/kansei-core/src/shaders/cluster_cull.wgsl`
- Modify: `rust/kansei-core/src/clusters/gpu.rs`, `rust/kansei-core/src/clusters/mod.rs` (`pub use gpu::InstanceTransform;`)
- Test: `rust/kansei-core/src/clusters/gpu_tests.rs`

**Interfaces:**
- Consumes: `gpu_words`, `CLUSTER_MESH_WGSL` (Task 2); `LevelBounds` semantics and `WINDOW_SLACK` (Task 1); `crate::culling::frustum_planes`.
- Produces:
  - `pub enum InstanceTransform { Matrix { offset: u32 }, Placement { position: u32, scale: Option<u32>, yaw: Option<u32>, rotation: Option<u32> } }`
  - `pub(crate) enum InstanceSource<'a> { None, All { records, count }, Culled { records, first_record, capacity, args, count_word } }`
  - `pub(crate) struct ClusterCullGpu` with `ClusterCullGpu::new(world: glam::Mat4, transform: Option<InstanceTransform>, stride: u32, source: &InstanceSource, capacity: u32, vertex_count: u32, cone_culling: bool) -> Self`
  - `pub(crate) struct ClusterViewGpu` with `ClusterViewGpu::new(view_proj: glam::Mat4, eye: glam::Vec3, pixels_per_radian: f32, near: f32, threshold: f32) -> Self`
  - `pub(crate) struct ClusterCulling`: `new(device)`, `set_view(queue, &ClusterViewGpu)`, `encode(encoder, &[&ClusterGpu])`
  - `pub(crate) struct ClusterGpu`: `new(device, &ClusterMesh)`, `bind(device, queue, &ClusterCulling, InstanceSource, ClusterCullGpu) -> bool`, `args()`, `draws()`, `vertex_count()`, `cluster_count()`
  - `pub(crate) const DRAW_ARGS_BYTES: u64 = 32`, `DEFAULT_MAX_DRAWN: u32 = 1 << 22`, `NO_WORD: u32 = u32::MAX`
  - The draw's words: `[vertex_count, instance_count, first_vertex, first_instance, visible instances, clusters claimed, triangles drawn, 0]`

- [ ] **Step 1: Write the failing tests** (append to `gpu_tests.rs`)

```rust
use std::collections::BTreeSet;

#[test]
fn the_cull_shader_validates_and_its_uniforms_match() {
    let module = validate("cluster_cull", &format!("{CLUSTER_CULL_WGSL}\n{CLUSTER_MESH_WGSL}"));
    assert_eq!(struct_size(&module, "ClusterCull"), std::mem::size_of::<ClusterCullGpu>());
    assert_eq!(struct_size(&module, "ClusterView"), std::mem::size_of::<ClusterViewGpu>());
    assert_eq!(struct_size(&module, "ClusterDraw") as u64, DRAW_ARGS_BYTES);
}

/// A view for the cull tests: a camera at `eye` looking at `target` (60°, square), 512 pixels
/// high, and the budget.
struct TestView {
    view_proj: glam::Mat4,
    eye: glam::Vec3,
    ppr: f32,
    near: f32,
    threshold: f32,
}

impl TestView {
    fn looking(eye: glam::Vec3, target: glam::Vec3, threshold: f32) -> Self {
        let fov = 60f32.to_radians();
        let view_proj = glam::Mat4::perspective_rh(fov, 1.0, 0.1, 1000.0) * glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        Self { view_proj, eye, ppr: 512.0 / (2.0 * (fov / 2.0).tan()), near: 0.1, threshold }
    }

    fn gpu(&self) -> ClusterViewGpu {
        ClusterViewGpu::new(self.view_proj, self.eye, self.ppr, self.near, self.threshold)
    }
}

/// What the cull should draw of `mesh` placed by `model` (a uniform scale, maybe mirrored):
/// M1's `select` in the mesh's space, less the clusters outside the frustum and (with `cone`)
/// those facing away. Also returns the clusters within rounding of any of those decisions.
fn expected(mesh: &ClusterMesh, model: glam::Mat4, v: &TestView, cone: bool) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let m = glam::Mat3::from_mat4(model);
    let scale = m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length());
    let eye = model.inverse().transform_point3(v.eye);
    let lod = LodView { eye, pixels_per_radian: v.ppr, near: v.near / scale, threshold: v.threshold };
    let cone = cone && m.determinant() > 0.0;
    let planes = crate::culling::frustum_planes(v.view_proj);
    let selected: BTreeSet<usize> = mesh.select(&lod).into_iter().collect();
    let close = |p: f32| (p - v.threshold).abs() <= 2e-3 * v.threshold.max(1e-3);
    let (mut drawn, mut ambiguous) = (BTreeSet::new(), BTreeSet::new());
    for (i, c) in mesh.clusters.iter().enumerate() {
        let center = model.transform_point3(c.bounds.center);
        let radius = c.bounds.radius * scale;
        let outside: Vec<f32> = planes.iter().map(|p| p.truncate().dot(center) + p.w + radius).collect();
        let facing = (c.cone_apex - eye).normalize_or_zero().dot(c.cone_axis) - c.cone_cutoff;
        if close(projected_error(c.error, c.lod_bounds, &lod))
            || close(projected_error(c.parent_error, c.parent_bounds, &lod))
            || outside.iter().any(|d| d.abs() < 1e-4 * radius.max(1.0))
            || (cone && facing.abs() < 1e-4)
        {
            ambiguous.insert(i as u32);
        } else if selected.contains(&i) && outside.iter().all(|&d| d >= 0.0) && !(cone && facing >= 0.0) {
            drawn.insert(i as u32);
        }
    }
    (drawn, ambiguous)
}

/// Cull once and read back the draw's words and the drawn (record, cluster) pairs.
fn cull(device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, gpu: &mut ClusterGpu, source: InstanceSource, params: ClusterCullGpu, view: &TestView) -> (Vec<u32>, Vec<(u32, u32)>) {
    gpu.bind(device, queue, culling, source, params);
    culling.set_view(queue, &view.gpu());
    let mut encoder = device.create_command_encoder(&Default::default());
    culling.encode(&mut encoder, &[gpu]);
    queue.submit(Some(encoder.finish()));
    let args = read_words(device, queue, gpu.args());
    let list = read_words(device, queue, gpu.draws());
    let pairs = list.chunks(2).take(args[1] as usize).map(|p| (p[0], p[1])).collect();
    (args, pairs)
}

/// Asserts the GPU drew, of record `record`, the expected clusters (give or take the ambiguous).
fn assert_cut(label: &str, pairs: &[(u32, u32)], record: u32, (drawn, ambiguous): &(BTreeSet<u32>, BTreeSet<u32>)) {
    let gpu: BTreeSet<u32> = pairs.iter().filter(|p| p.0 == record).map(|p| p.1).collect();
    assert_eq!(gpu.len(), pairs.iter().filter(|p| p.0 == record).count(), "{label}: a cluster drawn twice");
    let missing: Vec<_> = drawn.difference(&gpu).collect();
    let extra: Vec<_> = gpu.difference(drawn).filter(|c| !ambiguous.contains(c)).collect();
    assert!(missing.is_empty() && extra.is_empty(), "{label}: missing {missing:?}, extra {extra:?} ({} expected)", drawn.len());
}

fn buffer(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(words), usage: wgpu::BufferUsages::STORAGE })
}

/// Records of 12 floats: position, scale, yaw, pad, rotation (x y z w), pad.
const PLACEMENT: InstanceTransform = InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), rotation: Some(24) };

fn placement_record(position: glam::Vec3, scale: f32, yaw: f32, rotation: glam::Quat) -> [f32; 12] {
    [position.x, position.y, position.z, scale, yaw, 0.0, rotation.x, rotation.y, rotation.z, rotation.w, 0.0, 0.0]
}

fn placement_matrix(r: &[f32; 12]) -> glam::Mat4 {
    glam::Mat4::from_translation(glam::Vec3::new(r[0], r[1], r[2])) * glam::Mat4::from_rotation_y(r[4]) * glam::Mat4::from_quat(glam::Quat::from_xyzw(r[6], r[7], r[8], r[9])) * glam::Mat4::from_scale(glam::Vec3::splat(r[3]))
}

#[test]
fn the_gpu_cut_of_placed_instances_is_the_cpu_cut() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let records = [
        placement_record(glam::Vec3::ZERO, 1.0, 0.0, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(6.0, 0.0, -4.0), 2.5, 1.2, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(-5.0, 1.5, -9.0), 0.5, -2.0, glam::Quat::from_rotation_x(0.7)),
        placement_record(glam::Vec3::new(30.0, 0.0, -60.0), 3.0, 0.3, glam::Quat::IDENTITY),
    ];
    let world = glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(1.5), glam::Quat::from_rotation_z(0.2), glam::Vec3::new(1.0, 0.0, 0.0));
    let record_buffer = buffer(&device, bytemuck::cast_slice(&records.concat()));
    let culling = ClusterCulling::new(&device);
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let source = InstanceSource::All { records: &record_buffer, count: records.len() as u32 };
    let params = ClusterCullGpu::new(world, Some(PLACEMENT), 48, &source, 4 * mesh.clusters.len() as u32, gpu.vertex_count(), true);
    let mut totals = Vec::new();
    for (eye, target) in [(glam::Vec3::new(0.0, 2.0, 8.0), glam::Vec3::ZERO), (glam::Vec3::new(3.0, 0.5, 1.5), glam::Vec3::new(1.0, 0.0, 0.0)), (glam::Vec3::new(-20.0, 10.0, 30.0), glam::Vec3::new(10.0, 0.0, -30.0))] {
        for threshold in [0.0, 0.5, 1.0, 4.0] {
            let view = TestView::looking(eye, target, threshold);
            let (args, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
            assert_eq!((args[0], args[4]), (gpu.vertex_count(), records.len() as u32));
            for (k, r) in records.iter().enumerate() {
                let label = format!("eye {eye}, {threshold} px, instance {k}");
                assert_cut(&label, &pairs, k as u32, &expected(&mesh, world * placement_matrix(r), &view, true));
            }
            let triangles: u32 = pairs.iter().map(|p| mesh.clusters[p.1 as usize].triangle_count).sum();
            assert_eq!(args[6], triangles, "the triangles drawn");
            totals.push(triangles);
        }
    }
    // a coarser budget draws fewer triangles
    assert!(totals[3] < totals[0] && totals[0] > 0, "{totals:?}");
}

#[test]
fn matrix_culled_and_single_instances_cut_as_the_cpu_does() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let culling = ClusterCulling::new(&device);
    let view = TestView::looking(glam::Vec3::new(1.0, 1.5, 6.0), glam::Vec3::new(0.0, 0.0, -2.0), 1.0);
    let world = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.5, 0.0));

    // matrices, one of them mirrored (no cone test there: its winding is turned over), behind
    // two records the culled view doesn't count, with its count in word 5 of the instance draws
    let matrices = [
        glam::Mat4::IDENTITY,
        glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(2.0), glam::Quat::from_rotation_y(0.4), glam::Vec3::new(3.0, 0.0, -4.0)),
        glam::Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, -2.0)) * glam::Mat4::from_scale(glam::Vec3::new(-1.2, 1.2, 1.2)),
    ];
    let mut words: Vec<f32> = vec![9.0; 32];
    for m in &matrices {
        words.extend(m.to_cols_array());
    }
    let records = buffer(&device, bytemuck::cast_slice(&words));
    let instance_args = buffer(&device, &[0, 0, 0, 0, 0, 3, 0, 0]);
    let source = InstanceSource::Culled { records: &records, first_record: 2, capacity: 3, args: &instance_args, count_word: 5 };
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let params = ClusterCullGpu::new(world, Some(InstanceTransform::Matrix { offset: 0 }), 64, &source, 3 * mesh.clusters.len() as u32, gpu.vertex_count(), true);
    let (args, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
    assert_eq!(args[4], 3);
    assert!(pairs.iter().all(|p| (2..5).contains(&p.0)), "records from the view's first");
    for (k, m) in matrices.iter().enumerate() {
        assert_cut(&format!("matrix {k}"), &pairs, 2 + k as u32, &expected(&mesh, world * *m, &view, true));
    }

    // no instances: the mesh once, where the renderable is
    let mut single = ClusterGpu::new(&device, &mesh);
    let params = ClusterCullGpu::new(world, None, 0, &InstanceSource::None, mesh.clusters.len() as u32, single.vertex_count(), true);
    let (args, pairs) = cull(&device, &queue, &culling, &mut single, InstanceSource::None, params, &view);
    assert_eq!(args[4], 1);
    assert_cut("single", &pairs, 0, &expected(&mesh, world, &view, true));
}

#[test]
fn nothing_visible_draws_nothing_and_the_capacity_holds() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let culling = ClusterCulling::new(&device);
    let view = TestView::looking(glam::Vec3::new(0.0, 0.0, 1.5), glam::Vec3::ZERO, 0.0);
    let records = buffer(&device, bytemuck::cast_slice(&placement_record(glam::Vec3::ZERO, 1.0, 0.0, glam::Quat::IDENTITY)));
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let counted = |count: u32| buffer(&device, &[0, count, 0, 0, 0, 0, 0, 0]);

    // a full list: only the capacity drawn, the rest counted
    let one = counted(1);
    let source = InstanceSource::Culled { records: &records, first_record: 0, capacity: 1, args: &one, count_word: 1 };
    let params = ClusterCullGpu::new(glam::Mat4::IDENTITY, Some(PLACEMENT), 48, &source, 5, gpu.vertex_count(), true);
    let (args, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
    assert_eq!(args[1], 5, "drawn: the capacity");
    assert!(args[5] > 5, "claimed: {}", args[5]);
    assert!(pairs.iter().all(|p| p.0 == 0 && (p.1 as usize) < mesh.clusters.len()));

    // then no instance visible: nothing drawn, nothing left from before
    let none = counted(0);
    let source = InstanceSource::Culled { records: &records, first_record: 0, capacity: 1, args: &none, count_word: 1 };
    let (args, _) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
    assert_eq!((args[1], args[4], args[5], args[6]), (0, 0, 0, 0));
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core clusters::gpu_tests 2>&1 | tail -5`
Expected: FAIL to compile: `ClusterCulling`, `ClusterGpu`, `InstanceSource` not found.

- [ ] **Step 3: Implement `shaders/cluster_cull.wgsl`**

```wgsl
// Cluster LOD: a renderable's cut for one view (clusters/gpu.rs). `prepare` resets the draw and
// sizes `cull`'s dispatch from the visible instances. `cull` runs a workgroup per visible
// instance over the clusters of the levels its cut can reach (clusters::LevelBounds), testing
// the cut rule, the frustum and the backface cone, and appends the drawn ones to the draw list.
// `finish` draws at most the list's capacity. Included with cluster_mesh.wgsl.

struct ClusterCull {
    world: mat4x4<f32>,
    // 0: none (the renderable's transform), 1: placement, 2: a matrix
    kind: u32,
    position_word: u32,
    scale_word: u32,
    yaw_word: u32,
    rotation_word: u32,
    stride_words: u32,
    first_record: u32,
    // the visible instances without a count word
    instance_count: u32,
    count_word: u32,
    capacity: u32,
    vertex_count: u32,
    flags: u32,
}

struct ClusterView {
    planes: array<vec4<f32>, 6>,
    eye: vec3<f32>,
    pixels_per_radian: f32,
    near: f32,
    threshold: f32,
    pad: vec2<f32>,
}

// DrawIndirect's four words, then the visible instances, the clusters claimed and the triangles
// drawn
struct ClusterDraw {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
    visible: u32,
    claimed: atomic<u32>,
    triangles: atomic<u32>,
    pad: u32,
}

const NONE: u32 = 0xffffffffu;
const KIND_PLACEMENT: u32 = 1u;
const KIND_MATRIX: u32 = 2u;
const FLAG_CONE: u32 = 1u;
const MAX_GROUPS: u32 = 65535u;
// clusters::WINDOW_SLACK
const WINDOW_SLACK: f32 = 1e-3;

@group(0) @binding(0) var<uniform> params: ClusterCull;
@group(0) @binding(1) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(0) @binding(2) var<storage, read> records: array<u32>;
@group(0) @binding(3) var<storage, read_write> draws: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> draw: ClusterDraw;
@group(1) @binding(0) var<uniform> view: ClusterView;

// prepare and finish
@group(0) @binding(10) var<uniform> prepare_params: ClusterCull;
@group(0) @binding(11) var<storage, read> instance_args: array<u32>;
@group(0) @binding(12) var<storage, read_write> prepare_draw: array<u32, 8>;
@group(0) @binding(13) var<storage, read_write> dispatch: array<u32, 4>;

@compute @workgroup_size(1)
fn prepare() {
    var visible = prepare_params.instance_count;
    if (prepare_params.count_word != NONE) {
        visible = instance_args[prepare_params.count_word];
    }
    prepare_draw = array<u32, 8>(prepare_params.vertex_count, 0u, 0u, 0u, visible, 0u, 0u, 0u);
    let x = min(visible, MAX_GROUPS);
    dispatch = array<u32, 4>(x, select(0u, (visible + x - 1u) / x, x > 0u), 1u, 0u);
}

@compute @workgroup_size(1)
fn finish() {
    prepare_draw[1] = min(prepare_draw[5], prepare_params.capacity);
}

fn record_f32(record: u32, word: u32) -> f32 {
    return bitcast<f32>(records[record * params.stride_words + word]);
}

fn record_vec3(record: u32, word: u32) -> vec3<f32> {
    return vec3<f32>(record_f32(record, word), record_f32(record, word + 1u), record_f32(record, word + 2u));
}

fn record_vec4(record: u32, word: u32) -> vec4<f32> {
    return vec4<f32>(record_vec3(record, word), record_f32(record, word + 3u));
}

// a unit quaternion (x y z w) as a rotation (glam's Mat3::from_quat)
fn rotation(q: vec4<f32>) -> mat3x3<f32> {
    let x2 = q.x + q.x;
    let y2 = q.y + q.y;
    let z2 = q.z + q.z;
    let xx = q.x * x2;
    let xy = q.x * y2;
    let xz = q.x * z2;
    let yy = q.y * y2;
    let yz = q.y * z2;
    let zz = q.z * z2;
    let wx = q.w * x2;
    let wy = q.w * y2;
    let wz = q.w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

// where the record puts the mesh, in the renderable's space
fn placement(record: u32) -> mat4x4<f32> {
    if (params.kind == KIND_MATRIX) {
        let w = params.position_word;
        return mat4x4<f32>(record_vec4(record, w), record_vec4(record, w + 4u), record_vec4(record, w + 8u), record_vec4(record, w + 12u));
    }
    var m = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    if (params.kind != KIND_PLACEMENT) {
        return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
    if (params.rotation_word != NONE) {
        m = rotation(record_vec4(record, params.rotation_word));
    }
    if (params.yaw_word != NONE) {
        let a = record_f32(record, params.yaw_word);
        m = mat3x3<f32>(vec3<f32>(cos(a), 0.0, -sin(a)), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(sin(a), 0.0, cos(a))) * m;
    }
    if (params.scale_word != NONE) {
        m = m * record_f32(record, params.scale_word);
    }
    return mat4x4<f32>(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0), vec4<f32>(record_vec3(record, params.position_word), 1.0));
}

// clusters::projected_error_at: `error` (world units) seen `distance` away, in pixels
fn projected_at(error: f32, distance: f32) -> f32 {
    return error / max(distance, view.near) * view.pixels_per_radian;
}

// the error of a mesh sphere (xyz, radius) placed by `model`, whose largest scale is `scale`
fn projected(error: f32, sphere: vec4<f32>, model: mat4x4<f32>, scale: f32) -> f32 {
    let center = (model * vec4<f32>(sphere.xyz, 1.0)).xyz;
    return projected_at(error * scale, distance(view.eye, center) - sphere.w * scale);
}

// clusters::LevelBounds::may_draw, from the eye's distance to the mesh's origin
fn level_may_draw(level: u32, origin_distance: f32, scale: f32) -> bool {
    let min_error = bitcast<f32>(kansei_level_word(level, 2u));
    let max_parent_error = bitcast<f32>(kansei_level_word(level, 3u));
    let near_reach = bitcast<f32>(kansei_level_word(level, 4u));
    let far_reach = bitcast<f32>(kansei_level_word(level, 5u));
    let fine_enough = projected_at(min_error * scale, origin_distance + near_reach * scale) <= view.threshold * (1.0 + WINDOW_SLACK);
    let parent_over = max_parent_error < 0.0 || projected_at(max_parent_error * scale, origin_distance - far_reach * scale) > view.threshold * (1.0 - WINDOW_SLACK);
    return fine_enough && parent_over;
}

fn cluster_drawn(c: u32, model: mat4x4<f32>, scale: f32, eye_mesh: vec3<f32>, cone: bool) -> bool {
    // the cut rule: fine enough, and its parent not
    if (projected(kansei_cluster_f32(c, 15u), kansei_cluster_vec4(c, 16u), model, scale) > view.threshold) {
        return false;
    }
    let parent_error = kansei_cluster_f32(c, 24u);
    if (parent_error >= 0.0 && projected(parent_error, kansei_cluster_vec4(c, 20u), model, scale) <= view.threshold) {
        return false;
    }
    // the frustum
    let bounds = kansei_cluster_vec4(c, 4u);
    let center = (model * vec4<f32>(bounds.xyz, 1.0)).xyz;
    for (var p = 0u; p < 6u; p++) {
        if (dot(view.planes[p].xyz, center) + view.planes[p].w < -bounds.w * scale) {
            return false;
        }
    }
    // every triangle facing away (clusters::Cluster::backfacing, in the mesh's space)
    if (cone) {
        let apex = kansei_cluster_vec4(c, 8u);
        if (dot(normalize(apex.xyz - eye_mesh), kansei_cluster_vec4(c, 12u).xyz) >= apex.w) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn cull(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let slot = group.y * groups.x + group.x;
    if (slot >= draw.visible) {
        return;
    }
    let record = params.first_record + slot;
    let model = params.world * placement(record);
    let m = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    let scale = max(length(m[0]), max(length(m[1]), length(m[2])));
    let det = determinant(m);
    // a mirroring transform turns the winding over: no cone test
    let cone = (params.flags & FLAG_CONE) != 0u && det > 0.0;
    var eye_mesh = vec3<f32>(0.0);
    if (cone) {
        let inverse = transpose(mat3x3<f32>(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]))) * (1.0 / det);
        eye_mesh = inverse * (view.eye - model[3].xyz);
    }
    let origin_distance = distance(view.eye, model[3].xyz);
    let levels = kansei_cluster_mesh[6];
    for (var level = 0u; level < levels; level++) {
        if (!level_may_draw(level, origin_distance, scale)) {
            continue;
        }
        let first = kansei_level_word(level, 0u);
        let count = kansei_level_word(level, 1u);
        for (var i = lane; i < count; i += 64u) {
            let c = first + i;
            if (cluster_drawn(c, model, scale, eye_mesh, cone)) {
                let at = atomicAdd(&draw.claimed, 1u);
                if (at < params.capacity) {
                    draws[at] = vec2<u32>(record, c);
                    atomicAdd(&draw.triangles, kansei_cluster_word(c, 2u));
                }
            }
        }
    }
}
```

- [ ] **Step 4: Implement the Rust side** (append to `clusters/gpu.rs`; `clusters/mod.rs` gets `pub use gpu::InstanceTransform;` and `pub(crate) use gpu::{ClusterCulling, ClusterGpu, ClusterViewGpu, InstanceSource};`)

```rust
use bytemuck::{Pod, Zeroable};

pub(crate) const CLUSTER_CULL_WGSL: &str = include_str!("../shaders/cluster_cull.wgsl");
pub(crate) const NO_WORD: u32 = u32::MAX;
const KIND_NONE: u32 = 0;
const KIND_PLACEMENT: u32 = 1;
const KIND_MATRIX: u32 = 2;
const FLAG_CONE: u32 = 1;
/// Entries of a draw list by default, at most (32 MB).
pub(crate) const DEFAULT_MAX_DRAWN: u32 = 1 << 22;
/// Bytes of a cluster draw: `DrawIndirect`'s four words, then the visible instances, the
/// clusters claimed (drawn up to the capacity), the triangles drawn, and a pad.
pub(crate) const DRAW_ARGS_BYTES: u64 = 32;

/// Where an instance record places the mesh, as far as the cluster test needs it: spheres,
/// errors and cones follow the instance. It should say what the material's vertex stage does
/// with the record. Offsets are in bytes into the record, multiples of 4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstanceTransform {
    /// A column-major 4x4 matrix of f32.
    Matrix { offset: u32 },
    /// A position (3 x f32), then optionally a uniform scale (f32), a turn about +y in radians
    /// (f32, as `glam::Mat3::from_rotation_y`) and a rotation (a unit quaternion, x y z w):
    /// `position + yaw * rotation * (scale * p)`.
    Placement { position: u32, scale: Option<u32>, yaw: Option<u32>, rotation: Option<u32> },
}

/// Where the cull reads a renderable's instances: none (the mesh once, where the renderable
/// is), every record of a buffer, or a view's compacted records (`InstanceCulling`), as many
/// as word `count_word` of its draws says.
#[derive(Clone, Copy)]
pub(crate) enum InstanceSource<'a> {
    None,
    All { records: &'a wgpu::Buffer, count: u32 },
    Culled { records: &'a wgpu::Buffer, first_record: u32, capacity: u32, args: &'a wgpu::Buffer, count_word: u32 },
}

impl InstanceSource<'_> {
    /// Instances it can hold, which sizes the draw list.
    pub(crate) fn capacity(&self) -> u32 {
        match self {
            Self::None => 1,
            Self::All { count, .. } => *count,
            Self::Culled { capacity, .. } => *capacity,
        }
    }
}

/// `ClusterCull` in cluster_cull.wgsl: a renderable's parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ClusterCullGpu {
    world: [f32; 16],
    kind: u32,
    position_word: u32,
    scale_word: u32,
    yaw_word: u32,
    rotation_word: u32,
    stride_words: u32,
    first_record: u32,
    instance_count: u32,
    count_word: u32,
    capacity: u32,
    vertex_count: u32,
    flags: u32,
}

impl ClusterCullGpu {
    /// A renderable's parameters. Its world matrix; how its records (`stride` bytes each) place
    /// the mesh, and where they come from; the draw list's capacity; the draw's vertices
    /// (3 x the mesh's max triangles); and whether to test the cones.
    pub(crate) fn new(world: glam::Mat4, transform: Option<InstanceTransform>, stride: u32, source: &InstanceSource, capacity: u32, vertex_count: u32, cone_culling: bool) -> Self {
        let word = |offset: u32| offset / 4;
        let (kind, position_word, scale_word, yaw_word, rotation_word) = match (source, transform) {
            (InstanceSource::None, _) | (_, None) => (KIND_NONE, 0, NO_WORD, NO_WORD, NO_WORD),
            (_, Some(InstanceTransform::Matrix { offset })) => (KIND_MATRIX, word(offset), NO_WORD, NO_WORD, NO_WORD),
            (_, Some(InstanceTransform::Placement { position, scale, yaw, rotation })) => (KIND_PLACEMENT, word(position), scale.map_or(NO_WORD, word), yaw.map_or(NO_WORD, word), rotation.map_or(NO_WORD, word)),
        };
        let (first_record, instance_count, count_word) = match *source {
            InstanceSource::None => (0, 1, NO_WORD),
            InstanceSource::All { count, .. } => (0, count, NO_WORD),
            InstanceSource::Culled { first_record, count_word, .. } => (first_record, 0, count_word),
        };
        Self {
            world: world.to_cols_array(),
            kind,
            position_word,
            scale_word,
            yaw_word,
            rotation_word,
            stride_words: stride / 4,
            first_record,
            instance_count,
            count_word,
            capacity,
            vertex_count,
            flags: if cone_culling { FLAG_CONE } else { 0 },
        }
    }
}

/// `ClusterView` in cluster_cull.wgsl: the view every renderable is cut for.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ClusterViewGpu {
    planes: [[f32; 4]; 6],
    eye: [f32; 3],
    pixels_per_radian: f32,
    near: f32,
    threshold: f32,
    _pad: [f32; 2],
}

impl ClusterViewGpu {
    /// A view: its view-projection's frustum, the eye errors are measured from, pixels per
    /// radian, the distance errors are clamped to, and the budget in pixels.
    pub(crate) fn new(view_proj: glam::Mat4, eye: glam::Vec3, pixels_per_radian: f32, near: f32, threshold: f32) -> Self {
        Self { planes: crate::culling::frustum_planes(view_proj).map(|p| p.to_array()), eye: eye.to_array(), pixels_per_radian, near, threshold, _pad: [0.0; 2] }
    }
}

/// The cluster cull's pipelines and its view (one per renderer).
pub(crate) struct ClusterCulling {
    cull_bgl: wgpu::BindGroupLayout,
    prepare_bgl: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    cull: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    view: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
}

impl ClusterCulling {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let uniform = |binding| buffer_entry(binding, wgpu::BufferBindingType::Uniform);
        let storage = |binding, read_only| buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only });
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let cull_bgl = layout("ClusterCulling/Cull", &[uniform(0), storage(1, true), storage(2, true), storage(3, false), storage(4, false)]);
        let view_bgl = layout("ClusterCulling/View", &[uniform(0)]);
        let prepare_bgl = layout("ClusterCulling/Prepare", &[uniform(10), storage(11, true), storage(12, false), storage(13, false)]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ClusterCulling"), source: wgpu::ShaderSource::Wgsl(format!("{CLUSTER_CULL_WGSL}\n{CLUSTER_MESH_WGSL}").into()) });
        let pipeline = |entry: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ClusterCulling"), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&format!("ClusterCulling/{entry}")), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let view = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ClusterCulling/View"), size: std::mem::size_of::<ClusterViewGpu>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ClusterCulling/View"), layout: &view_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view.as_entire_binding() }] });
        Self { prepare: pipeline("prepare", &[&prepare_bgl]), cull: pipeline("cull", &[&cull_bgl, &view_bgl]), finish: pipeline("finish", &[&prepare_bgl]), cull_bgl, prepare_bgl, view, view_bind_group }
    }

    /// The frame's view, shared by every renderable (one write).
    pub(crate) fn set_view(&self, queue: &wgpu::Queue, view: &ClusterViewGpu) {
        queue.write_buffer(&self.view, 0, bytemuck::bytes_of(view));
    }

    /// Cut each of `clusters` (bound with `ClusterGpu::bind`) for the view in one compute pass:
    /// every prepare, then every cull (dispatched indirectly, a workgroup per visible instance),
    /// then every finish. A cull's dispatch buffer is never bound while it is dispatched.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, clusters: &[&ClusterGpu]) {
        let bound: Vec<(&ClusterGpu, &Bound)> = clusters.iter().filter_map(|c| Some((*c, c.bound.as_ref()?))).collect();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Renderer/ClusterCulling"), timestamp_writes: crate::profiling::gpu_pass("Renderer/ClusterCulling").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.prepare);
        for (_, b) in &bound {
            pass.set_bind_group(0, &b.prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        pass.set_pipeline(&self.cull);
        pass.set_bind_group(1, &self.view_bind_group, &[]);
        for (c, b) in &bound {
            pass.set_bind_group(0, &b.cull_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&c.dispatch, 0);
        }
        pass.set_pipeline(&self.finish);
        for (_, b) in &bound {
            pass.set_bind_group(0, &b.prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}

/// A renderable's cluster mesh on the GPU, and a view's cut of it. It holds the packed mesh,
/// the parameters, the draw list and its indirect draw, and the cull's indirect dispatch.
pub(crate) struct ClusterGpu {
    mesh: wgpu::Buffer,
    vertex_count: u32,
    cluster_count: u32,
    params: wgpu::Buffer,
    written: Option<ClusterCullGpu>,
    draws: wgpu::Buffer,
    capacity: u32,
    args: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    /// bound in place of a missing instance buffer or count
    empty: wgpu::Buffer,
    bound: Option<Bound>,
}

/// The cull's bind groups, and the instance buffers they were made with.
struct Bound {
    records: Option<wgpu::Buffer>,
    count: Option<wgpu::Buffer>,
    cull_bind_group: wgpu::BindGroup,
    prepare_bind_group: wgpu::BindGroup,
}

impl ClusterGpu {
    pub(crate) fn new(device: &wgpu::Device, mesh: &ClusterMesh) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        Self {
            mesh: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Clusters/Mesh"), contents: bytemuck::cast_slice(&mesh.gpu_words()), usage: wgpu::BufferUsages::STORAGE }),
            vertex_count: 3 * mesh.max_triangles(),
            cluster_count: mesh.clusters.len() as u32,
            params: buffer("Clusters/Params", std::mem::size_of::<ClusterCullGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            written: None,
            draws: buffer("Clusters/Draws", 8, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            capacity: 0,
            // (COPY_SRC: read back by the tests)
            args: buffer("Clusters/Args", DRAW_ARGS_BYTES, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            dispatch: buffer("Clusters/Dispatch", 16, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE),
            empty: buffer("Clusters/Empty", 32, wgpu::BufferUsages::STORAGE),
            bound: None,
        }
    }

    /// Bind the cull to `source`, with a draw list of at least `params.capacity` entries (it grows,
    /// never shrinks), and write the parameters if they changed. True when the draw list was
    /// remade: draws recorded with the old one are stale.
    pub(crate) fn bind(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, source: InstanceSource, params: ClusterCullGpu) -> bool {
        let grown = params.capacity > self.capacity;
        if grown {
            self.capacity = params.capacity;
            self.draws = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Clusters/Draws"), size: self.capacity as u64 * 8, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            self.bound = None;
        }
        let (records, count) = match source {
            InstanceSource::None => (None, None),
            InstanceSource::All { records, .. } => (Some(records.clone()), None),
            InstanceSource::Culled { records, args, .. } => (Some(records.clone()), Some(args.clone())),
        };
        if self.bound.as_ref().is_none_or(|b| b.records != records || b.count != count) {
            let entry = |binding, buffer: &wgpu::Buffer| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() };
            let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Clusters/Cull"),
                layout: &culling.cull_bgl,
                entries: &[entry(0, &self.params), entry(1, &self.mesh), entry(2, records.as_ref().unwrap_or(&self.empty)), entry(3, &self.draws), entry(4, &self.args)],
            });
            let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Clusters/Prepare"),
                layout: &culling.prepare_bgl,
                entries: &[entry(10, &self.params), entry(11, count.as_ref().unwrap_or(&self.empty)), entry(12, &self.args), entry(13, &self.dispatch)],
            });
            self.bound = Some(Bound { records, count, cull_bind_group, prepare_bind_group });
        }
        if self.written != Some(params) {
            queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
            self.written = Some(params);
        }
        grown
    }

    /// The indirect draw (`DRAW_ARGS_BYTES`, see `DRAW_ARGS_BYTES` for its words).
    pub(crate) fn args(&self) -> &wgpu::Buffer {
        &self.args
    }

    /// The draw list: (record, cluster) per drawn cluster.
    pub(crate) fn draws(&self) -> &wgpu::Buffer {
        &self.draws
    }

    /// Vertices a cluster is drawn as.
    pub(crate) fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    pub(crate) fn cluster_count(&self) -> u32 {
        self.cluster_count
    }
}
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `cd rust && cargo test -p kansei-core clusters:: 2>&1 | tail -5`
Expected: PASS, 22 tests (1 ignored).

- [ ] **Step 6: Commit**

```bash
git add rust/kansei-core/src/clusters rust/kansei-core/src/shaders/cluster_cull.wgsl
git commit -m "feat(clusters): the GPU cut: a workgroup per visible instance over the levels it can reach"
```

---

### Task 4: The generated vertex stage

**Files:**
- Create: `rust/kansei-core/src/clusters/vertex_stage.rs`
- Modify: `rust/kansei-core/src/clusters/mod.rs` (`mod vertex_stage; pub(crate) use vertex_stage::{cluster_vertex_stage, CLUSTER_VERTEX_ENTRY};`)
- Test: `rust/kansei-core/src/clusters/gpu_tests.rs`

**Interfaces:**
- Consumes: `CLUSTER_MESH_WGSL`'s `kansei_cluster_vertex` and `kansei_vertex_f32` (Task 2); `crate::buffers::InstanceBufferLayout { stride: u64, attributes: Vec<wgpu::VertexAttribute> }`.
- Produces: `pub(crate) fn cluster_vertex_stage(code: &str, instances: Option<&InstanceBufferLayout>) -> Result<String, String>` and `pub(crate) const CLUSTER_VERTEX_ENTRY: &str = "kansei_cluster_vertex_main"`. The generated stage binds group 2:
  - 2: `kansei_cluster_mesh`
  - 3: `kansei_cluster_draws: array<vec2<u32>>` (record, cluster)
  - 4: `kansei_cluster_records: array<u32>`
- The vertex layout is `Vertex` at locations 0-2: position vec4 in words 0-3, normal vec3 in words 4-6, uv vec2 in words 7-8. Instance attributes are 32-bit formats only (the engine's `VertexFormat`).

- [ ] **Step 1: Write the failing tests** (append to `gpu_tests.rs`)

```rust
use super::vertex_stage::*;
use crate::buffers::InstanceBufferLayout;
use wgpu::VertexFormat::*;

fn layout(stride: u64, attributes: &[(u32, u64, wgpu::VertexFormat)]) -> InstanceBufferLayout {
    InstanceBufferLayout { stride, attributes: attributes.iter().map(|&(shader_location, offset, format)| wgpu::VertexAttribute { format, offset, shader_location }).collect() }
}

fn validate_stage(name: &str, code: &str, instances: Option<&InstanceBufferLayout>) -> naga::Module {
    let stage = cluster_vertex_stage(code, instances).unwrap_or_else(|e| panic!("{name}: {e}"));
    let module = validate(name, &stage);
    assert!(module.entry_points.iter().any(|e| e.name == CLUSTER_VERTEX_ENTRY && e.stage == naga::ShaderStage::Vertex), "{name}: no cluster entry point");
    assert!(!module.entry_points.iter().any(|e| e.name == "vertex_main"), "{name}: vertex_main is still an entry point");
    module
}

#[test]
fn the_engine_materials_get_a_cluster_stage() {
    validate_stage("basic", include_str!("../shaders/basic.wgsl"), None);
    validate_stage("basic_lit", include_str!("../shaders/basic_lit.wgsl"), None);
    let mat4 = layout(64, &[(3, 0, Float32x4), (4, 16, Float32x4), (5, 32, Float32x4), (6, 48, Float32x4)]);
    validate_stage("basic_instanced", include_str!("../shaders/basic_instanced.wgsl"), Some(&mat4));
    validate_stage("particle_billboard", include_str!("../shaders/particle_billboard.wgsl"), Some(&layout(16, &[(3, 0, Float32x4)])));
}

/// The forms the repository's materials take (the examples, the film): located parameters with
/// comments and a trailing comma, one returning a builtin; and a struct of located members with
/// instance attributes of each kind.
#[test]
fn located_and_struct_forms_get_a_cluster_stage() {
    let located = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) n: vec3<f32> };
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
@vertex
fn vertex_main(
    // per vertex
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>, /* unused */
) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.n = normal + vec3<f32>(uv, 0.0);
    return out;
}
@fragment fn fragment_main(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(in.n, 1.0); }
"#;
    validate_stage("located", located, None);
    let builtin_out = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex fn vertex_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> { return view_matrix * position; }
@fragment fn fragment_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
"#;
    validate_stage("builtin return", builtin_out, None);
    let instanced = r#"
struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) place: vec4<f32>,
    @location(4) yaw: f32,
    @location(5) ids: vec2<u32>,
    @location(6) offset: vec3i,
};
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) @interpolate(flat) id: u32 };
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = view_matrix * vec4<f32>(v.position.xyz * v.place.w + v.place.xyz + vec3<f32>(v.offset) + v.normal * v.yaw, 1.0);
    out.id = v.ids.x + v.ids.y;
    return out;
}
@fragment fn fragment_main(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(f32(in.id)); }
"#;
    let records = layout(48, &[(3, 0, Float32x4), (4, 16, Float32), (5, 20, Uint32x2), (6, 28, Sint32x3)]);
    validate_stage("instanced struct", instanced, Some(&records));
}

#[test]
fn inputs_the_cluster_path_cannot_feed_are_errors() {
    let error = |code: &str, instances: Option<&InstanceBufferLayout>| cluster_vertex_stage(code, instances).expect_err(code);
    assert!(error("@vertex fn vertex_main(@location(0) p: vec4<f32>, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> { return p; }", None).contains("instance_index"));
    assert!(error("struct VIn { @location(0) p: vec4<f32>, @builtin(vertex_index) i: u32 };\n@vertex fn vertex_main(v: VIn) -> @builtin(position) vec4<f32> { return v.p; }", None).contains("vertex_index"));
    assert!(error("@vertex fn vertex_main(@location(3) q: vec4<f32>) -> @builtin(position) vec4<f32> { return q; }", None).contains("location(3)"));
    let floats = layout(16, &[(3, 0, Float32x4)]);
    assert!(error("@vertex fn vertex_main(@location(3) q: vec4<u32>) -> @builtin(position) vec4<f32> { return vec4<f32>(q); }", Some(&floats)).contains("location(3)"));
    assert!(error("@fragment fn fragment_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }", None).contains("vertex_main"));
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core clusters::gpu_tests 2>&1 | tail -5`
Expected: FAIL to compile: `cluster_vertex_stage` not found.

- [ ] **Step 3: Implement `clusters/vertex_stage.rs`**

```rust
//! The vertex stage of a material's cluster pipeline, generated from its WGSL. `vertex_main`
//! becomes a plain function. `kansei_cluster_vertex_main` finds its draw-list entry, reads the
//! mesh's vertex and the instance record's attributes (from the instance buffer's layout),
//! fills `vertex_main`'s inputs, and calls it.

use super::gpu::CLUSTER_MESH_WGSL;
use crate::buffers::InstanceBufferLayout;

/// The generated entry point.
pub(crate) const CLUSTER_VERTEX_ENTRY: &str = "kansei_cluster_vertex_main";

/// The generated stage's group 2 bindings, after the mesh matrices at 0 and 1: the packed mesh,
/// the view's draw list and the instance records; and the readers of 1-4 components, missing
/// ones (0, 0, 0, 1) as a vertex fetch fills them.
const PRELUDE: &str = r#"
@group(2) @binding(2) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(2) @binding(3) var<storage, read> kansei_cluster_draws: array<vec2<u32>>;
@group(2) @binding(4) var<storage, read> kansei_cluster_records: array<u32>;

fn kansei_cluster_attribute(vertex: u32, word: u32, count: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    for (var i = 0u; i < count; i++) {
        v[i] = kansei_vertex_f32(vertex, word + i);
    }
    return v;
}

fn kansei_record_f32(record: u32, word: u32, count: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    for (var i = 0u; i < count; i++) {
        v[i] = bitcast<f32>(kansei_cluster_records[record * KANSEI_RECORD_WORDS + word + i]);
    }
    return v;
}

fn kansei_record_u32(record: u32, word: u32, count: u32) -> vec4<u32> {
    var v = vec4<u32>(0u, 0u, 0u, 1u);
    for (var i = 0u; i < count; i++) {
        v[i] = kansei_cluster_records[record * KANSEI_RECORD_WORDS + word + i];
    }
    return v;
}

fn kansei_record_i32(record: u32, word: u32, count: u32) -> vec4<i32> {
    var v = vec4<i32>(0, 0, 0, 1);
    for (var i = 0u; i < count; i++) {
        v[i] = bitcast<i32>(kansei_cluster_records[record * KANSEI_RECORD_WORDS + word + i]);
    }
    return v;
}
"#;

/// An input of `vertex_main`: its location, name and WGSL type.
struct Input {
    location: u32,
    name: String,
    ty: String,
}

/// `code` (a material's WGSL, includes resolved) with the cluster vertex stage, for instance
/// records laid out as `instances` (the geometry's one instance buffer, if any). The Err says
/// what the stage can't feed: a builtin input, a location no buffer provides, or a type its
/// buffer's format doesn't match.
pub(crate) fn cluster_vertex_stage(code: &str, instances: Option<&InstanceBufferLayout>) -> Result<String, String> {
    let code = strip_comments(code);
    let name = find_fn(&code, "vertex_main").ok_or("no fn vertex_main")?;
    let attributed = code[..name].trim_end();
    let vertex_attribute = attributed.strip_suffix("@vertex").ok_or("vertex_main is not marked @vertex")?.len();
    let open = name + code[name..].find('(').ok_or("vertex_main has no parameter list")?;
    let close = matching_paren(&code, open).ok_or("vertex_main's parameters are unbalanced")?;
    let body = close + code[close..].find('{').ok_or("vertex_main has no body")?;
    let returns = code[close + 1..body].trim().strip_prefix("->").ok_or("vertex_main returns nothing")?.trim().to_string();
    let params = split_top_level(&code[open + 1..close], ',');

    let (plain_params, fill, args) = if params.len() == 1 && !params[0].starts_with('@') {
        // one struct of located members
        let (arg, ty) = params[0].split_once(':').ok_or("vertex_main's parameter has no type")?;
        let ty = ty.trim();
        let members = struct_members(&code, ty).ok_or_else(|| format!("vertex_main's input struct {ty} not found"))?;
        let mut fill = format!("    var kansei_input: {ty};\n");
        for member in members? {
            fill += &format!("    kansei_input.{} = {};\n", member.name, input_expression(&member, instances)?);
        }
        (format!("{}: {ty}", arg.trim()), fill, "kansei_input".to_string())
    } else {
        let inputs = params.iter().map(|p| parse_input(p)).collect::<Result<Vec<_>, _>>()?;
        let plain = inputs.iter().map(|i| format!("{}: {}", i.name, i.ty)).collect::<Vec<_>>().join(", ");
        let args = inputs.iter().map(|i| input_expression(i, instances)).collect::<Result<Vec<_>, _>>()?.join(", ");
        (plain, String::new(), args)
    };

    let record_words = instances.map_or(0, |l| l.stride / 4);
    let mut out = String::with_capacity(code.len() + PRELUDE.len() + CLUSTER_MESH_WGSL.len() + 1024);
    out += &code[..vertex_attribute];
    out += &code[name..open + 1];
    out += &plain_params;
    out += &format!(") -> {} ", strip_attributes(&returns));
    out += &code[body..];
    out += &format!("\nconst KANSEI_RECORD_WORDS: u32 = {record_words}u;\n{PRELUDE}\n{CLUSTER_MESH_WGSL}\n");
    out += &format!("@vertex\nfn {CLUSTER_VERTEX_ENTRY}(@builtin(vertex_index) kansei_vertex_index: u32, @builtin(instance_index) kansei_instance_index: u32) -> {returns} {{\n");
    out += "    let kansei_draw = kansei_cluster_draws[kansei_instance_index];\n";
    out += "    let kansei_vertex = kansei_cluster_vertex(kansei_draw.y, kansei_vertex_index);\n";
    out += "    let kansei_record = kansei_draw.x;\n";
    out += &fill;
    out += &format!("    return vertex_main({args});\n}}\n");
    Ok(out)
}

/// `code` without its `//` and (nested) `/* */` comments.
fn strip_comments(code: &str) -> String {
    let bytes = code.as_bytes();
    let mut out = String::with_capacity(code.len());
    let (mut i, mut depth) = (0, 0);
    while i < bytes.len() {
        if bytes[i..].starts_with(b"/*") {
            depth += 1;
            i += 2;
        } else if depth > 0 && bytes[i..].starts_with(b"*/") {
            depth -= 1;
            i += 2;
            out.push(' ');
        } else if depth > 0 {
            i += 1;
        } else if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            let c = code[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// Where `fn <name>` starts: the byte index of its `fn`.
fn find_fn(code: &str, name: &str) -> Option<usize> {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    code.match_indices(name).find_map(|(at, _)| {
        let before = code[..at].trim_end();
        let fn_at = before.strip_suffix("fn")?.len();
        let boundary_before = code[..fn_at].chars().next_back().is_none_or(|c| !is_ident(c));
        let boundary_after = code[at + name.len()..].chars().next().is_some_and(|c| !is_ident(c));
        (boundary_before && boundary_after && before.len() < at).then_some(fn_at)
    })
}

/// The `)` closing the `(` at `open`.
fn matching_paren(code: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in code[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `text` split at `separator` outside brackets, trimmed, empty pieces (a trailing separator)
/// dropped.
fn split_top_level(text: &str, separator: char) -> Vec<String> {
    let (mut pieces, mut current, mut depth) = (Vec::new(), String::new(), 0i32);
    for c in text.chars() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            _ => {}
        }
        if c == separator && depth == 0 {
            pieces.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    pieces.push(current);
    pieces.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}

/// The members of `struct <name>`, or None when there is no such struct.
fn struct_members(code: &str, name: &str) -> Option<Result<Vec<Input>, String>> {
    let start = code.match_indices("struct").find_map(|(at, _)| {
        let rest = code[at + "struct".len()..].trim_start();
        let after = rest.strip_prefix(name)?;
        after.trim_start().starts_with('{').then(|| code.len() - after.trim_start().len())
    })?;
    let end = start + code[start..].find('}')?;
    Some(split_top_level(&code[start + 1..end], ',').iter().map(|m| parse_input(m)).collect())
}

/// `@location(n) name: type` (other attributes skipped); a builtin is an Err.
fn parse_input(text: &str) -> Result<Input, String> {
    let mut rest = text.trim();
    let mut location = None;
    while let Some(after) = rest.strip_prefix('@') {
        let name_end = after.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(after.len());
        let (attribute, mut tail) = after.split_at(name_end);
        let mut argument = "";
        if tail.trim_start().starts_with('(') {
            let open = tail.find('(').unwrap();
            let close = matching_paren(tail, open).ok_or_else(|| format!("unbalanced attribute in `{text}`"))?;
            argument = tail[open + 1..close].trim();
            tail = &tail[close + 1..];
        }
        match attribute {
            "builtin" => return Err(format!("vertex_main reads the builtin {argument} (`{text}`), which the cluster path doesn't provide")),
            "location" => location = Some(argument.parse::<u32>().map_err(|_| format!("bad location in `{text}`"))?),
            _ => {}
        }
        rest = tail.trim_start();
    }
    let (name, ty) = rest.split_once(':').ok_or_else(|| format!("no type in `{text}`"))?;
    let location = location.ok_or_else(|| format!("`{text}` has no @location"))?;
    Ok(Input { location, name: name.trim().to_string(), ty: ty.split_whitespace().collect() })
}

/// `text` without its `@attribute(...)`s.
fn strip_attributes(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text.trim();
    while let Some(after) = rest.strip_prefix('@') {
        let name_end = after.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(after.len());
        let mut tail = &after[name_end..];
        if tail.trim_start().starts_with('(') {
            let open = tail.find('(').unwrap();
            tail = &tail[matching_paren(tail, open).map_or(tail.len(), |c| c + 1)..];
        }
        rest = tail.trim_start();
    }
    out += rest;
    out
}

/// A WGSL input type's components and scalar kind ('f', 'u' or 'i').
fn shape(ty: &str) -> Option<(u32, char)> {
    let scalar = |s: &str| match s {
        "f32" | "f" => Some('f'),
        "u32" | "u" => Some('u'),
        "i32" | "i" => Some('i'),
        _ => None,
    };
    if let Some(kind) = scalar(ty) {
        return Some((1, kind));
    }
    let rest = ty.strip_prefix("vec")?;
    let n = rest.chars().next()?.to_digit(10).filter(|n| (2..=4).contains(n))?;
    let element = rest[1..].trim_start_matches('<').trim_end_matches('>');
    Some((n, scalar(element)?))
}

/// A 32-bit vertex format's components and scalar kind.
fn format_shape(format: wgpu::VertexFormat) -> Option<(u32, char)> {
    use wgpu::VertexFormat::*;
    Some(match format {
        Float32 => (1, 'f'),
        Float32x2 => (2, 'f'),
        Float32x3 => (3, 'f'),
        Float32x4 => (4, 'f'),
        Uint32 => (1, 'u'),
        Uint32x2 => (2, 'u'),
        Uint32x3 => (3, 'u'),
        Uint32x4 => (4, 'u'),
        Sint32 => (1, 'i'),
        Sint32x2 => (2, 'i'),
        Sint32x3 => (3, 'i'),
        Sint32x4 => (4, 'i'),
        _ => return None,
    })
}

/// The WGSL expression that feeds `input`: the mesh vertex's (locations 0-2, `Vertex::LAYOUT`)
/// or the instance record's attribute at its location.
fn input_expression(input: &Input, instances: Option<&InstanceBufferLayout>) -> Result<String, String> {
    let location = input.location;
    let (count, kind) = shape(&input.ty).ok_or_else(|| format!("@location({location}) {}: {} isn't a 32-bit scalar or vector", input.name, input.ty))?;
    let (expression, provided) = match location {
        0 => ("kansei_cluster_attribute(kansei_vertex, 0u, 4u)".to_string(), 'f'),
        1 => ("kansei_cluster_attribute(kansei_vertex, 4u, 3u)".to_string(), 'f'),
        2 => ("kansei_cluster_attribute(kansei_vertex, 7u, 2u)".to_string(), 'f'),
        _ => {
            let attribute = instances
                .and_then(|l| l.attributes.iter().find(|a| a.shader_location == location))
                .ok_or_else(|| format!("vertex_main reads @location({location}), which no buffer provides"))?;
            let (n, k) = format_shape(attribute.format).ok_or_else(|| format!("@location({location}): {:?} isn't a 32-bit format", attribute.format))?;
            if attribute.offset % 4 != 0 {
                return Err(format!("@location({location}): offset {} isn't a multiple of 4", attribute.offset));
            }
            let reader = match k {
                'f' => "kansei_record_f32",
                'u' => "kansei_record_u32",
                _ => "kansei_record_i32",
            };
            (format!("{reader}(kansei_record, {}u, {n}u)", attribute.offset / 4), k)
        }
    };
    if kind != provided {
        return Err(format!("@location({location}) is {} in vertex_main, but its buffer holds {}", input.ty, match provided { 'f' => "floats", 'u' => "u32s", _ => "i32s" }));
    }
    Ok(match count {
        1 => format!("{expression}.x"),
        2 => format!("{expression}.xy"),
        3 => format!("{expression}.xyz"),
        _ => expression,
    })
}
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `cd rust && cargo test -p kansei-core clusters:: 2>&1 | tail -5`
Expected: PASS, 25 tests (1 ignored).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/clusters
git commit -m "feat(clusters): the vertex stage generated around a material's vertex_main"
```

---

### Task 5: Materials and the renderer: the camera draws the cut

**Files:**
- Modify:
  - `rust/kansei-core/src/materials/material.rs`
  - `rust/kansei-core/src/renderers/shared_layouts.rs`
  - `rust/kansei-core/src/objects/renderable.rs`
  - `rust/kansei-core/src/renderers/renderer.rs`
  - `rust/kansei-core/src/culling/mod.rs`
  - `rust/kansei-core/src/clusters/gpu.rs`
  - `rust/kansei-core/src/clusters/mod.rs` (`pub use gpu::ClusterLod;`)
- Test: `rust/kansei-core/src/clusters/gpu_tests.rs`

**Interfaces:**
- Consumes: `ClusterGpu`, `ClusterCulling`, `ClusterCullGpu`, `ClusterViewGpu`, `InstanceSource` (Task 3); `cluster_vertex_stage`, `CLUSTER_VERTEX_ENTRY` (Task 4); `InstanceCulling::view(MAIN_VIEW) -> Option<CulledDraw { instances, instances_offset, args, offset }>`.
- Produces:
  - `pub struct ClusterLod { pub mesh: Arc<ClusterMesh>, pub transform: Option<InstanceTransform>, pub cone_culling: bool, pub capacity: Option<u32>, .. }` with `new`, `with_transform`, `with_cone_culling`, `with_capacity`
  - `Renderable::clusters: Option<ClusterLod>`
  - `Renderer::set_cluster_error_threshold(pixels)` and `cluster_error_threshold()`
  - `Material::get_cluster_pipeline(..) -> Result<&RenderPipeline, String>` and `cluster_pipeline(&PipelineKey)`
  - `SharedLayouts::cluster_mesh_bgl`
  - `ClusterGpu::bind_draw` and `draw_bind_group`
  - `Renderer::render_scene_to_gbuffer` becomes `pub(crate)`

- [ ] **Step 1: Write the failing tests** (append to `gpu_tests.rs`)

```rust
use crate::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use crate::cameras::Camera;
use crate::culling::InstanceCulling;
use crate::geometries::InstancedGeometry;
use crate::materials::{Binding, Material, MaterialOptions, ShaderStages};
use crate::objects::{Renderable, Scene, SceneNode};
use crate::renderers::{GBuffer, Renderer, RendererConfig};

/// Rocks placed by records of position + scale, then yaw (8 floats), coloured by their normals
/// in every GBuffer target.
const ROCKS_WGSL: &str = r#"
struct Tint { color: vec4<f32> };
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) yaw: f32 };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32> };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let local = turn(v.position.xyz * v.place.w, v.yaw) + v.place.xyz;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(local, 1.0);
    out.normal = (world_matrix * vec4<f32>(turn(v.normal, v.yaw), 0.0)).xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0) * tint.color;
    return FOut(n, vec4<f32>(0.0), n, n);
}
"#;

const SIZE: u32 = 192;

fn headless() -> Option<Renderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: SIZE, height: SIZE, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// A scene of rocks at `placements` (x, y, z, scale, yaw), culled per instance (the first
/// `visible` records counted), with cluster LOD or without, and a camera looking at them.
fn rocks(renderer: &Renderer, placements: &[[f32; 5]], visible: u32, clusters: bool) -> (Scene, Camera, usize) {
    use wgpu::util::DeviceExt;
    let data: Vec<f32> = placements.iter().flat_map(|p| [p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, 0.0]).collect();
    let source = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE });
    let instances = ComputeBuffer::from_external("Rocks", source.clone(), BufferType::Storage).with_vertex_layout(
        32,
        vec![InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 }, InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32 }],
    );
    let mut material = Material::new("Rocks", ROCKS_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    let mut r = Renderable::new(InstancedGeometry::new(rock(4, false), visible, vec![instances]), material);
    r.instance_culling = Some(InstanceCulling::new(source, visible, 32, 0, 1.2).with_radius_scale(12));
    if clusters {
        let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
        r.clusters = Some(ClusterLod::new(mesh).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), rotation: None }));
    }
    let mut scene = Scene::new();
    let index = scene.add(SceneNode::Renderable(r));
    let mut camera = Camera::new(50.0, 0.1, 200.0, 1.0);
    camera.set_position(0.5, 1.5, 6.0);
    camera.look_at(&crate::math::Vec3::new(0.0, 0.0, -1.5));
    camera.update_projection_matrix();
    (scene, camera, index)
}

/// Draw the scene into a GBuffer and read the albedo back.
fn draw(renderer: &mut Renderer, scene: &mut Scene, camera: &mut Camera) -> Vec<[u8; 4]> {
    let gbuffer = GBuffer::new(renderer.device(), SIZE, SIZE, 1);
    renderer.render_scene_to_gbuffer(scene, camera, &gbuffer);
    crate::impostors::tests_support::read_texels(renderer.device(), renderer.queue(), &gbuffer.albedo_texture)
}

/// (texels covered, texels differing by more than 1/255 in a channel).
fn compare(a: &[[u8; 4]], b: &[[u8; 4]]) -> (usize, usize) {
    let covered = a.iter().filter(|t| t[3] > 0).count();
    let differing = a.iter().zip(b).filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.abs_diff(*q) > 1)).count();
    (covered, differing)
}

/// The cluster draw's words, read back.
fn cluster_args(renderer: &Renderer, scene: &Scene, index: usize) -> Vec<u32> {
    let gpu = scene.get_renderable(index).unwrap().clusters.as_ref().expect("still on the cluster path").gpu.as_ref().unwrap();
    read_words(renderer.device(), renderer.queue(), gpu.args())
}

const PLACEMENTS: [[f32; 5]; 4] = [[0.0, 0.0, 0.0, 1.0, 0.0], [2.6, 0.3, -1.5, 0.8, 1.1], [-2.4, -0.2, -2.0, 1.2, -0.6], [0.5, 1.8, -4.0, 1.5, 2.2]];

#[test]
fn at_zero_error_the_clusters_draw_what_the_mesh_does() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    renderer.set_cluster_error_threshold(0.0);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, false);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, true);
    let clusters = draw(&mut renderer, &mut scene, &mut camera);
    let args = cluster_args(&renderer, &scene, index);
    assert!(args[1] > 0 && args[4] == 4, "clusters drawn: {args:?}");
    let (covered, differing) = compare(&mesh, &clusters);
    assert!(covered > (SIZE * SIZE / 10) as usize, "the rocks cover {covered} texels");
    assert!(differing * 200 < covered, "{differing} of {covered} texels differ");
}

#[test]
fn a_pixel_of_error_draws_far_fewer_triangles_and_nearly_the_same_image() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let far: Vec<[f32; 5]> = (0..12).map(|k| [(k % 4) as f32 * 3.0 - 4.5, 0.0, -10.0 - (k / 4) as f32 * 6.0, 1.3, k as f32]).collect();
    let mut runs = Vec::new();
    for threshold in [0.0, 1.0] {
        renderer.set_cluster_error_threshold(threshold);
        let (mut scene, mut camera, index) = rocks(&renderer, &far, far.len() as u32, true);
        let image = draw(&mut renderer, &mut scene, &mut camera);
        runs.push((image, cluster_args(&renderer, &scene, index)[6]));
    }
    assert!(runs[1].1 * 3 < runs[0].1, "triangles at 0 and 1 px: {} and {}", runs[0].1, runs[1].1);
    let (covered, differing) = compare(&runs[0].0, &runs[1].0);
    assert!(covered > 500 && differing * 20 < covered, "{differing} of {covered} texels differ");
}

#[test]
fn grown_instances_rebind_the_clusters() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    renderer.set_cluster_error_threshold(0.0);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 2, true);
    draw(&mut renderer, &mut scene, &mut camera);
    // the other two records join: instance culling remakes its buffers, the clusters rebind
    let r = scene.get_renderable_mut(index).unwrap();
    r.instance_culling.as_mut().unwrap().count = 4;
    r.geometry.instance_count = 4;
    let clusters = draw(&mut renderer, &mut scene, &mut camera);
    assert_eq!(cluster_args(&renderer, &scene, index)[4], 4);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, false);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (covered, differing) = compare(&mesh, &clusters);
    assert!(differing * 200 < covered, "{differing} of {covered} texels differ");
}

#[test]
fn a_material_the_cluster_path_cannot_feed_keeps_the_mesh() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, true);
    // reads the instance index: no cluster stage, drawn as before
    let r = scene.get_renderable_mut(index).unwrap();
    r.material = Material::new(
        "Rocks",
        &ROCKS_WGSL.replace("@location(4) yaw: f32", "@location(4) yaw: f32, @builtin(instance_index) instance: u32"),
        vec![Binding::uniform(0, ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    r.material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    let image = draw(&mut renderer, &mut scene, &mut camera);
    assert!(scene.get_renderable(index).unwrap().clusters.is_none(), "cluster LOD dropped");
    assert!(image.iter().filter(|t| t[3] > 0).count() > (SIZE * SIZE / 10) as usize, "still drawn");
}
```

(`read_texels` moves from `impostors/tests.rs` into a `#[cfg(test)] pub(crate) mod tests_support` in `impostors/mod.rs`, which `impostors/tests.rs` then uses: one helper, two users.)

- [ ] **Step 2: Run them to see them fail**

Run: `cd rust && cargo test -p kansei-core clusters::gpu_tests 2>&1 | tail -5`
Expected: FAIL to compile: `ClusterLod`, `Renderable::clusters`, `set_cluster_error_threshold` not found; `render_scene_to_gbuffer` is private.

- [ ] **Step 3: `ClusterLod` and the draw's bind group** (`clusters/gpu.rs`)

```rust
/// Cluster LOD for a renderable (`Renderable::clusters`). Each frame, the camera draws the cut of
/// `mesh` its view needs (`Renderer::set_cluster_error_threshold`) instead of the geometry,
/// through a vertex stage generated around the material's `vertex_main`. The geometry, which
/// must be the mesh `mesh` was built from, is still what the other views draw (shadow maps,
/// reflections, velocity, impostor bakes) until they move to clusters too.
///
/// The instances are the geometry's one instance buffer (if any), culled by the renderable's
/// `InstanceCulling` when it has one (without occlusion phases) and placed as `transform` says.
/// Set it before the renderable is first drawn, or call `Renderer::invalidate_bundle` after.
pub struct ClusterLod {
    pub mesh: std::sync::Arc<ClusterMesh>,
    /// How an instance record places the mesh. None: the instances are drawn where the renderable
    /// is (or there are none).
    pub transform: Option<InstanceTransform>,
    /// Skip clusters whose every triangle faces away (on by default). Turn it off when the
    /// material turns instances in a way `transform` doesn't describe.
    pub cone_culling: bool,
    /// Clusters drawn per frame, at most. By default every cluster of every instance, up to
    /// 4 194 304. Clusters past it aren't drawn.
    pub capacity: Option<u32>,
    pub(crate) gpu: Option<ClusterGpu>,
}

impl ClusterLod {
    pub fn new(mesh: impl Into<std::sync::Arc<ClusterMesh>>) -> Self {
        Self { mesh: mesh.into(), transform: None, cone_culling: true, capacity: None, gpu: None }
    }

    pub fn with_transform(mut self, transform: InstanceTransform) -> Self {
        self.transform = Some(transform);
        self
    }

    pub fn with_cone_culling(mut self, on: bool) -> Self {
        self.cone_culling = on;
        self
    }

    pub fn with_capacity(mut self, clusters: u32) -> Self {
        self.capacity = Some(clusters);
        self
    }

    /// Ready this frame's cut: the GPU state made once, the cull bound to `source` (records of
    /// `stride` bytes) with the parameters, and the vertex stage's group 2 (`layout`) over the
    /// renderer's normal and world matrices. True when what bundles recorded changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, layout: &wgpu::BindGroupLayout, matrices: (&wgpu::Buffer, &wgpu::Buffer), source: InstanceSource, stride: u32, world: glam::Mat4) -> bool {
        let gpu = self.gpu.get_or_insert_with(|| ClusterGpu::new(device, &self.mesh));
        let every = (source.capacity() as u64 * gpu.cluster_count as u64).min(DEFAULT_MAX_DRAWN as u64) as u32;
        let capacity = self.capacity.unwrap_or(every).max(1);
        let params = ClusterCullGpu::new(world, self.transform, stride, &source, capacity, gpu.vertex_count, self.cone_culling);
        let grown = gpu.bind(device, queue, culling, source, params);
        gpu.bind_draw(device, layout, matrices.0, matrices.1) || grown
    }
}
```

In `ClusterGpu`, add a field `draw: Option<((wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, Option<wgpu::Buffer>), wgpu::BindGroup)>` (initialised to `None`) and:

```rust
    /// The vertex stage's group 2 (`SharedLayouts::cluster_mesh_bgl`): the renderer's normal and
    /// world matrices, the mesh, the draw list and the bound records. Remade when any of them
    /// changed; true then (bundles recorded the old one).
    pub(crate) fn bind_draw(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, normal: &wgpu::Buffer, world: &wgpu::Buffer) -> bool {
        let key = (normal.clone(), world.clone(), self.draws.clone(), self.bound.as_ref().and_then(|b| b.records.clone()));
        if self.draw.as_ref().is_some_and(|(k, _)| *k == key) {
            return false;
        }
        let matrix = |binding, buffer: &wgpu::Buffer, size| wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: 0, size: std::num::NonZeroU64::new(size) }) };
        let entry = |binding, buffer: &wgpu::Buffer| wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clusters/Draw"),
            layout,
            entries: &[matrix(0, normal, 64), matrix(1, world, 128), entry(2, &self.mesh), entry(3, &self.draws), entry(4, key.3.as_ref().unwrap_or(&self.empty))],
        });
        self.draw = Some((key, group));
        true
    }

    pub(crate) fn draw_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.draw.as_ref().map(|(_, group)| group)
    }
```

- [ ] **Step 4: The shared layout** (`renderers/shared_layouts.rs`: a `cluster_mesh_bgl` field, made in `new` after `mesh_bgl`)

```rust
        // Group 2 of cluster pipelines (clusters::cluster_vertex_stage): the mesh matrices as in
        // `mesh_bgl`, then the packed cluster mesh, the view's draw list and the instance records
        let matrix = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None }, count: None };
        let storage = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let cluster_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shared/ClusterMeshBGL"),
            entries: &[matrix(0), matrix(1), storage(2), storage(3), storage(4)],
        });
```

- [ ] **Step 5: The material's cluster pipeline** (`materials/material.rs`)

Add the fields and their `new` initialisers:

```rust
    /// Cluster pipelines (`get_cluster_pipeline`), keyed with no vertex buffers.
    pub(crate) cluster_pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    /// The cluster stage's module for the instance layout it was made for, or why there is none.
    cluster_module: Option<(Option<(u64, Vec<wgpu::VertexAttribute>)>, Result<wgpu::ShaderModule, String>)>,
    cluster_pipeline_layout: Option<wgpu::PipelineLayout>,
```

`ensure_shared` builds its module from `self.processed_code()`, and `get_pipeline`'s pipeline creation moves into `create_pipeline`, which `get_pipeline` calls with `self.pipeline_layout`, `self.shader_module`, `"vertex_main"`, `vertex_layouts` and label `"Pipeline"`:

```rust
    /// The WGSL with its includes resolved.
    fn processed_code(&self) -> String {
        match &self.shader_chunks {
            Some(chunks) => crate::materials::parse_includes(&self.shader_code, chunks),
            None => self.shader_code.clone(),
        }
    }

    /// A render pipeline of this material: `vertex_entry` of `module` over `vertex_layouts`, its
    /// `fragment_main`, into the given targets.
    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        vertex_entry: &str,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
        label: &str,
    ) -> wgpu::RenderPipeline {
        // (the body of get_pipeline's `if !contains_key` block, unchanged, with `layout`, `module`,
        // `vertex_entry` and `vertex_layouts` in place of the material's own, returning the
        // pipeline instead of inserting it; label format!("{}/{label}", self.label))
    }

    /// Get or create the pipeline that draws this material over a cluster draw (the camera's cut
    /// of `Renderable::clusters`): its WGSL with the generated vertex stage for `instances`'
    /// records. The Err says why the stage can't be generated; the renderable then keeps the
    /// ordinary path.
    pub(crate) fn get_cluster_pipeline(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedLayouts,
        instances: Option<&crate::buffers::InstanceBufferLayout>,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Result<&wgpu::RenderPipeline, String> {
        assert!(self.pipeline_layout.is_some(), "Material not initialized — call initialize() first");
        let layout_key = instances.map(|l| (l.stride, l.attributes.clone()));
        if self.cluster_module.as_ref().is_none_or(|(k, _)| *k != layout_key) {
            let module = crate::clusters::cluster_vertex_stage(&self.processed_code(), instances).map(|code| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("{}/ClusterShader", self.label)), source: wgpu::ShaderSource::Wgsl(code.into()) })
            });
            self.cluster_module = Some((layout_key, module));
            self.cluster_pipeline_cache.clear();
        }
        let module = self.cluster_module.as_ref().unwrap().1.clone()?;
        let layout = self
            .cluster_pipeline_layout
            .get_or_insert_with(|| {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}/ClusterPipelineLayout", self.label)),
                    bind_group_layouts: &[self.material_bgl.as_ref().unwrap(), &shared.camera_bgl, &shared.cluster_mesh_bgl, &shared.shadow_bgl],
                    push_constant_ranges: &[],
                })
            })
            .clone();
        let key = PipelineKey { color_formats: color_formats.to_vec(), depth_format, sample_count, num_vertex_buffers: 0 };
        if !self.cluster_pipeline_cache.contains_key(&key) {
            let pipeline = self.create_pipeline(device, &layout, &module, crate::clusters::CLUSTER_VERTEX_ENTRY, &[], color_formats, depth_format, sample_count, "ClusterPipeline");
            self.cluster_pipeline_cache.insert(key.clone(), pipeline);
        }
        Ok(&self.cluster_pipeline_cache[&key])
    }

    /// The cluster pipeline made for a pass with `key` (whatever its vertex buffers), if any.
    pub(crate) fn cluster_pipeline(&self, key: &PipelineKey) -> Option<&wgpu::RenderPipeline> {
        self.cluster_pipeline_cache.get(&PipelineKey { num_vertex_buffers: 0, ..key.clone() })
    }
```

- [ ] **Step 6: The renderable and the renderer**

`objects/renderable.rs`: add `pub clusters: Option<crate::clusters::ClusterLod>` (docs: "Draw the camera's cut of a cluster graph instead of the geometry; see `ClusterLod`"), initialised to `None` in `new`. `culling/mod.rs`: `pub(crate) use occlusion::{MainView, Occlusion};`.

`renderers/renderer.rs`:

1. Fields `cluster_culling: Option<crate::clusters::ClusterCulling>` (`None`) and `cluster_threshold: f32` (`1.0`), and:

```rust
    /// Cluster LOD's error budget (`Renderable::clusters`), in pixels at the render size: each
    /// cluster drawn is the coarsest whose simplification error the camera sees under it. 1 by
    /// default; 0 draws the full mesh.
    pub fn set_cluster_error_threshold(&mut self, pixels: f32) {
        self.cluster_threshold = pixels.max(0.0);
    }

    pub fn cluster_error_threshold(&self) -> f32 {
        self.cluster_threshold
    }

    /// Make `r`'s cluster pipeline for a pass's targets, or drop its cluster LOD with a warning
    /// when the cluster path can't draw it: it keeps the ordinary path.
    fn prepare_cluster_pipeline(&self, r: &mut crate::objects::Renderable, color_formats: &[wgpu::TextureFormat], depth_format: wgpu::TextureFormat, sample_count: u32) {
        if r.clusters.is_none() {
            return;
        }
        let first = r.geometry.instance_buffers.first();
        let problem = if r.geometry.instance_buffers.len() > 1 {
            Some("more than one instance buffer".to_string())
        } else if first.is_some_and(|cb| cb.vertex_layout().is_none_or(|l| l.stride % 4 != 0)) {
            Some("an instance buffer without a vertex layout of whole words".to_string())
        } else if r.instance_culling.is_none() && first.is_some_and(|cb| cb.gpu_buffer().is_some_and(|b| !b.usage().contains(wgpu::BufferUsages::STORAGE))) {
            Some("an instance buffer without STORAGE usage (and no InstanceCulling)".to_string())
        } else {
            let layout = first.and_then(|cb| cb.vertex_layout());
            r.material.get_cluster_pipeline(self.device.as_ref().unwrap(), self.shared_layouts.as_ref().unwrap(), layout.as_ref(), color_formats, depth_format, sample_count).err()
        };
        if let Some(problem) = problem {
            log::warn!("{}: no cluster LOD ({problem}); drawn as is", r.material.label);
            r.clusters = None;
        }
    }
```

2. Call `self.prepare_cluster_pipeline(r, &GBuffer::MRT_FORMATS, GBuffer::DEPTH_FORMAT, sample_count)` after `prepare_for_gbuffer` in `render_scene_to_gbuffer`'s prepare loop. In `render`'s loop, call `self.prepare_cluster_pipeline(r, &[format], depth_format, sample_count)` after its `get_pipeline`. Make `render_scene_to_gbuffer` `pub(crate)`.

3. `run_instance_culling(&mut self, scene, camera, depth_size, target_height: u32)`. Callers pass `gbuffer.height` (`render_scene_to_gbuffer`) and `self.config.height` (`render`).
   - When `culled` is empty, it now runs the clusters before returning:

```rust
        if culled.is_empty() {
            let main = self.occlusion.main_view(camera);
            self.run_cluster_culling(scene, camera, &main, target_height);
            return;
        }
```

   - In the per-renderable loop, cluster renderables get no occlusion phases:

```rust
            let clustered = r.clusters.is_some();
            let culling = r.instance_culling.as_mut().unwrap();
            ...
            let two_phase_views: &[usize] = if culling.occlusion && !clustered { &occlusion_views } else { &[] };
```

   - After its `queue.submit(..)` and bookkeeping, it calls `self.run_cluster_culling(scene, camera, &main, target_height);`.

4. The cluster pass:

```rust
    /// Cluster LOD for the camera, once the instances are culled. Each visible renderable with
    /// `clusters` gets its mesh's cut for the main view (`main`: the live or frozen camera,
    /// `target_height` pixels high), which `CameraClusterDraw` draws.
    fn run_cluster_culling(&mut self, scene: &mut Scene, camera: &Camera, main: &crate::culling::MainView, target_height: u32) {
        let indices: Vec<usize> = scene.ordered_indices().filter(|&i| scene.get_renderable(i).is_some_and(|r| r.visible && r.clusters.is_some() && r.geometry.initialized)).collect();
        if indices.is_empty() {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let culling = self.cluster_culling.get_or_insert_with(|| crate::clusters::ClusterCulling::new(device));
        let pixels_per_radian = target_height as f32 / (2.0 * (camera.fov.to_radians() * 0.5).tan());
        culling.set_view(queue, &crate::clusters::ClusterViewGpu::new(main.cull.view_proj, main.lod_origin, pixels_per_radian, camera.near, self.cluster_threshold));
        let layout = &self.shared_layouts.as_ref().unwrap().cluster_mesh_bgl;
        let matrices = (self.normal_matrices_buf.as_ref().unwrap(), self.world_matrices_buf.as_ref().unwrap());
        let mut stale = false;
        let mut prepared = Vec::new();
        for &idx in &indices {
            let r = scene.get_renderable_mut(idx).unwrap();
            let world = r.world_matrix.to_glam();
            let first = r.geometry.instance_buffers.first();
            let stride = first.and_then(|cb| cb.vertex_layout()).map_or(0, |l| l.stride as u32);
            let source = match (first, r.instance_culling.as_ref()) {
                (None, _) => crate::clusters::InstanceSource::None,
                (Some(_), Some(c)) => {
                    let Some(draw) = c.view(MAIN_VIEW) else { continue };
                    crate::clusters::InstanceSource::Culled { records: draw.instances, first_record: (draw.instances_offset / c.stride as u64) as u32, capacity: c.count, args: draw.args, count_word: (draw.offset / 4) as u32 + 1 }
                }
                (Some(cb), None) => match cb.gpu_buffer() {
                    Some(records) => crate::clusters::InstanceSource::All { records, count: r.geometry.instance_count },
                    None => continue,
                },
            };
            stale |= r.clusters.as_mut().unwrap().prepare(device, queue, culling, layout, matrices, source, stride, world);
            prepared.push(idx);
        }
        let gpus: Vec<&crate::clusters::ClusterGpu> = prepared.iter().filter_map(|&i| scene.get_renderable(i)?.clusters.as_ref()?.gpu.as_ref()).collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/ClusterCulling") });
        culling.encode(&mut encoder, &gpus);
        queue.submit(Some(encoder.finish()));
        if stale {
            self.invalidate_bundle();
        }
    }
```

5. The draws. In `build_render_bundle` and `draw_dynamic_renderables`, the pipeline comes from `CameraClusterDraw::of(r, &key)` when there is one; otherwise, as before:

```rust
            let cluster = CameraClusterDraw::of(r, &key);
            let Some(pipeline) = cluster.map(|c| c.pipeline).or_else(|| r.material.pipeline_cache.get(&key)) else { continue };
            // ... pipeline and material group as before ...
            let offset = mesh_offset(scene_idx, alignment);
            match cluster {
                Some(cluster) => cluster.draw(&mut encoder, set, offset),
                None => {
                    encoder.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);
                    set.draw(&mut encoder, r);
                }
            }
```

with, next to `draw_geometry`:

```rust
/// A renderable's camera draw on the cluster path: its material's cluster pipeline for the pass,
/// the vertex stage's group 2, and the indirect draw the cluster cull wrote.
#[derive(Clone, Copy)]
struct CameraClusterDraw<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    group: &'a wgpu::BindGroup,
    args: &'a wgpu::Buffer,
}

impl<'a> CameraClusterDraw<'a> {
    /// `r`'s, when it has cluster LOD ready and a cluster pipeline for a pass of `key`.
    fn of(r: &'a crate::objects::Renderable, key: &crate::materials::PipelineKey) -> Option<Self> {
        let gpu = r.clusters.as_ref()?.gpu.as_ref()?;
        Some(Self { pipeline: r.material.cluster_pipeline(key)?, group: gpu.draw_bind_group()?, args: gpu.args() })
    }

    /// Draw it in `set` (nothing in the late set: clusters have no occlusion phases yet).
    fn draw(self, enc: &mut impl wgpu::util::RenderEncoder<'a>, set: DrawSet, offset: u32) {
        if set != DrawSet::Late {
            enc.set_bind_group(2, self.group, &[offset, offset]);
            enc.draw_indirect(self.args, 0);
        }
    }
}
```

- [ ] **Step 7: Run the tests to see them pass, then the whole crate**

Run: `cd rust && cargo test -p kansei-core 2>&1 | tail -5`
Expected: PASS, every test (the 4 new renderer tests included), and M1's and the renderer's own unchanged.

- [ ] **Step 8: Build for wasm32**

Run: `cd rust && cargo build -p kansei-core --target wasm32-unknown-unknown 2>&1 | tail -2`
Expected: `Finished`.

- [ ] **Step 9: Commit**

```bash
git add rust/kansei-core/src
git commit -m "feat(clusters): the camera draws each cluster renderable's cut"
```

---

### Task 6: Example `cluster-lod`, and the measurement

**Files:**
- Create: `rust/kansei-wasm/examples/cluster-lod/Cargo.toml`, `rust/kansei-wasm/examples/cluster-lod/src/lib.rs`, `rust/kansei-wasm/examples/cluster-lod/www/index.html`
- Modify: `rust/Cargo.toml` (`exclude` += `"kansei-wasm/examples/cluster-lod"`)

**Interfaces:**
- Consumes: `ClusterMesh::build`, `ClusterLod`, `InstanceTransform::Placement`, `Renderer::set_cluster_error_threshold`, `pacing::FrameTimer`.
- Produces: a page with these URL parameters:
  - `mode=clusters|lods|full`
  - `bench=lods|full`: alternate `clusters` with that mode, 3 s phases × 8, and report each mode's mean GPU time and frame interval
  - `n=<rocks per side>`, `sub=<subdivisions>`, `tau=<pixels>`, `size=<w>x<h>`

The field is `n` × `n` rocks (default 24 × 24), each a noisy icosphere of `sub` subdivisions (default 6: 81,920 triangles), turned and scaled 0.7-1.3. It is drawn three ways, with one material:
- `clusters`: one renderable with cluster LOD.
- `lods`: four discrete LODs cut from the same graph, each the cut seen from its band's near edge at `tau` (the same error budget), switched by distance with `InstanceCulling`.
- `full`: the mesh as is.

The camera flies a loop through the field.

- [ ] **Step 1: `Cargo.toml`** (as `impostors/Cargo.toml`, with `name = "kansei-wasm-cluster-lod"`)

- [ ] **Step 2: `www/index.html`**

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Kansei — Cluster LOD</title>
    <style>
        body { margin: 0; background: #000; overflow: hidden; font: 12px/1.5 ui-monospace, Menlo, monospace; color: #eee; }
        canvas { width: 100vw; height: 100vh; display: block; }
        #panel { position: fixed; left: 12px; top: 12px; padding: 8px 10px; background: rgba(0, 0, 0, 0.6); border-radius: 4px; }
        #hud, #bench { margin: 0; white-space: pre-wrap; }
    </style>
</head>
<body>
    <canvas id="kansei"></canvas>
    <div id="panel">
        <pre id="hud"></pre>
        <pre id="bench"></pre>
    </div>
    <script type="module">
        import init, { start } from '../pkg/kansei_wasm_cluster_lod.js';
        await init();
        await start('kansei');
    </script>
</body>
</html>
```

- [ ] **Step 3: `src/lib.rs`**

```rust
//! Cluster LOD: a field of rocks (24 x 24 by default, 81 920 triangles each), drawn three ways
//! with one material. `clusters`: one renderable with cluster LOD, each rock's cut picked per
//! cluster every frame. `lods`: four discrete LODs cut from the same cluster graph, each at the
//! error budget from its band's near edge, switched per rock by distance. `full`: the mesh as is.
//! The HUD shows the GPU time of each frame (timestamp queries when the adapter has them) and
//! the frame interval.
//!
//! URL parameters: `mode=clusters|lods|full`, `n=<rocks per side>`, `sub=<subdivisions>`,
//! `tau=<pixels>` (the error budget), `size=<w>x<h>`, `bench=lods|full` (alternate clusters and
//! that mode every 3 s, 8 times, and report the mean GPU time and frame interval of each).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, InstanceTransform, LodView};
use kansei_core::culling::InstanceCulling;
use kansei_core::geometries::{Geometry, InstancedGeometry, PlaneGeometry, Vertex};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FrameTimer;
use kansei_core::postprocessing::{effects::{exposure_from_ev100, ToneMapEffect, ToneMapOptions}, PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};

/// Rocks placed by records of position + scale, then yaw (8 floats), lit by a low sun.
const ROCK_WGSL: &str = r#"
struct Surface { color: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) yaw: f32 };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) world: vec3<f32> };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let world = world_matrix * vec4<f32>(turn(v.position.xyz * v.place.w, v.yaw) + v.place.xyz, 1.0);
    out.clip = projection_matrix * view_matrix * world;
    out.normal = (world_matrix * vec4<f32>(turn(v.normal, v.yaw), 0.0)).xyz;
    out.world = world.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = normalize(in.normal);
    let sun = normalize(vec3<f32>(0.4, 0.6, 0.3));
    let light = vec3<f32>(1.0, 0.95, 0.85) * max(dot(n, sun), 0.0) * 3.0 + vec3<f32>(0.25, 0.3, 0.4) * (0.6 + 0.4 * n.y);
    let color = surface.color.rgb * light;
    return FOut(vec4<f32>(color, 1.0), vec4<f32>(0.0), vec4<f32>(n * 0.5 + 0.5, 1.0), surface.color);
}
"#;

/// Distances (metres) where the discrete LODs start.
const BANDS: [f32; 4] = [0.0, 8.0, 24.0, 72.0];
const SPACING: f32 = 4.0;
const MODES: [&str; 3] = ["clusters", "lods", "full"];

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

/// A noisy icosphere of `subdivisions`, about a metre across.
fn rock(subdivisions: u32) -> Geometry {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut p: Vec<glam::Vec3> = [[-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0], [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t], [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]]
        .iter()
        .map(|v| glam::Vec3::from(*v).normalize())
        .collect();
    let mut f: Vec<[u32; 3]> = vec![[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for _ in 0..subdivisions {
        let mut mid = HashMap::new();
        let mut next = Vec::with_capacity(f.len() * 4);
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
    let height = |v: glam::Vec3| 1.0 + 0.12 * (5.0 * v.x).sin() * (4.0 * v.y).cos() + 0.06 * (13.0 * v.z + 2.0 * v.x).sin() + 0.02 * (37.0 * v.y).sin() * (31.0 * v.x).cos();
    let vertices: Vec<Vertex> = p
        .iter()
        .map(|&v| {
            // the normal of the displaced surface, from two nearby points on it
            let (a, b) = (v.any_orthonormal_vector(), v.cross(v.any_orthonormal_vector()));
            let at = |d: glam::Vec3| (d.normalize()) * height(d.normalize());
            let n = (at(v + a * 1e-3) - at(v - a * 1e-3)).cross(at(v + b * 1e-3) - at(v - b * 1e-3)).normalize();
            let q = v * height(v);
            Vertex { position: [q.x, q.y * 0.7, q.z, 1.0], normal: (n * glam::Vec3::new(0.7, 1.0, 0.7)).normalize().to_array(), uv: [v.x * 0.5 + 0.5, v.y * 0.5 + 0.5] }
        })
        .collect();
    Geometry::new("Rock", vertices, f.into_iter().flatten().collect())
}

/// The cut of `mesh` seen from `distance` away at `tau` pixels (`ppr` pixels per radian), as a
/// mesh: a discrete LOD for a band starting there (for the largest rock, `scale`).
fn lod_mesh(mesh: &ClusterMesh, distance: f32, scale: f32, ppr: f32, tau: f32) -> Geometry {
    let view = LodView { eye: glam::Vec3::new(0.0, 0.0, distance / scale), pixels_per_radian: ppr, near: 0.1 / scale, threshold: tau };
    let indices: Vec<u32> = mesh.select(&view).into_iter().flat_map(|c| mesh.triangles(c).flatten().collect::<Vec<_>>()).collect();
    Geometry::new("Rock/LOD", mesh.vertices.clone(), indices)
}

fn material() -> Material {
    let mut m = Material::new("Rock", ROCK_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "Rock", &[[0.42f32, 0.4, 0.37, 1.0]]);
    m
}

/// `bench=<mode>`: alternate clusters and `mode`, averaging each one's GPU time and frame
/// interval (after a warm-up, and ignoring the start of each phase).
struct Bench {
    other: usize,
    start: f64,
    /// (GPU ms, GPU samples, frame intervals ms, frames) of clusters and the other mode
    sums: [(f64, u32, f64, u32); 2],
    last_frame: f64,
    report: Option<String>,
}

const BENCH_WARMUP_MS: f64 = 3000.0;
const BENCH_PHASE_MS: f64 = 3000.0;
const BENCH_SETTLE_MS: f64 = 500.0;
const BENCH_PHASES: u32 = 8;

impl Bench {
    /// (mode, measuring) at `now`, or None when done.
    fn phase(&self, now: f64) -> Option<(usize, bool)> {
        let t = now - self.start - BENCH_WARMUP_MS;
        if t < 0.0 {
            return Some((0, false));
        }
        let phase = (t / BENCH_PHASE_MS) as u32;
        (phase < BENCH_PHASES).then_some((if phase.is_multiple_of(2) { 0 } else { self.other }, t % BENCH_PHASE_MS >= BENCH_SETTLE_MS))
    }

    fn record(&mut self, gpu: &[f64], now: f64) {
        if let Some((mode, true)) = self.phase(now) {
            let sum = &mut self.sums[(mode != 0) as usize];
            sum.0 += gpu.iter().sum::<f64>();
            sum.1 += gpu.len() as u32;
            sum.2 += now - self.last_frame;
            sum.3 += 1;
        }
        self.last_frame = now;
        if self.phase(now).is_none() && self.report.is_none() {
            let mean = |(sum, n, _, _): (f64, u32, f64, u32)| if n > 0 { format!("{:.2} ms GPU ({n} samples)", sum / n as f64) } else { "no GPU timestamps".into() };
            let interval = |(_, _, sum, n): (f64, u32, f64, u32)| format!("{:.2} ms/frame ({n} frames)", sum / n.max(1) as f64);
            self.report = Some(format!("bench: clusters {}, {} | {} {}, {}", mean(self.sums[0]), interval(self.sums[0]), MODES[self.other], mean(self.sums[1]), interval(self.sums[1])));
        }
    }
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    volume: PostProcessingVolume,
    timer: FrameTimer,
    bench: Option<Bench>,
    /// scene indices of each mode's renderables
    modes: [Vec<usize>; 3],
    mode: usize,
    rocks: u32,
    triangles: usize,
    build_ms: f64,
    start: f64,
    frame: u32,
    last_frame: f64,
    interval_ms: f64,
    gpu_ms: f64,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

fn set_text(id: &str, text: &str) {
    if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id(id)) {
        el.set_text_content(Some(text));
    }
}

fn set_mode(st: &mut State, mode: usize) {
    for (m, indices) in st.modes.iter().enumerate() {
        for &i in indices {
            if let Some(r) = st.scene.get_renderable_mut(i) {
                r.visible = m == mode;
            }
        }
    }
    st.mode = mode;
}

/// A loop through the field, low over the rocks, looking ahead and down.
fn place_camera(camera: &mut Camera, extent: f32, t: f32) {
    let a = t * 0.08;
    let r = extent * 0.35;
    let (x, z) = (r * a.cos(), r * a.sin());
    camera.set_position(x, 2.2 + 0.8 * (t * 0.3).sin(), z);
    camera.look_at(&Vec3::new(x - 12.0 * a.sin(), 0.0, z + 12.0 * a.cos()));
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id(canvas_id).ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;
    let (width, height) = query_param("size")
        .and_then(|s| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?))))
        .unwrap_or((canvas.client_width() as u32, canvas.client_height() as u32));
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 1, clear_color: Vec4::new(0.45, 0.55, 0.7, 1.0), ..Default::default() });
    renderer.initialize_with_canvas(canvas.clone()).await;
    let tau: f32 = query_param("tau").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    renderer.set_cluster_error_threshold(tau);

    // the rocks and their graph
    let n: u32 = query_param("n").and_then(|v| v.parse().ok()).unwrap_or(24);
    let sub: u32 = query_param("sub").and_then(|v| v.parse().ok()).unwrap_or(6);
    let geometry = rock(sub);
    let triangles = geometry.indices.len() / 3;
    let before = now_ms();
    let mesh = Arc::new(ClusterMesh::build(&geometry, &ClusterOptions::default()));
    let build_ms = now_ms() - before;

    // the field: position + scale, yaw (8 floats a rock)
    let extent = n as f32 * SPACING;
    let mut data: Vec<f32> = Vec::with_capacity((n * n * 8) as usize);
    for k in 0..n * n {
        let (i, j) = (k % n, k / n);
        let x = -extent / 2.0 + (i as f32 + 0.2 + 0.6 * hash(k)) * SPACING;
        let z = -extent / 2.0 + (j as f32 + 0.2 + 0.6 * hash(k + 7919)) * SPACING;
        data.extend_from_slice(&[x, 0.2, z, 0.7 + 0.6 * hash(k + 104729), hash(k + 3) * std::f32::consts::TAU, 0.0, 0.0, 0.0]);
    }
    let source = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Rocks"), contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE })
    };
    let rocks = n * n;
    let instances = || {
        ComputeBuffer::from_external("Rocks", source.clone(), BufferType::Storage).with_vertex_layout(
            32,
            vec![InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 }, InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32 }],
        )
    };
    let culling = |near: f32, far: f32| InstanceCulling::new(source.clone(), rocks, 32, 0, 1.2).with_radius_scale(12).with_lod_range(near, far);

    let mut scene = Scene::new();
    let mut ground = Renderable::new(PlaneGeometry::new(extent * 2.0, extent * 2.0), {
        let mut m = Material::new("Ground", GROUND_WGSL, vec![], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
        m.options.cull_mode = kansei_core::materials::CullMode::None;
        m
    });
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    let mut modes: [Vec<usize>; 3] = Default::default();
    // clusters
    let mut r = Renderable::new(InstancedGeometry::new(geometry.clone(), rocks, vec![instances()]), material());
    r.instance_culling = Some(culling(0.0, f32::INFINITY));
    r.clusters = Some(ClusterLod::new(mesh.clone()).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), rotation: None }));
    modes[0].push(scene.add(SceneNode::Renderable(r)));
    // discrete LODs cut from the graph at the same budget, for the largest rock (1.3)
    let ppr = height as f32 / (2.0 * (45f32.to_radians() / 2.0).tan());
    for (k, &near) in BANDS.iter().enumerate() {
        let far = BANDS.get(k + 1).copied().unwrap_or(f32::INFINITY);
        let lod = lod_mesh(&mesh, near.max(0.5), 1.3, ppr, tau);
        let mut r = Renderable::new(InstancedGeometry::new(lod, rocks, vec![instances()]), material());
        r.instance_culling = Some(culling(near, far));
        modes[1].push(scene.add(SceneNode::Renderable(r)));
    }
    // the full mesh
    let mut r = Renderable::new(InstancedGeometry::new(geometry, rocks, vec![instances()]), material());
    r.instance_culling = Some(culling(0.0, f32::INFINITY));
    modes[2].push(scene.add(SceneNode::Renderable(r)));

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(1.0);
        ToneMapEffect::new(options)
    };
    let effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(tonemap)];
    let volume = PostProcessingVolume::new(&renderer, effects);
    let mut camera = Camera::new(45.0, 0.1, 1000.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    let mode = MODES.iter().position(|&m| Some(m) == query_param("mode").as_deref()).unwrap_or(0);
    let bench = query_param("bench").and_then(|b| MODES.iter().position(|&m| m == b).filter(|&m| m != 0)).map(|other| Bench { other, start: now_ms(), sums: [(0.0, 0, 0.0, 0); 2], last_frame: now_ms(), report: None });
    let timer = FrameTimer::new(renderer.device(), renderer.queue());
    log::info!("Kansei — Cluster LOD (WASM) ready: {rocks} rocks of {triangles} triangles, {} clusters over {} levels built in {build_ms:.0} ms", mesh.clusters.len(), mesh.levels().len());

    let mut state = State { renderer, scene, camera, volume, timer, bench, modes, mode, rocks, triangles, build_ms, start: now_ms(), frame: 0, last_frame: now_ms(), interval_ms: 0.0, gpu_ms: 0.0 };
    set_mode(&mut state, mode);
    let state = Rc::new(RefCell::new(state));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut guard = state.borrow_mut();
            let st = &mut *guard;
            if let Some((mode, _)) = st.bench.as_ref().and_then(|b| b.phase(now_ms())) {
                if mode != st.mode {
                    set_mode(st, mode);
                }
            }
            let t = ((now_ms() - st.start) / 1000.0) as f32;
            place_camera(&mut st.camera, extent, t);
            st.timer.begin();
            st.renderer.render_with_postprocessing(&mut st.scene, &mut st.camera, &mut st.volume);
            st.timer.end();

            let now = now_ms();
            st.interval_ms += (now - st.last_frame - st.interval_ms) * 0.05;
            st.last_frame = now;
            let gpu = st.timer.take();
            for ms in &gpu {
                st.gpu_ms += (ms - st.gpu_ms) * 0.05;
            }
            if let Some(bench) = st.bench.as_mut() {
                let was_done = bench.report.is_some();
                bench.record(&gpu, now);
                if let (false, Some(report)) = (was_done, &bench.report) {
                    log::info!("{report}");
                    set_text("bench", report);
                }
            }
            st.frame += 1;
            if st.frame.is_multiple_of(10) {
                let (w, h) = st.renderer.render_size();
                set_text(
                    "hud",
                    &format!(
                        "{} · {} rocks of {} triangles · graph built in {:.0} ms · budget {} px\n{w} x {h} · GPU {:.2} ms · {:.1} ms between frames",
                        MODES[st.mode], st.rocks, st.triangles, st.build_ms, st.renderer.cluster_error_threshold(), st.gpu_ms, st.interval_ms
                    ),
                );
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

const GROUND_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32> };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    return VOut(projection_matrix * view_matrix * world_matrix * position);
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    let albedo = vec3<f32>(0.2, 0.22, 0.16);
    return FOut(vec4<f32>(albedo * 1.6, 1.0), vec4<f32>(0.0), vec4<f32>(0.5, 1.0, 0.5, 1.0), vec4<f32>(albedo, 1.0));
}
"#;
```

- [ ] **Step 4: Build it**

Run: `cd rust/kansei-wasm/examples/cluster-lod && wasm-pack build --target web --release 2>&1 | tail -3`
Expected: `Your wasm pkg is ready to publish`.

- [ ] **Step 5: See it, then measure it** (headless Chrome, own session, a free port)

```bash
cd rust/kansei-wasm/examples/cluster-lod && (python3 -m http.server 8737 >/dev/null 2>&1 &)
export CHROME_DEVTOOLS_AXI_SESSION=cluster-lod CHROME_DEVTOOLS_AXI_CHROME_ARGS="--enable-unsafe-webgpu --enable-gpu --ignore-gpu-blocklist --disable-gpu-vsync --disable-frame-rate-limit"
chrome-devtools-axi open "http://localhost:8737/www/index.html?size=1280x720"   # screenshot: rocks drawn, no holes
chrome-devtools-axi open "http://localhost:8737/www/index.html?size=1280x720&bench=lods"   # ~30 s, read #bench
chrome-devtools-axi open "http://localhost:8737/www/index.html?size=1280x720&bench=full"
chrome-devtools-axi open about:blank && chrome-devtools-axi stop; kill the server
```

Expected:
- **Stills:** the rocks look the same in `clusters`, `lods` and `full` at 1 px.
- **Bench:** clusters take less GPU time than `full`, by a wide margin. Against `lods` the result is recorded whatever it is: it is the honest comparison, discrete LODs cut at the same budget.
- If clusters lose to `lods`, `set_profiling` shows whether the cull or the draw costs. That goes into the PR as a finding for M3/M5, not a fix here.
- Keep one still (`screenshot-clusters.jpg`) in the example directory for the PR.

- [ ] **Step 6: Commit**

```bash
git add rust/Cargo.toml rust/kansei-wasm/examples/cluster-lod
git commit -m "feat(examples): cluster-lod, a field of rocks with an in-page A/B against discrete LODs and the full mesh"
```

---

### Task 7: Docs, and the PR

**Files:**
- Modify: `docs/plans/2026-09-28-cluster-lod-design.md`, `AGENTS.md`

- [ ] **Step 1: The design doc.**
  - Under "2. Selecting and culling on the GPU", add an "As built (M2)" paragraph: the level window instead of the prefix sum, one mesh buffer, the cone switch, the draw list's capacity, and the stats in M3.
  - Add the measured numbers from Task 6 to "Risks: vertex pulling cost".
- [ ] **Step 2: `AGENTS.md`**, one line under Sharp edges: "Cluster LOD (`Renderable::clusters`) draws only the camera's pass until M3: other views draw the renderable's geometry, which must be the mesh its graph was built from."
- [ ] **Step 3: The whole suite, clippy, wasm**

Run: `cd rust && cargo test -p kansei-core 2>&1 | tail -3 && cargo clippy -p kansei-core --all-targets 2>&1 | grep -c "^warning\|^error"; cargo build -p kansei-core --target wasm32-unknown-unknown 2>&1 | tail -1`
Expected: all pass, no new clippy warnings in the files touched, wasm `Finished`.

- [ ] **Step 4: Commit and open the PR** against `development` (`fm/kansei-cluster-lod-m2`), with the bench numbers and the still (absolute pinned image URL).

```bash
git add docs AGENTS.md
git commit -m "docs(clusters): M2 as built"
```
