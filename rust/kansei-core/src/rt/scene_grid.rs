//! `SceneRtGrid`: the renderer's ray tracing grid, gathered from its renderables on the GPU.

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Vec3};

use super::grid::{GatherCount, GatherMesh, GatherSource, RtGrid, RtGridOptions, RtGridStats, RtPlacement};
use super::mesh::{transform_box, RtMesh};
use crate::objects::{Renderable, Scene};

/// What `Renderer::enable_rt_grid` builds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneRtGridOptions {
    pub grid: RtGridOptions,
    /// The error budget of the cluster cuts gathered into the grid (`Renderable::clusters`), in
    /// cells: 1 keeps every error within a cell (the scout traced the 871k-triangle dragon as
    /// its 29k-triangle cut at one 4.75 cm cell, looking the same).
    pub cluster_error_cells: f32,
    /// Rebuild the static layer every frame, not only when the box moved or what it holds
    /// changed (the `dynamic` renderables' moving layer is rebuilt every frame anyway).
    pub rebuild_every_frame: bool,
}

impl Default for SceneRtGridOptions {
    fn default() -> Self {
        Self { grid: RtGridOptions::default(), cluster_error_cells: 1.0, rebuild_every_frame: false }
    }
}

/// What the grid did, for a stats overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneRtGridStats {
    pub grid: RtGridStats,
    /// Sources gathered at the last rebuild of the static layer (a renderable's mesh or cut each),
    /// and of the moving layer.
    pub sources: u32,
    pub moving_sources: u32,
    /// Whether this frame rebuilt the static layer, and how many frames have.
    pub rebuilt: bool,
    pub rebuilds: u64,
    /// Whether this frame rebuilt the moving layer (a dynamic renderable has `rt`).
    pub moving_rebuilt: bool,
    /// CPU time of the last rebuild's recording, ms.
    pub cpu_ms: f64,
}

/// A renderable's mesh for the gather: its vertex and index counts and geometry revision when
/// made, its bounds, its buffer and triangles.
struct SceneMesh {
    counts: (u32, u32),
    revision: u32,
    min: Vec3,
    max: Vec3,
    buffer: wgpu::Buffer,
    triangles: u32,
}

/// Frames a moving grid rests before the hardware backend rebuilds its BLAS once to trace fast.
const SETTLE_FRAMES: u32 = 30;

/// The renderer's ray tracing grid (`Renderer::enable_rt_grid`): an `RtGrid` round the camera
/// holding the triangles of the renderables with `Renderable::rt`, read with `rt::RT_GRID_WGSL`
/// (or, with the hardware backend, `grid().backend().wgsl()`) from any compute pass after the
/// frame's culling (`grid().bind_group_entries`).
///
/// Its box is one more cull view: `InstanceCulling` compacts the instances meeting it (by
/// `rt_lod_range`, the camera's bands by default), and a cluster view cuts cluster LOD
/// renderables there (orthographic, `cluster_error_cells` cells of error). The gather then reads
/// on the GPU each clustered renderable's draw list, each instanced one's culled records (times
/// its mesh, placed by `Renderable::rt_placement`), and each single mesh whose world box meets the
/// grid's (one box test per renderable on the CPU). The renderables that aren't `dynamic` go into
/// the grid's static layer, which rebuilds when the box moves, when they change (transform,
/// visibility, surface, `Geometry::revision`: a mesh whose CPU vertices were moved, such as a
/// skinned proxy posed on the CPU, or whose `Renderable::rt_mesh` the GPU rewrote; a renderable
/// turning `dynamic` or back), when a cut it gathered grew, and on `invalidate`. The `dynamic` ones
/// go into its moving layer (`RtGrid::begin_moving`), fitted round their world boxes and rebuilt
/// every frame, so what moves costs a frame its own triangles, not the scene's. The hardware
/// backend (`RtGridOptions::backend`) builds the static BLAS fast while the box moves, and once
/// more to trace fast when it has rested `SETTLE_FRAMES` frames; the moving BLAS always fast.
pub struct SceneRtGrid {
    options: SceneRtGridOptions,
    grid: RtGrid,
    meshes: HashMap<usize, SceneMesh>,
    key: Option<Vec<u32>>,
    force: bool,
    /// this frame: rebuild the static layer, rebuild the moving layer, empty the moving layer
    rebuild: bool,
    moving: bool,
    clear_moving: bool,
    stats: SceneRtGridStats,
    /// renderables left out (no CPU geometry, or instances with no placement), warned once
    warned: HashSet<usize>,
    /// frames since a hardware BLAS was built fast, while it stands (None: built to trace fast)
    fast_built: Option<u32>,
}

impl SceneRtGrid {
    pub(crate) fn new(device: &wgpu::Device, options: SceneRtGridOptions) -> Self {
        Self {
            grid: RtGrid::new(device, options.grid),
            options,
            meshes: HashMap::new(),
            key: None,
            force: true,
            rebuild: false,
            moving: false,
            clear_moving: false,
            stats: SceneRtGridStats::default(),
            warned: HashSet::new(),
            fast_built: None,
        }
    }

    pub fn options(&self) -> &SceneRtGridOptions {
        &self.options
    }

    /// Rebuild every frame or only when needed (see `SceneRtGridOptions::rebuild_every_frame`).
    pub fn set_rebuild_every_frame(&mut self, on: bool) {
        self.options.rebuild_every_frame = on;
    }

    /// The grid, for the passes that trace it.
    pub fn grid(&self) -> &RtGrid {
        &self.grid
    }

    /// A handle to the grid's buffers for an effect (`RtReflectionsEffect`): it follows them when
    /// the grid grows.
    pub fn handle(&self) -> super::RtGridHandle {
        self.grid.handle()
    }

    /// Rebuild next frame (after changing what the renderer can't see, such as a GPU-written
    /// instance buffer of a renderable that isn't `dynamic`).
    pub fn invalidate(&mut self) {
        self.force = true;
    }

    pub fn stats(&self) -> SceneRtGridStats {
        SceneRtGridStats { grid: self.grid.stats(), ..self.stats }
    }

    /// Bytes on the GPU: the grid's and the renderables' meshes.
    pub fn memory_bytes(&self) -> u64 {
        self.grid.memory_bytes() + self.meshes.values().map(|m| m.buffer.size()).sum::<u64>()
    }

    /// Plan the frame, before the culling: follow `eye`; rebuild the static layer if the box
    /// moved, `key` (what the static renderables in the grid are made of) changed, its buffers
    /// grew, or it was forced; and the moving layer while a dynamic renderable has `rt`.
    pub(crate) fn plan(&mut self, device: &wgpu::Device, eye: Vec3, key: Vec<u32>, any_dynamic: bool) {
        self.grid.poll_readback(device);
        let moved = self.grid.follow(eye);
        let changed = self.key.as_ref() != Some(&key);
        self.key = Some(key);
        let overflowed = self.grid.overflowed();
        let moving = moved || self.options.rebuild_every_frame;
        self.rebuild = moving || changed || overflowed || std::mem::take(&mut self.force);
        self.moving = any_dynamic;
        self.clear_moving = !any_dynamic && self.grid.stats().moving_cells > 0;
        if self.grid.backend() == super::RtBackend::Hardware {
            // a BLAS built fast while moving, rebuilt once to trace fast when it has rested
            let settle = self.fast_built.is_some_and(|f| f >= SETTLE_FRAMES);
            self.rebuild |= settle;
            self.grid.set_prefer_fast_build(moving);
            self.fast_built = match (self.rebuild, moving) {
                (true, true) => Some(0),
                (true, false) => None,
                (false, _) => self.fast_built.map(|f| f + 1),
            };
        }
        self.stats.rebuilt = self.rebuild;
        self.stats.moving_rebuilt = self.moving;
    }

    pub(crate) fn rebuilding(&self) -> bool {
        self.rebuild || self.moving || self.clear_moving
    }

    /// The cull view of the box: an orthographic view along -z whose frustum is the box.
    pub(crate) fn cull_view_proj(&self) -> Mat4 {
        let (lo, hi) = self.grid.bounds();
        Mat4::orthographic_rh(lo.x, hi.x, lo.y, hi.y, -hi.z, -lo.z)
    }

    /// The cluster view of the box: orthographic, a pixel a cell, the cut's errors within
    /// `cluster_error_cells`.
    pub(crate) fn cluster_view(&self) -> crate::clusters::ClusterViewGpu {
        let (lo, hi) = self.grid.bounds();
        crate::clusters::ClusterViewGpu::new(self.cull_view_proj(), (lo + hi) * 0.5, 1.0 / self.options.grid.cell, 0.01, self.options.cluster_error_cells, true)
    }

    /// A cut gathered from the box's view grew: rebuild next frame too (its clusters past the old
    /// list were missing).
    pub(crate) fn cut_grew(&mut self) {
        self.force = true;
    }

    /// Where a renderable's records put its mesh: its `rt_placement`, its cluster LOD's
    /// transform, or nowhere (a single mesh); None when it is instanced and neither says.
    fn placement(r: &Renderable) -> Option<RtPlacement> {
        if let Some(p) = &r.rt_placement {
            return Some(p.clone());
        }
        if let Some(t) = r.clusters.as_ref().and_then(|c| c.transform) {
            return Some(RtPlacement::Instance(t));
        }
        r.geometry.instance_buffers.is_empty().then_some(RtPlacement::None)
    }

    /// Rebuild the grid's layers this frame's plan asked for from `scene`'s renderables with `rt`,
    /// as culled and cut for cull view `slot` (the box), into one submit, and read back what they
    /// needed.
    pub(crate) fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, slot: usize) {
        let t0 = crate::profiling::now_ms();
        if self.rebuild {
            self.stats.rebuilds += 1;
        }
        let (lo, hi) = self.grid.bounds();
        let eps = Vec3::splat(self.options.grid.effective_epsilon());
        // the meshes of what isn't cut by cluster LOD (single meshes only where they meet the box),
        // whether each is clustered and dynamic, and the dynamic ones' world box (None: not known
        // on the CPU, a cut or instances)
        let mut chosen: Vec<(usize, bool, bool)> = Vec::new();
        let mut moving_box: Option<(Vec3, Vec3)> = None;
        let mut moving_unbounded = false;
        let mut grow = |dynamic: bool, b: Option<(Vec3, Vec3)>| {
            if dynamic {
                match b {
                    Some((a, b)) => moving_box = Some(moving_box.map_or((a, b), |(lo, hi)| (lo.min(a), hi.max(b)))),
                    None => moving_unbounded = true,
                }
            }
        };
        for idx in scene.ordered_indices() {
            let Some(r) = scene.get_renderable(idx) else { continue };
            if r.rt.is_none() || !r.visible || !r.geometry.initialized {
                continue;
            }
            let dynamic = r.dynamic;
            if !(if dynamic { self.moving } else { self.rebuild }) {
                continue;
            }
            let cut = r.clusters.as_ref().and_then(|c| c.gpu.as_ref()).is_some_and(|g| g.cut(slot as u32).is_some());
            if Self::placement(r).is_none() {
                if self.warned.insert(idx) {
                    log::warn!("rt grid: renderable {idx} ({}) is instanced with no rt_placement: left out", r.geometry.label);
                }
                continue;
            }
            if cut {
                chosen.push((idx, true, dynamic));
                grow(dynamic, None);
                continue;
            }
            if let Some(m) = &r.rt_mesh {
                // written on the GPU: its own buffer and bounds
                let (a, b) = transform_box(&r.world_matrix.to_glam(), m.min, m.max);
                if a.cmple(hi + eps).all() && b.cmpge(lo - eps).all() {
                    chosen.push((idx, false, dynamic));
                    grow(dynamic, Some((a, b)));
                }
                continue;
            }
            if r.geometry.vertices.is_empty() || r.geometry.indices.is_empty() {
                if self.warned.insert(idx) {
                    log::warn!("rt grid: renderable {idx} ({}) has no CPU geometry: left out", r.geometry.label);
                }
                continue;
            }
            let counts = (r.geometry.vertex_count(), r.geometry.index_count());
            // read again when its counts or its vertices (`Geometry::revision`) changed
            let revision = r.geometry.revision;
            if self.meshes.get(&idx).is_none_or(|m| m.counts != counts || m.revision != revision) {
                let mesh = RtMesh::from_geometry(&r.geometry);
                self.meshes.insert(idx, SceneMesh { counts, revision, min: mesh.min, max: mesh.max, buffer: mesh.create_buffer(device), triangles: mesh.triangle_count() });
            }
            let m = &self.meshes[&idx];
            if r.geometry.instance_buffers.is_empty() {
                let (a, b) = transform_box(&r.world_matrix.to_glam(), m.min, m.max);
                if !(a.cmple(hi + eps).all() && b.cmpge(lo - eps).all()) {
                    continue;
                }
                grow(dynamic, Some((a, b)));
            } else {
                grow(dynamic, None);
            }
            chosen.push((idx, false, dynamic));
        }
        let moving_box = if moving_unbounded { None } else { moving_box };
        // the static layer's sources, then the moving layer's
        let mut sources = [Vec::new(), Vec::new()];
        for &(idx, clustered, dynamic) in &chosen {
            let r = scene.get_renderable(idx).unwrap();
            let first = r.geometry.instance_buffers.first();
            let stride = first.and_then(|cb| cb.vertex_layout()).map_or(0, |l| l.stride as u32);
            let culled = r.instance_culling.as_ref().filter(|_| first.is_some()).and_then(|c| c.view(slot).map(|d| (c, d)));
            // the instance records the renderable's culling left for the box, or all of them
            let (records, first_record, count) = match (first, culled) {
                (None, _) => (None, 0, GatherCount::Fixed(1)),
                (Some(_), Some((c, draw))) => (
                    Some((draw.instances, stride)),
                    (draw.instances_offset / c.culled_stride() as u64) as u32,
                    GatherCount::Word { args: draw.args, word: (draw.offset / 4) as u32 + 1, at_most: c.count },
                ),
                (Some(cb), None) => match cb.gpu_buffer() {
                    Some(buffer) => (Some((buffer, stride)), 0, GatherCount::Fixed(r.geometry.instance_count)),
                    None => continue,
                },
            };
            let (mesh, count) = if clustered {
                let gpu = r.clusters.as_ref().unwrap().gpu.as_ref().unwrap();
                let cut = gpu.cut(slot as u32).unwrap();
                let entries = GatherCount::Word { args: cut.args(), word: crate::clusters::CLAIMED_WORD as u32, at_most: cut.capacity() };
                (GatherMesh::Clusters { buffer: gpu.mesh_buffer(), draws: cut.draws() }, entries)
            } else if let Some(m) = &r.rt_mesh {
                (GatherMesh::Mesh { buffer: &m.buffer, triangles: m.triangles }, count)
            } else {
                let m = &self.meshes[&idx];
                (GatherMesh::Mesh { buffer: &m.buffer, triangles: m.triangles }, count)
            };
            sources[dynamic as usize].push(GatherSource {
                mesh,
                records,
                first_record,
                count,
                world: r.world_matrix.to_glam(),
                placement: Self::placement(r).unwrap().wgsl(),
                surface: r.rt.unwrap(),
                id: idx as u32,
            });
        }
        let [static_sources, moving_sources] = sources;
        if self.clear_moving || (self.moving && moving_sources.is_empty()) {
            // (no dynamic renderable in the box)
            self.moving = false;
            self.grid.clear_moving(queue);
        }
        if !self.rebuild && !self.moving {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/RtGrid") });
        if self.rebuild {
            self.grid.begin(device, queue, &mut encoder);
            self.grid.gather_sources(device, queue, &mut encoder, &static_sources);
            self.grid.finish(&mut encoder);
            self.stats.sources = static_sources.len() as u32;
        }
        if self.moving {
            self.grid.begin_moving(device, queue, &mut encoder, moving_box);
            // room for every triangle the meshes may place, so none is lost while a readback is
            // in flight (cuts: what the readback says)
            let most: Option<u64> = moving_sources
                .iter()
                .map(|s| match (&s.mesh, &s.count) {
                    (GatherMesh::Mesh { triangles, .. }, GatherCount::Fixed(n)) => Some(*triangles as u64 * *n as u64),
                    (GatherMesh::Mesh { triangles, .. }, GatherCount::Word { at_most, .. }) => Some(*triangles as u64 * *at_most as u64),
                    (GatherMesh::Clusters { .. }, _) => None,
                })
                .sum();
            if let Some(n) = most {
                self.grid.reserve_triangles(device, queue, &mut encoder, n.min(u32::MAX as u64) as u32);
            }
            self.grid.gather_sources(device, queue, &mut encoder, &moving_sources);
            self.grid.finish(&mut encoder);
            self.stats.moving_sources = moving_sources.len() as u32;
        }
        queue.submit(Some(encoder.finish()));
        self.grid.read_back(device, queue);
        self.stats.cpu_ms = crate::profiling::now_ms() - t0;
    }

    /// Wait for the readback in flight and take it (native; for tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_readback(&mut self, device: &wgpu::Device) {
        self.grid.wait_readback(device);
    }
}
