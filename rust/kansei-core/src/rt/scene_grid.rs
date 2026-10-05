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
    /// Rebuild every frame, not only when the box moved or what it holds changed (`dynamic`
    /// renderables in it rebuild it every frame anyway).
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
    /// Sources gathered at the last rebuild (a renderable's mesh or cut each).
    pub sources: u32,
    /// Whether this frame rebuilt it, and how many frames have.
    pub rebuilt: bool,
    pub rebuilds: u64,
    /// CPU time of the last rebuild's recording, ms.
    pub cpu_ms: f64,
}

/// A renderable's mesh for the gather: its vertex and index counts when made, its bounds, its
/// buffer and triangles.
struct SceneMesh {
    counts: (u32, u32),
    min: Vec3,
    max: Vec3,
    buffer: wgpu::Buffer,
    triangles: u32,
}

/// The renderer's ray tracing grid (`Renderer::enable_rt_grid`): an `RtGrid` round the camera
/// holding the triangles of the renderables with `Renderable::rt`, read with `rt::RT_GRID_WGSL`
/// from any compute pass after the frame's culling (`grid().bind_group_entries`).
///
/// Its box is one more cull view: `InstanceCulling` compacts the instances meeting it (by
/// `rt_lod_range`, the camera's bands by default), and a cluster view cuts cluster LOD
/// renderables there (orthographic, `cluster_error_cells` cells of error). The gather then reads
/// on the GPU each clustered renderable's draw list, each instanced one's culled records (times
/// its mesh, placed by `Renderable::rt_placement`), and each single mesh whose world box meets the
/// grid's (one box test per renderable on the CPU). It rebuilds when the box moves, when the
/// static renderables in it change (transform, visibility, surface), when a cut it gathered grew,
/// on `invalidate`, and every frame while a `dynamic` one is in it.
pub struct SceneRtGrid {
    options: SceneRtGridOptions,
    grid: RtGrid,
    meshes: HashMap<usize, SceneMesh>,
    key: Option<Vec<u32>>,
    force: bool,
    rebuild: bool,
    stats: SceneRtGridStats,
    /// renderables left out (no CPU geometry, or instances with no placement), warned once
    warned: HashSet<usize>,
}

impl SceneRtGrid {
    pub(crate) fn new(device: &wgpu::Device, options: SceneRtGridOptions) -> Self {
        Self { grid: RtGrid::new(device, options.grid), options, meshes: HashMap::new(), key: None, force: true, rebuild: false, stats: SceneRtGridStats::default(), warned: HashSet::new() }
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

    /// Plan the frame, before the culling: follow `eye`, and rebuild if the box moved, `key`
    /// (what the static renderables in the grid are made of) changed, a dynamic one is in it,
    /// the buffers grew, or it was forced.
    pub(crate) fn plan(&mut self, device: &wgpu::Device, eye: Vec3, key: Vec<u32>, any_dynamic: bool) {
        self.grid.poll_readback(device);
        let moved = self.grid.follow(eye);
        let changed = self.key.as_ref() != Some(&key);
        self.key = Some(key);
        let s = self.grid.stats();
        let overflowed = s.triangles > s.triangle_capacity || s.references > s.reference_capacity;
        self.rebuild = moved || changed || any_dynamic || overflowed || self.options.rebuild_every_frame || std::mem::take(&mut self.force);
        self.stats.rebuilt = self.rebuild;
    }

    pub(crate) fn rebuilding(&self) -> bool {
        self.rebuild
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

    /// Rebuild the grid from `scene`'s renderables with `rt`, as culled and cut for cull view
    /// `slot` (the box), into one submit, and read back what it needed.
    pub(crate) fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, slot: usize) {
        let t0 = crate::profiling::now_ms();
        self.stats.rebuilds += 1;
        let (lo, hi) = self.grid.bounds();
        let eps = Vec3::splat(self.options.grid.effective_epsilon());
        // the meshes of what isn't cut by cluster LOD (single meshes only where they meet the box)
        let mut chosen: Vec<(usize, bool)> = Vec::new();
        for idx in scene.ordered_indices() {
            let Some(r) = scene.get_renderable(idx) else { continue };
            if r.rt.is_none() || !r.visible || !r.geometry.initialized {
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
                chosen.push((idx, true));
                continue;
            }
            if r.geometry.vertices.is_empty() || r.geometry.indices.is_empty() {
                if self.warned.insert(idx) {
                    log::warn!("rt grid: renderable {idx} ({}) has no CPU geometry: left out", r.geometry.label);
                }
                continue;
            }
            let counts = (r.geometry.vertex_count(), r.geometry.index_count());
            if self.meshes.get(&idx).is_none_or(|m| m.counts != counts) {
                let mesh = RtMesh::from_geometry(&r.geometry);
                self.meshes.insert(idx, SceneMesh { counts, min: mesh.min, max: mesh.max, buffer: mesh.create_buffer(device), triangles: mesh.triangle_count() });
            }
            let m = &self.meshes[&idx];
            if r.geometry.instance_buffers.is_empty() {
                let (a, b) = transform_box(&r.world_matrix.to_glam(), m.min, m.max);
                if !(a.cmple(hi + eps).all() && b.cmpge(lo - eps).all()) {
                    continue;
                }
            }
            chosen.push((idx, false));
        }
        let mut sources = Vec::new();
        for &(idx, clustered) in &chosen {
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
            } else {
                let m = &self.meshes[&idx];
                (GatherMesh::Mesh { buffer: &m.buffer, triangles: m.triangles }, count)
            };
            sources.push(GatherSource {
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/RtGrid") });
        self.grid.begin(device, queue, &mut encoder);
        self.grid.gather_sources(device, queue, &mut encoder, &sources);
        self.grid.finish(&mut encoder);
        queue.submit(Some(encoder.finish()));
        self.grid.read_back(device, queue);
        self.stats.sources = sources.len() as u32;
        self.stats.cpu_ms = crate::profiling::now_ms() - t0;
    }

    /// Wait for the readback in flight and take it (native; for tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_readback(&mut self, device: &wgpu::Device) {
        self.grid.wait_readback(device);
    }
}
