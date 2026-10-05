use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use super::instance_culling::ARGS_BYTES;

/// What became of the instances culled for one view in one frame, summed over the renderables
/// that draw in it (`Renderer::culling_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullStats {
    /// Instances culled: each renderable's whole list.
    pub tested: u32,
    /// Outside their renderable's LOD band.
    pub lod_culled: u32,
    /// In the band, outside the view's frustum.
    pub frustum_culled: u32,
    /// In view but hidden behind the rest of the scene (the camera, with occlusion culling).
    pub occlusion_culled: u32,
    /// Drawn (in either phase, with occlusion culling).
    pub drawn: u32,
    /// Triangles drawn: each draw's instances times its mesh's, and each cluster cut's
    /// (`Renderable::clusters`) triangles.
    pub triangles: u64,
    /// Clusters drawn by the renderables with cluster LOD.
    pub clusters: u32,
}

impl std::ops::AddAssign for CullStats {
    fn add_assign(&mut self, o: Self) {
        self.tested += o.tested;
        self.lod_culled += o.lod_culled;
        self.frustum_culled += o.frustum_culled;
        self.occlusion_culled += o.occlusion_culled;
        self.drawn += o.drawn;
        self.triangles += o.triangles;
        self.clusters += o.clusters;
    }
}

/// A view the renderer culls instances for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullViewKind {
    Camera,
    /// A layer of the spot shadow atlas.
    SpotShadow(u32),
    /// A planar reflection, by index.
    Reflection(u32),
    /// A cascade of the sun's shadows.
    Cascade(u32),
    /// The top-down view of the sky occlusion (`Renderer::enable_sky_occlusion`).
    SkyOcclusion,
    /// A region the voxel clipmap voxelizes (`Renderer::enable_voxel_clipmap`), by job slot.
    VoxelGi(u32),
    /// The ray tracing grid's box (`Renderer::enable_rt_grid`).
    RtGrid,
}

/// Instance culling statistics of a recent frame, per view.
#[derive(Clone, Debug, Default)]
pub struct CullingStats {
    /// The camera's frame number (`Camera::frame`) they are from.
    pub frame: u32,
    /// Every view culled that frame, in the renderer's order.
    pub views: Vec<(CullViewKind, CullStats)>,
}

impl CullingStats {
    pub fn view(&self, kind: CullViewKind) -> Option<CullStats> {
        self.views.iter().find(|(k, _)| *k == kind).map(|(_, s)| *s)
    }

    /// The camera's view (zero if nothing was culled for it).
    pub fn camera(&self) -> CullStats {
        self.view(CullViewKind::Camera).unwrap_or_default()
    }
}

/// Reads the culled draws' counters back asynchronously: at most one copy in flight, collected
/// when mapped (a few frames later), so the frame never waits for the GPU.
pub(crate) struct StatsReadback {
    pub enabled: bool,
    kinds: Vec<CullViewKind>,
    entries: Vec<Entry>,
    pending: Option<Pending>,
    staging: Option<wgpu::Buffer>,
    latest: Option<CullingStats>,
}

struct Entry {
    view: usize,
    tested: u32,
    args: wgpu::Buffer,
    offset: u64,
    kind: EntryKind,
}

/// What an entry's 32 bytes are.
#[derive(Clone, Copy, PartialEq)]
enum EntryKind {
    /// An instance draw (`ARGS_BYTES`); `clustered` when its renderable draws cluster cuts
    /// instead, whose triangles count for it.
    Instances { clustered: bool },
    /// A cluster cut's draw (`clusters::DRAW_ARGS_BYTES`).
    Clusters,
}

impl EntryKind {
    /// Its bytes in the readback.
    fn bytes(self) -> u64 {
        match self {
            Self::Instances { .. } => ARGS_BYTES,
            Self::Clusters => crate::clusters::DRAW_ARGS_BYTES,
        }
    }
}

struct Pending {
    frame: u32,
    kinds: Vec<CullViewKind>,
    // (view, tested, kind) one after another in the staging buffer (`EntryKind::bytes` each)
    entries: Vec<(usize, u32, EntryKind)>,
    // MAPPING, then MAPPED or FAILED
    state: Arc<AtomicU8>,
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

impl StatsReadback {
    pub fn new() -> Self {
        Self { enabled: false, kinds: Vec::new(), entries: Vec::new(), pending: None, staging: None, latest: None }
    }

    pub fn latest(&self) -> Option<&CullingStats> {
        self.latest.as_ref().filter(|_| self.enabled)
    }

    /// Start a frame culling `kinds` (by view index): collect a finished readback.
    pub fn begin_frame(&mut self, device: &wgpu::Device, kinds: Vec<CullViewKind>) {
        self.entries.clear();
        self.kinds = kinds;
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending.is_some() {
            device.poll(wgpu::Maintain::Poll);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let Some(pending) = self.pending.take_if(|p| p.state.load(Ordering::Acquire) != MAPPING) else { return };
        if pending.state.load(Ordering::Acquire) == FAILED {
            return;
        }
        let staging = self.staging.as_ref().unwrap();
        {
            let bytes = staging.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            let mut views: Vec<(CullViewKind, CullStats)> = Vec::new();
            let mut at = 0;
            for &(view, tested, entry) in &pending.entries {
                let a = &words[at..][..entry.bytes() as usize / 4];
                at += entry.bytes() as usize / 4;
                let kind = pending.kinds[view];
                let stats = match entry {
                    // (the draw's index count, then its instance count)
                    EntryKind::Instances { clustered } => {
                        let triangles = if clustered { 0 } else { a[0] as u64 / 3 * a[1] as u64 };
                        CullStats { tested, drawn: a[1], lod_culled: a[5], frustum_culled: a[6], occlusion_culled: a[7], triangles, clusters: 0 }
                    }
                    // (the clusters listed, then their triangles)
                    EntryKind::Clusters => CullStats { clusters: a[7], triangles: a[9] as u64, ..Default::default() },
                };
                match views.iter_mut().find(|(k, _)| *k == kind) {
                    Some((_, s)) => *s += stats,
                    None => views.push((kind, stats)),
                }
            }
            views.sort_by_key(|(kind, _)| pending.kinds.iter().position(|k| k == kind));
            self.latest = Some(CullingStats { frame: pending.frame, views });
        }
        staging.unmap();
    }

    /// A draw culled this frame for view `view`: `tested` instances, counted into `args` at
    /// `offset`. `clustered`: the renderable draws cluster cuts instead (`record_clusters`).
    pub fn record(&mut self, view: usize, tested: u32, args: &wgpu::Buffer, offset: u64, clustered: bool) {
        if self.enabled {
            self.entries.push(Entry { view, tested, args: args.clone(), offset, kind: EntryKind::Instances { clustered } });
        }
    }

    /// A cluster cut made this frame for view `view`, its draw in `args`.
    pub fn record_clusters(&mut self, view: usize, args: &wgpu::Buffer) {
        if self.enabled {
            self.entries.push(Entry { view, tested: 0, args: args.clone(), offset: 0, kind: EntryKind::Clusters });
        }
    }

    /// After the frame's culling is submitted: copy its counters for reading, unless a copy is
    /// still in flight.
    pub fn end_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: u32) {
        if !self.enabled || self.pending.is_some() || self.entries.is_empty() {
            return;
        }
        let size: u64 = self.entries.iter().map(|e| e.kind.bytes()).sum();
        if self.staging.as_ref().is_none_or(|s| s.size() < size) {
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("InstanceCulling/StatsReadback"),
                size: size.next_power_of_two(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let staging = self.staging.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("InstanceCulling/Stats") });
        let mut at = 0;
        for entry in &self.entries {
            encoder.copy_buffer_to_buffer(&entry.args, entry.offset, staging, at, entry.kind.bytes());
            at += entry.kind.bytes();
        }
        queue.submit(Some(encoder.finish()));
        let state = Arc::new(AtomicU8::new(MAPPING));
        let done = state.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| done.store(if result.is_ok() { MAPPED } else { FAILED }, Ordering::Release));
        let entries = self.entries.drain(..).map(|e| (e.view, e.tested, e.kind)).collect();
        self.pending = Some(Pending { frame, kinds: std::mem::take(&mut self.kinds), entries, state });
    }
}
