use bytemuck::{Pod, Zeroable};
use glam::{IVec3, UVec3};

use super::clipmap::{ClipmapLayout, VoxelClipmap};
use super::voxelize::{axis_views, next_voxelizer_id, voxelize_bind_group_layout, voxelize_target, GiSurface, VoxelDrawGpu, VoxelizeParamsGpu, VOXEL_FRAGMENT_WGSL};
use crate::cameras::Camera;
use crate::renderers::SharedLayouts;

const CLEAR_WGSL: &str = include_str!("shaders/clipmap_clear.wgsl");

/// u32 per voxel in a clipmap level's surface buffers: `SURFACE_WORDS_PER_VOXEL`'s, then the
/// surface's area in the voxel (voxel faces, fixed point 1/256), which makes its opacity.
pub const CLIP_SURFACE_WORDS: u64 = super::voxelize::SURFACE_WORDS_PER_VOXEL + 1;

/// Pixels a clipmap voxelization draws along each side of a voxel face, single-sampled
/// (voxel_write.wgsl).
const PIXELS_PER_VOXEL: u32 = 2;

/// A box of a clipmap level's lattice: voxels `lo .. lo + size` of level `level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipRegion {
    pub level: u32,
    pub lo: IVec3,
    pub size: UVec3,
}

impl ClipRegion {
    /// The region's world box.
    pub fn bounds(&self, layout: &ClipmapLayout) -> (glam::Vec3, glam::Vec3) {
        let s = layout.level_voxel_size(self.level);
        (self.lo.as_vec3() * s, (self.lo + self.size.as_ivec3()).as_vec3() * s)
    }

    /// Voxels in it.
    pub fn voxel_count(&self) -> u64 {
        self.size.as_u64vec3().element_product()
    }
}

/// The WGSL `ClearRegion` (clipmap_clear.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClearRegionGpu {
    lo: [i32; 3],
    words: u32,
    size: [u32; 3],
    _pad0: u32,
    dims: [u32; 3],
    _pad1: u32,
}

/// The three axis cameras and parameters that voxelize a region (`ClipRegion`): uploaded when
/// the region changes.
pub(crate) struct RegionView {
    cameras: [Camera; 3],
    params: [wgpu::Buffer; 3],
    viewports: [[u32; 2]; 3],
    region: Option<ClipRegion>,
}

impl RegionView {
    fn new(device: &wgpu::Device, shared: &SharedLayouts, light_buf: &wgpu::Buffer) -> Self {
        let cameras = std::array::from_fn(|_| {
            let mut camera = Camera::new(60.0, 0.1, 100.0, 1.0);
            camera.gpu_initialize(device, &shared.camera_bgl, light_buf);
            camera
        });
        let params = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelClipmap/VoxelizeParams"),
                size: std::mem::size_of::<VoxelizeParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Self { cameras, params, viewports: [[0; 2]; 3], region: None }
    }

    /// Point the cameras at `region` of a clipmap laid out as `layout` (written if it changed).
    fn set(&mut self, queue: &wgpu::Queue, layout: &ClipmapLayout, region: ClipRegion) {
        if self.region == Some(region) {
            return;
        }
        let size = layout.level_voxel_size(region.level);
        for (a, axis) in axis_views(region.lo.as_vec3() * size, size, region.size.to_array()).into_iter().enumerate() {
            axis.apply(&mut self.cameras[a]);
            self.cameras[a].upload(queue);
            let viewport = axis.viewport.map(|v| v * PIXELS_PER_VOXEL);
            let gpu = VoxelizeParamsGpu::new(axis.clip_to_voxel, axis.look, viewport, layout.dims, region.lo.to_array(), region.size.to_array(), CLIP_SURFACE_WORDS as u32, PIXELS_PER_VOXEL);
            queue.write_buffer(&self.params[a], 0, bytemuck::bytes_of(&gpu));
            self.viewports[a] = viewport;
        }
        self.region = Some(region);
    }

    /// The view-projection of one of its axes: a box frustum round the region.
    pub(crate) fn view_proj(&self) -> glam::Mat4 {
        self.cameras[0].projection_matrix.to_glam() * self.cameras[0].view_matrix.to_glam()
    }
}

/// Which of a clipmap level's surface buffers a pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClipSurfaces {
    /// The renderables that don't move, voxelized a region at a time (`ClipmapVoxelizer::job`).
    Static,
    /// `Renderable::dynamic` ones, cleared and voxelized over the whole window every frame.
    Dynamic,
}

/// The scene's meshes into a voxel clipmap's levels (`SceneVoxelClipmap`): `MeshVoxelizer`'s
/// raster voxelization (each renderable drawn along x, y and z through its own `vertex_main`, a
/// pixel per voxel, its surface written with storage atomics by `VOXEL_WRITE_WGSL`) over a region
/// of a level's window at a time, each voxel stored in its toroidal texel.
///
/// Static renderables go into each level's static buffer by jobs: the slab a level's window
/// moved into, or a whole window when it is first filled or invalidated; the region is cleared
/// first. Dynamic ones go into the dynamic buffers of the finest `dynamic_levels` levels, whole
/// windows, every frame.
pub struct ClipmapVoxelizer {
    id: u64,
    layout: ClipmapLayout,
    bgl: wgpu::BindGroupLayout,
    fragment: wgpu::ShaderModule,
    target: wgpu::TextureView,
    draws: wgpu::Buffer,
    draw_stride: u64,
    draw_capacity: u64,
    static_surfaces: Vec<wgpu::Buffer>,
    dynamic_surfaces: Vec<Option<wgpu::Buffer>>,
    /// Per job slot: its cameras and its region's clear.
    jobs: Vec<(RegionView, wgpu::Buffer)>,
    /// Per dynamic level: its cameras over the level's window.
    dynamic_views: Vec<RegionView>,
    clear_pipeline: wgpu::ComputePipeline,
    clear_bgl: wgpu::BindGroupLayout,
    /// What the static surfaces hold (`static_changed`).
    static_key: Option<Vec<u32>>,
    device: wgpu::Device,
}

impl ClipmapVoxelizer {
    /// A voxelizer for `layout`, with `job_slots` regions voxelized a frame at most and dynamic
    /// renderables in the finest `dynamic_levels` levels; its cameras' group 1 also holds
    /// `light_buf` (the renderer's).
    pub(crate) fn new(device: &wgpu::Device, shared: &SharedLayouts, light_buf: &wgpu::Buffer, layout: ClipmapLayout, job_slots: u32, dynamic_levels: u32) -> Self {
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelClipmap/VoxelFragment"), source: wgpu::ShaderSource::Wgsl(VOXEL_FRAGMENT_WGSL.into()) });
        let [dx, dy, dz] = layout.dims;
        let surfaces = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: layout.voxel_count() * CLIP_SURFACE_WORDS * 4,
                // (COPY_SRC: readable in tests)
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let draw_stride = (std::mem::size_of::<VoxelDrawGpu>() as u64).next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as u64);
        let compute = wgpu::ShaderStages::COMPUTE;
        let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelClipmap/ClearBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: compute, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: compute, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelClipmap/Clear"), source: wgpu::ShaderSource::Wgsl(CLEAR_WGSL.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelClipmap/Clear"), bind_group_layouts: &[&clear_bgl], push_constant_ranges: &[] });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelClipmap/Clear"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dynamic_levels = dynamic_levels.min(layout.levels);
        Self {
            id: next_voxelizer_id(),
            layout,
            bgl: voxelize_bind_group_layout(device),
            fragment,
            target: voxelize_target(device, dx.max(dy).max(dz) * PIXELS_PER_VOXEL, 1),
            draws: uniform("VoxelClipmap/Draws", draw_stride * 16),
            draw_stride,
            draw_capacity: 16,
            static_surfaces: (0..layout.levels).map(|_| surfaces("VoxelClipmap/StaticSurfaces")).collect(),
            dynamic_surfaces: (0..dynamic_levels).map(|_| None).collect(),
            jobs: (0..job_slots.max(1)).map(|_| (RegionView::new(device, shared, light_buf), uniform("VoxelClipmap/ClearParams", std::mem::size_of::<ClearRegionGpu>() as u64))).collect(),
            dynamic_views: (0..dynamic_levels).map(|_| RegionView::new(device, shared, light_buf)).collect(),
            clear_pipeline,
            clear_bgl,
            static_key: None,
            device: device.clone(),
        }
    }

    /// Tells voxelizers apart in the materials' pipeline caches.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Group 3 of the voxelization pipelines.
    pub(crate) fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }

    /// The engine's fragment stage: its module and entry (`voxel_fragment`).
    pub(crate) fn fragment(&self) -> (&wgpu::ShaderModule, &'static str) {
        (&self.fragment, "voxel_fragment")
    }

    /// Samples of its passes' target: one (it draws `PIXELS_PER_VOXEL` squared pixels a voxel
    /// face instead).
    pub(crate) fn sample_count(&self) -> u32 {
        1
    }

    pub fn layout(&self) -> &ClipmapLayout {
        &self.layout
    }

    /// Level `level`'s static surfaces: `CLIP_SURFACE_WORDS` u32 per voxel, by texel (x fastest),
    /// each texel holding the voxel of the window congruent to it.
    pub fn static_surfaces(&self, level: u32) -> &wgpu::Buffer {
        &self.static_surfaces[level as usize]
    }

    /// Level `level`'s dynamic surfaces this frame (none until one is voxelized there).
    pub fn dynamic_surfaces(&self, level: u32) -> Option<&wgpu::Buffer> {
        self.dynamic_surfaces.get(level as usize)?.as_ref()
    }

    /// The finest levels dynamic renderables go into.
    pub fn dynamic_levels(&self) -> u32 {
        self.dynamic_surfaces.len() as u32
    }

    /// Regions voxelized a frame at most.
    pub fn job_slots(&self) -> u32 {
        self.jobs.len() as u32
    }

    /// Whether the static surfaces must be voxelized again for `key` (a description of the
    /// static renderables, as `MeshVoxelizer::static_changed`); remembers it.
    pub(crate) fn static_changed(&mut self, key: Vec<u32>) -> bool {
        if self.static_key.as_ref() == Some(&key) {
            return false;
        }
        self.static_key = Some(key);
        true
    }

    /// Upload the draws' constant surfaces (one slot each, in draw order), growing their buffer.
    pub(crate) fn write_draws(&mut self, queue: &wgpu::Queue, surfaces: &[GiSurface]) {
        if surfaces.is_empty() {
            return;
        }
        if surfaces.len() as u64 > self.draw_capacity {
            self.draw_capacity = (surfaces.len() as u64).next_power_of_two();
            self.draws = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelClipmap/Draws"),
                size: self.draw_stride * self.draw_capacity,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mut data = vec![0u8; (self.draw_stride * surfaces.len() as u64) as usize];
        for (k, s) in surfaces.iter().enumerate() {
            let at = k * self.draw_stride as usize;
            data[at..at + std::mem::size_of::<VoxelDrawGpu>()].copy_from_slice(bytemuck::bytes_of(&VoxelDrawGpu::new(s)));
        }
        queue.write_buffer(&self.draws, 0, &data);
    }

    /// The dynamic offset of draw `k`'s surface.
    pub(crate) fn draw_offset(&self, k: usize) -> u32 {
        (k as u64 * self.draw_stride) as u32
    }

    /// Ready job slot `slot` for `region` (its cameras and parameters).
    pub(crate) fn set_job(&mut self, queue: &wgpu::Queue, slot: usize, region: ClipRegion) {
        let layout = self.layout;
        let (view, clear) = &mut self.jobs[slot];
        view.set(queue, &layout, region);
        let gpu = ClearRegionGpu { lo: region.lo.to_array(), words: CLIP_SURFACE_WORDS as u32, size: region.size.to_array(), _pad0: 0, dims: layout.dims, _pad1: 0 };
        queue.write_buffer(clear, 0, bytemuck::bytes_of(&gpu));
    }

    /// Job slot `slot`'s view (its region and frustum).
    pub(crate) fn job_view(&self, slot: usize) -> &RegionView {
        &self.jobs[slot].0
    }

    /// Ready the dynamic views over each dynamic level's window at `origins` (None: the level has
    /// no window yet).
    pub(crate) fn set_dynamic_windows(&mut self, queue: &wgpu::Queue, origins: &[Option<IVec3>]) {
        let layout = self.layout;
        for (level, view) in self.dynamic_views.iter_mut().enumerate() {
            if let Some(Some(lo)) = origins.get(level) {
                view.set(queue, &layout, ClipRegion { level: level as u32, lo: *lo, size: UVec3::from(layout.dims) });
            }
        }
    }

    /// Make the dynamic buffers (the first frame a dynamic renderable is voxelized).
    pub(crate) fn ensure_dynamic(&mut self) {
        let size = self.layout.voxel_count() * CLIP_SURFACE_WORDS * 4;
        for slot in &mut self.dynamic_surfaces {
            if slot.is_none() {
                *slot = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("VoxelClipmap/DynamicSurfaces"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
            }
        }
    }

    /// Record clearing job slot `slot`'s region of its level: its static surfaces and its texels
    /// of the clipmap's radiance (no stale light there before the injection relights it).
    pub(crate) fn encode_clear(&self, encoder: &mut wgpu::CommandEncoder, slot: usize, clipmap: &VoxelClipmap) {
        let (view, params) = &self.jobs[slot];
        let Some(region) = view.region else { return };
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VoxelClipmap/ClearBG"),
            layout: &self.clear_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.static_surfaces[region.level as usize].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(clipmap.view(region.level)) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelClipmap/Clear"), timestamp_writes: None });
        pass.set_pipeline(&self.clear_pipeline);
        pass.set_bind_group(0, &group, &[]);
        let [w, h, d] = region.size.to_array();
        pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
    }

    /// Record clearing every dynamic buffer.
    pub(crate) fn clear_dynamic(&self, encoder: &mut wgpu::CommandEncoder) {
        for buffer in self.dynamic_surfaces.iter().flatten() {
            encoder.clear_buffer(buffer, 0, None);
        }
    }

    /// Group 3 for each axis of a pass into `set`: job slot `index`'s region (static), or dynamic
    /// level `index`'s window.
    pub(crate) fn groups(&self, set: ClipSurfaces, index: usize) -> Option<[wgpu::BindGroup; 3]> {
        let (view, surfaces) = match set {
            ClipSurfaces::Static => {
                let view = &self.jobs[index].0;
                (view, &self.static_surfaces[view.region?.level as usize])
            }
            ClipSurfaces::Dynamic => (&self.dynamic_views[index], self.dynamic_surfaces[index].as_ref()?),
        };
        view.region?;
        let draw_size = wgpu::BufferSize::new(std::mem::size_of::<VoxelDrawGpu>() as u64);
        Some(std::array::from_fn(|a| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VoxelClipmap/VoxelizeBG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 100, resource: view.params[a].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 101, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.draws, offset: 0, size: draw_size }) },
                    wgpu::BindGroupEntry { binding: 102, resource: surfaces.as_entire_binding() },
                ],
            })
        }))
    }

    /// Begin the voxelization pass of `axis` (0 x, 1 y, 2 z) of `set` / `index` (as `groups`):
    /// the axis' camera bound as group 1 and its viewport set.
    pub(crate) fn begin_pass<'e>(&self, encoder: &'e mut wgpu::CommandEncoder, set: ClipSurfaces, index: usize, axis: usize) -> wgpu::RenderPass<'e> {
        let view = match set {
            ClipSurfaces::Static => &self.jobs[index].0,
            ClipSurfaces::Dynamic => &self.dynamic_views[index],
        };
        let label = match set {
            ClipSurfaces::Static => ["VoxelClipmap/VoxelizeX", "VoxelClipmap/VoxelizeY", "VoxelClipmap/VoxelizeZ"][axis],
            ClipSurfaces::Dynamic => ["VoxelClipmap/DynamicX", "VoxelClipmap/DynamicY", "VoxelClipmap/DynamicZ"][axis],
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Discard },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: crate::profiling::gpu_pass(label).as_ref().map(crate::profiling::PassStamp::render),
            ..Default::default()
        });
        let [w, h] = view.viewports[axis];
        pass.set_viewport(0.0, 0.0, w.max(1) as f32, h.max(1) as f32, 0.0, 1.0);
        pass.set_bind_group(1, view.cameras[axis].bind_group(), &[]);
        pass
    }

    /// Bytes of the surface buffers.
    pub fn memory_bytes(&self) -> u64 {
        let buffers = self.static_surfaces.len() + self.dynamic_surfaces.iter().flatten().count();
        buffers as u64 * self.layout.voxel_count() * CLIP_SURFACE_WORDS * 4
    }
}
