use bytemuck::{Pod, Zeroable};

use super::volume::VolumeLayout;
use crate::cameras::Camera;
use crate::renderers::SharedLayouts;

pub(crate) const VOXEL_WRITE_WGSL: &str = include_str!("shaders/voxel_write.wgsl");
pub(crate) const VOXEL_FRAGMENT_WGSL: &str = concat!(include_str!("shaders/voxel_write.wgsl"), include_str!("shaders/voxel_fragment.wgsl"));

/// u32 per voxel in a voxelizer's surface buffers: average albedo (rgb8 + count), average
/// normal (xyz8 + count), average normal folded onto one hemisphere (xyz8 + count: the axis of a
/// sheet thinner than a voxel, whose two faces' normals cancel), brightest emission (RGB9E5).
pub const SURFACE_WORDS_PER_VOXEL: u64 = 4;

/// A renderable's surface in voxel GI (`Renderable::gi`): the albedo it reflects and the light it
/// emits (scene radiance, cd/m²), the same over the whole renderable. For a textured surface,
/// use its mean colour, or give its material a `voxel_fragment_entry` that reads the texture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiSurface {
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
}

impl GiSurface {
    pub fn new(albedo: [f32; 3]) -> Self {
        Self { albedo, emission: [0.0; 3] }
    }

    pub fn with_emission(mut self, emission: [f32; 3]) -> Self {
        self.emission = emission;
        self
    }
}

/// The WGSL `KanseiVoxelizeParams` (voxel_write.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VoxelizeParamsGpu {
    clip_to_voxel: [f32; 16],
    view_dir: [f32; 3],
    _pad0: f32,
    viewport: [f32; 2],
    _pad1: [f32; 2],
    dims: [u32; 3],
    _pad2: u32,
}

/// The WGSL `KanseiVoxelDraw`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VoxelDrawGpu {
    albedo: [f32; 3],
    _pad0: f32,
    emission: [f32; 3],
    _pad1: f32,
}

/// Which surface buffer a voxelization pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceSet {
    /// Renderables that don't move: voxelized again only when one of them changes.
    Static,
    /// `Renderable::dynamic` ones: cleared and voxelized every frame.
    Dynamic,
}

/// Meshes into voxels through the rasterizer (miaumiau.cat/?p=1457's voxelization, without its
/// CPU triangle splitting or its "big triangle" pass): every GI renderable is drawn three times,
/// with an orthographic camera along x, y and z over the volume and a pixel per voxel, through its
/// material's own `vertex_main` (instancing, skinning and vertex animation voxelize as they
/// draw), and a fragment stage that writes the voxel it lands in with storage atomics
/// (`VOXEL_WRITE_WGSL`): the engine's, with the renderable's constant `GiSurface`, or the
/// material's `voxel_fragment_entry`.
///
/// Each voxel keeps its surfaces' average albedo and normal and their brightest emission
/// (`SURFACE_WORDS_PER_VOXEL`). Renderables that don't move go into the static buffer, again only
/// when one of them changes; `Renderable::dynamic` ones into the dynamic buffer, every frame.
/// `SceneVoxelGi` lights both into its volume.
pub struct MeshVoxelizer {
    id: u64,
    layout: VolumeLayout,
    bgl: wgpu::BindGroupLayout,
    fragment: wgpu::ShaderModule,
    cameras: [Camera; 3],
    params: [wgpu::Buffer; 3],
    draws: wgpu::Buffer,
    draw_stride: u64,
    draw_capacity: u64,
    static_surfaces: wgpu::Buffer,
    dynamic_surfaces: Option<wgpu::Buffer>,
    static_groups: Vec<wgpu::BindGroup>,
    dynamic_groups: Vec<wgpu::BindGroup>,
    target: wgpu::TextureView,
    /// What the static surfaces hold (`static_key`), once voxelized.
    static_key: Option<Vec<u32>>,
}

impl MeshVoxelizer {
    /// The dummy target's format and samples: masked off, it only sets the fragments' coverage.
    pub(crate) const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
    pub(crate) const SAMPLE_COUNT: u32 = 4;

    /// A voxelizer over `layout`, whose cameras' group 1 also holds `light_buf` (the renderer's).
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, shared: &SharedLayouts, light_buf: &wgpu::Buffer, layout: VolumeLayout) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let fragment_visible = wgpu::ShaderStages::FRAGMENT;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VoxelGI/VoxelizeBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 100,
                    visibility: fragment_visible,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 101,
                    visibility: fragment_visible,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<VoxelDrawGpu>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 102,
                    visibility: fragment_visible,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelGI/VoxelFragment"), source: wgpu::ShaderSource::Wgsl(VOXEL_FRAGMENT_WGSL.into()) });

        // the three cameras: orthographic over the volume along x, y and z, one pixel per voxel
        let lo = glam::Vec3::from(layout.origin);
        let extent = glam::UVec3::from(layout.dims).as_vec3() * layout.voxel_size;
        let centre = lo + extent * 0.5;
        let world_to_voxel = glam::Mat4::from_scale(glam::Vec3::splat(1.0 / layout.voxel_size)) * glam::Mat4::from_translation(-lo);
        let [dx, dy, dz] = layout.dims;
        // (looking along, up, viewport width and height in voxels, extents across, depth)
        let axes = [
            (glam::Vec3::NEG_X, glam::Vec3::Y, [dz, dy], [extent.z, extent.y], extent.x),
            (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z, [dx, dz], [extent.x, extent.z], extent.y),
            (glam::Vec3::NEG_Z, glam::Vec3::Y, [dx, dy], [extent.x, extent.y], extent.z),
        ];
        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        let mut cameras: [Camera; 3] = std::array::from_fn(|_| Camera::new(60.0, 0.1, 100.0, 1.0));
        let params: [wgpu::Buffer; 3] = std::array::from_fn(|_| uniform("VoxelGI/VoxelizeParams", std::mem::size_of::<VoxelizeParamsGpu>() as u64));
        for (a, (look, up, viewport, across, depth)) in axes.into_iter().enumerate() {
            let eye = centre - look * (depth * 0.5);
            let view = glam::Mat4::look_at_rh(eye, centre, up);
            let projection = glam::Mat4::orthographic_rh(-across[0] * 0.5, across[0] * 0.5, -across[1] * 0.5, across[1] * 0.5, 0.0, depth);
            let camera = &mut cameras[a];
            camera.gpu_initialize(device, &shared.camera_bgl, light_buf);
            camera.view_matrix = view.into();
            camera.inverse_view_matrix = view.inverse().into();
            camera.projection_matrix = projection.into();
            camera.upload(queue);
            let gpu = VoxelizeParamsGpu {
                clip_to_voxel: (world_to_voxel * (projection * view).inverse()).to_cols_array(),
                view_dir: look.to_array(),
                _pad0: 0.0,
                viewport: [viewport[0] as f32, viewport[1] as f32],
                _pad1: [0.0; 2],
                dims: layout.dims,
                _pad2: 0,
            };
            queue.write_buffer(&params[a], 0, bytemuck::bytes_of(&gpu));
        }

        let side = dx.max(dy).max(dz);
        let target = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VoxelGI/VoxelizeTarget"),
                size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: Self::SAMPLE_COUNT,
                dimension: wgpu::TextureDimension::D2,
                format: Self::TARGET_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let draw_stride = (std::mem::size_of::<VoxelDrawGpu>() as u64).next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as u64);
        let static_surfaces = Self::surface_buffer(device, &layout, "VoxelGI/StaticSurfaces");
        let mut voxelizer = Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            layout,
            bgl,
            fragment,
            cameras,
            params,
            draws: uniform("VoxelGI/Draws", draw_stride * 16),
            draw_stride,
            draw_capacity: 16,
            static_surfaces,
            dynamic_surfaces: None,
            static_groups: Vec::new(),
            dynamic_groups: Vec::new(),
            target,
            static_key: None,
        };
        voxelizer.rebuild_groups(device);
        voxelizer
    }

    fn surface_buffer(device: &wgpu::Device, layout: &VolumeLayout, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: layout.voxel_count() * SURFACE_WORDS_PER_VOXEL * 4,
            // (COPY_SRC: readable in tests)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn rebuild_groups(&mut self, device: &wgpu::Device) {
        let draw_size = wgpu::BufferSize::new(std::mem::size_of::<VoxelDrawGpu>() as u64);
        let groups = |surfaces: &wgpu::Buffer| -> Vec<wgpu::BindGroup> {
            self.params
                .iter()
                .map(|params| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("VoxelGI/VoxelizeBG"),
                        layout: &self.bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 100, resource: params.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 101, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.draws, offset: 0, size: draw_size }) },
                            wgpu::BindGroupEntry { binding: 102, resource: surfaces.as_entire_binding() },
                        ],
                    })
                })
                .collect()
        };
        self.static_groups = groups(&self.static_surfaces);
        self.dynamic_groups = self.dynamic_surfaces.as_ref().map(groups).unwrap_or_default();
    }

    /// Tells voxelizers apart in the materials' pipeline caches.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Group 3 of the voxelization pipelines.
    pub(crate) fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }

    /// The engine's fragment stage (`voxel_fragment`).
    pub(crate) fn fragment_module(&self) -> &wgpu::ShaderModule {
        &self.fragment
    }

    pub fn layout(&self) -> &VolumeLayout {
        &self.layout
    }

    /// The static renderables' voxels: `SURFACE_WORDS_PER_VOXEL` u32 per voxel, x fastest.
    pub fn static_surfaces(&self) -> &wgpu::Buffer {
        &self.static_surfaces
    }

    /// The dynamic renderables' voxels this frame (none until one is voxelized).
    pub fn dynamic_surfaces(&self) -> Option<&wgpu::Buffer> {
        self.dynamic_surfaces.as_ref()
    }

    /// Voxelize the static renderables again next frame (they are otherwise redone only when
    /// one of them changes its transform, visibility, geometry or surface).
    pub fn invalidate(&mut self) {
        self.static_key = None;
    }

    /// Upload the draws' constant surfaces (one slot each, in draw order), growing their buffer.
    /// True when the bind groups were rebuilt.
    pub(crate) fn write_draws(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surfaces: &[GiSurface]) -> bool {
        if surfaces.is_empty() {
            return false;
        }
        let mut grown = false;
        if surfaces.len() as u64 > self.draw_capacity {
            self.draw_capacity = (surfaces.len() as u64).next_power_of_two();
            self.draws = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelGI/Draws"),
                size: self.draw_stride * self.draw_capacity,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebuild_groups(device);
            grown = true;
        }
        let mut data = vec![0u8; (self.draw_stride * surfaces.len() as u64) as usize];
        for (k, s) in surfaces.iter().enumerate() {
            let draw = VoxelDrawGpu { albedo: s.albedo, _pad0: 0.0, emission: s.emission, _pad1: 0.0 };
            let at = k * self.draw_stride as usize;
            data[at..at + std::mem::size_of::<VoxelDrawGpu>()].copy_from_slice(bytemuck::bytes_of(&draw));
        }
        queue.write_buffer(&self.draws, 0, &data);
        grown
    }

    /// The dynamic offset of draw `k`'s surface.
    pub(crate) fn draw_offset(&self, k: usize) -> u32 {
        (k as u64 * self.draw_stride) as u32
    }

    /// Whether the static surfaces must be voxelized again for `key` (a description of the
    /// static renderables: see `Renderer::run_voxel_gi`); remembers it.
    pub(crate) fn static_changed(&mut self, key: Vec<u32>) -> bool {
        if self.static_key.as_ref() == Some(&key) {
            return false;
        }
        self.static_key = Some(key);
        true
    }

    /// Make the dynamic buffer (the first frame a dynamic renderable is voxelized).
    pub(crate) fn ensure_dynamic(&mut self, device: &wgpu::Device) {
        if self.dynamic_surfaces.is_none() {
            self.dynamic_surfaces = Some(Self::surface_buffer(device, &self.layout, "VoxelGI/DynamicSurfaces"));
            self.rebuild_groups(device);
        }
    }

    /// Record clearing `set`'s buffer.
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder, set: SurfaceSet) {
        let buffer = match set {
            SurfaceSet::Static => Some(&self.static_surfaces),
            SurfaceSet::Dynamic => self.dynamic_surfaces.as_ref(),
        };
        if let Some(buffer) = buffer {
            encoder.clear_buffer(buffer, 0, None);
        }
    }

    /// Begin the voxelization pass of `axis` (0 x, 1 y, 2 z) into `set`: the axis' camera bound
    /// as group 1 and its viewport set. Bind group 3 per draw with `group(axis, set)` and
    /// `draw_offset`.
    pub(crate) fn begin_pass<'e>(&self, encoder: &'e mut wgpu::CommandEncoder, axis: usize) -> wgpu::RenderPass<'e> {
        let label = ["VoxelGI/VoxelizeX", "VoxelGI/VoxelizeY", "VoxelGI/VoxelizeZ"][axis];
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
        let [dx, dy, dz] = self.layout.dims;
        let (w, h) = [(dz, dy), (dx, dz), (dx, dy)][axis];
        pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
        pass.set_bind_group(1, self.cameras[axis].bind_group(), &[]);
        pass
    }

    /// Group 3 for `axis` into `set`.
    pub(crate) fn group(&self, axis: usize, set: SurfaceSet) -> &wgpu::BindGroup {
        match set {
            SurfaceSet::Static => &self.static_groups[axis],
            SurfaceSet::Dynamic => &self.dynamic_groups[axis],
        }
    }
}
