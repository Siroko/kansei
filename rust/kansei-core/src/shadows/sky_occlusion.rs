use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;

const BUILD_WGSL: &str = include_str!("../shaders/sky_occlusion_build.wgsl");

/// Options of `Renderer::enable_sky_occlusion`.
#[derive(Debug, Clone, Copy)]
pub struct SkyOcclusionOptions {
    /// Side of the square area around the camera it covers, metres.
    pub extent_m: f32,
    /// Texels per side of the depth map the canopy is seen in from above.
    pub resolution: u32,
    /// Voxels of the visibility volume: per side, and in height.
    pub volume_size: (u32, u32),
    /// World heights the volume spans; the ground and the canopy should lie inside.
    pub min_height_m: f32,
    pub max_height_m: f32,
    /// How far the camera may move before the map is rebuilt around it, as a share of the extent.
    pub recenter: f32,
    /// How much a ray is dimmed per metre through canopy that covers its whole footprint.
    pub canopy_extinction: f32,
    /// Frames the volume's build is spread over, after the top-down pass (a share of it on each).
    /// Materials read the previous volume until the new one is done.
    pub frames: u32,
    /// Tiles per side the top-down pass is split into, one drawn per frame, each culled to its
    /// part of the view (1: the whole map in one frame). A rebuild then takes `depth_tiles`
    /// squared frames of the top-down pass, one for the pyramid and `frames` for the volume, and
    /// no frame carries the whole top-down pass.
    pub depth_tiles: u32,
    /// Scales the camera distance `InstanceCulling` picks LOD bands by, for the top-down view:
    /// above 1 it draws coarser LODs, whose detail the map's texels rarely resolve, but it also
    /// drops instances beyond a last LOD band that ends at a finite distance sooner.
    pub lod_distance_scale: f32,
    /// The layers (`Renderable::layers`) whose shadow casters occlude the sky; all by default.
    /// Leave solid ground out (put the vegetation on a layer of its own): the volume counts what
    /// is under a top as inside it, so where it is interpolated across the ground the voxels
    /// below it darken the ground's surface.
    pub layer_mask: u32,
}

impl Default for SkyOcclusionOptions {
    fn default() -> Self {
        Self {
            extent_m: 160.0,
            resolution: 1024,
            volume_size: (128, 16),
            min_height_m: -20.0,
            max_height_m: 60.0,
            recenter: 0.125,
            canopy_extinction: 0.3,
            frames: 4,
            depth_tiles: 2,
            lod_distance_scale: 1.0,
            layer_mask: u32::MAX,
        }
    }
}

/// The WGSL `SkyOcclusionParams` (sky_occlusion.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkyOcclusionParamsGpu {
    center: [f32; 2],
    inv_extent: f32,
    min_y: f32,
    inv_height: f32,
    enabled: f32,
    _pad: [f32; 2],
}

/// The WGSL `OcclusionBuild` (sky_occlusion_build.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct OcclusionBuildGpu {
    center: [f32; 2],
    extent: f32,
    min_y: f32,
    max_y: f32,
    eye_y: f32,
    near: f32,
    far: f32,
    extinction: f32,
    levels: u32,
    level: u32,
    first_layer: u32,
}

/// Sky occlusion around the camera (`Renderer::enable_sky_occlusion`): how much of the sky each
/// point sees past the canopy, as Lumen occludes Unreal's sky light under trees.
///
/// The renderer draws the shadow casters on `SkyOcclusionOptions::layer_mask`'s layers
/// (alpha-tested foliage included, through its shadow fragment) into a depth map from straight
/// above, culled on the GPU like a shadow cascade. From it the build makes a pyramid of the
/// canopy's cover and height, then a low-resolution volume of sky visibility: from each voxel, 24
/// cosine-weighted directions cone-traced through the canopy. It rebuilds only when the camera has
/// moved far enough (`SkyOcclusionOptions::recenter`), or on `refresh`, spread over frames: the
/// top-down pass a tile a frame (`depth_tiles`), the pyramid, the volume a slab a frame
/// (`frames`); the scene is taken to stand still in between. Materials read it with
/// `SKY_OCCLUSION_WGSL`'s `skyVisibility` and dim their sky ambient light by it.
///
/// The canopy is seen as a height field: its top, and how much of each area it covers. What is
/// under a crown is taken to be inside it, so rays that would slip beneath a neighbouring crown
/// count as dimmed.
pub struct SkyOcclusion {
    /// `resolution`, `volume_size` and `frames` are fixed when it is created; the others apply
    /// from the next rebuild.
    pub options: SkyOcclusionOptions,
    /// The visibility volume (rgba8unorm, r the visibility), for `skyVisibility`.
    pub volume: wgpu::TextureView,
    /// Its placement (uniform `SkyOcclusionParams`); off (visibility 1) until the first build.
    pub params: wgpu::Buffer,
    volume_texture: wgpu::Texture,
    /// The volume being built, copied into `volume` once complete.
    back: wgpu::Texture,
    back_view: wgpu::TextureView,
    depth: wgpu::TextureView,
    camera: Camera,
    pyramid: wgpu::Texture,
    pyramid_views: Vec<wgpu::TextureView>,
    /// The build's parameters, a slot per pyramid level then one per slab of the volume, written
    /// at once when a rebuild starts
    build_params: wgpu::Buffer,
    /// Layers of the volume built per frame
    slab: u32,
    sampler: wgpu::Sampler,
    top_pipeline: wgpu::ComputePipeline,
    top_bgl: wgpu::BindGroupLayout,
    down_pipeline: wgpu::ComputePipeline,
    down_bgl: wgpu::BindGroupLayout,
    volume_pipeline: wgpu::ComputePipeline,
    volume_bgl: wgpu::BindGroupLayout,
    /// Where the map was last built; while it is being rebuilt, the tile of the top-down pass due
    /// this frame, whether the pyramid (and the volume's first slab) is, and the volume's next
    /// layer.
    center: Option<glam::Vec2>,
    tile: Option<u32>,
    pyramid_due: bool,
    next_layer: Option<u32>,
}

impl SkyOcclusion {
    /// The depth format and bias of the top-down pass: the cascades', so materials share pipelines.
    pub(crate) const FORMAT: wgpu::TextureFormat = crate::shadows::CascadedShadowMap::FORMAT;
    pub(crate) const DEPTH_BIAS: wgpu::DepthBiasState = crate::shadows::CascadedShadowMap::DEPTH_BIAS;
    const NEAR: f32 = 1.0;
    /// Bytes per slot of `build_params` (WebGPU's uniform offset alignment).
    const SLOT: u64 = 256;

    pub(crate) fn new(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout, light_buf: &wgpu::Buffer, options: SkyOcclusionOptions) -> Self {
        let res = options.resolution.clamp(16, 8192).next_power_of_two();
        let texture = |label: &str, size: wgpu::Extent3d, dimension, format, mips: u32, usage| {
            device.create_texture(&wgpu::TextureDescriptor { label: Some(label), size, mip_level_count: mips, sample_count: 1, dimension, format, usage, view_formats: &[] })
        };
        let depth = texture(
            "SkyOcclusion/Depth",
            wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            wgpu::TextureDimension::D2,
            Self::FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default());
        let levels = res.ilog2() + 1;
        let pyramid = texture(
            "SkyOcclusion/Pyramid",
            wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            levels,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let pyramid_views = (0..levels)
            .map(|l| pyramid.create_view(&wgpu::TextureViewDescriptor { base_mip_level: l, mip_level_count: Some(1), ..Default::default() }))
            .collect();
        let (side, height) = (options.volume_size.0.max(2), options.volume_size.1.max(2));
        let volume_size = wgpu::Extent3d { width: side, height, depth_or_array_layers: side };
        let volume_texture = texture(
            "SkyOcclusion/Volume",
            volume_size,
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        );
        let volume = volume_texture.create_view(&Default::default());
        let back = texture("SkyOcclusion/VolumeBuild", volume_size, wgpu::TextureDimension::D3, wgpu::TextureFormat::Rgba8Unorm, 1, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let back_view = back.create_view(&Default::default());
        let uniform = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
        };
        // a slot per dispatch's parameters: one rewritten per dispatch would hold only the last
        // write when the passes run
        let slab = height.div_ceil(options.frames.clamp(1, height));
        let slots = (levels + height.div_ceil(slab)) as usize;
        let build_params = uniform("SkyOcclusion/Build", slots * Self::SLOT as usize);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let uniform_entry = entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None });
        let texture_entry = |binding, sample_type, dim| entry(binding, wgpu::BindingType::Texture { sample_type, view_dimension: dim, multisampled: false });
        let storage_entry = |binding, format, dim| entry(binding, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: dim });
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let top_bgl = bgl(
            "SkyOcclusion/TopBGL",
            &[uniform_entry, texture_entry(1, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2), storage_entry(3, wgpu::TextureFormat::Rgba16Float, wgpu::TextureViewDimension::D2)],
        );
        let down_bgl = bgl(
            "SkyOcclusion/DownBGL",
            &[
                uniform_entry,
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: false }, wgpu::TextureViewDimension::D2),
                storage_entry(3, wgpu::TextureFormat::Rgba16Float, wgpu::TextureViewDimension::D2),
            ],
        );
        let volume_bgl = bgl(
            "SkyOcclusion/VolumeBGL",
            &[
                uniform_entry,
                texture_entry(4, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2),
                entry(5, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                storage_entry(6, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureViewDimension::D3),
            ],
        );
        let pipeline = |label: &str, entry_point: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(BUILD_WGSL.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[layout], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None })
        };
        let mut camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        camera.gpu_initialize(device, camera_bgl, light_buf);
        Self {
            options,
            volume,
            params: uniform("SkyOcclusion/Params", std::mem::size_of::<SkyOcclusionParamsGpu>()),
            volume_texture,
            back,
            back_view,
            depth,
            camera,
            pyramid,
            pyramid_views,
            build_params,
            slab,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SkyOcclusion/Sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            top_pipeline: pipeline("SkyOcclusion/Top", "top", &top_bgl),
            top_bgl,
            down_pipeline: pipeline("SkyOcclusion/Down", "down", &down_bgl),
            down_bgl,
            volume_pipeline: pipeline("SkyOcclusion/Volume", "volume", &volume_bgl),
            volume_bgl,
            center: None,
            tile: None,
            pyramid_due: false,
            next_layer: None,
        }
    }

    /// Rebuild on the next frame (the scene under the map changed).
    pub fn refresh(&mut self) {
        self.center = None;
    }

    /// The visibility volume's texture (for `Texture::from_view` in a material).
    pub fn volume_texture(&self) -> &wgpu::Texture {
        &self.volume_texture
    }

    fn eye_y(&self) -> f32 {
        self.options.max_height_m + Self::NEAR
    }

    fn far(&self) -> f32 {
        self.options.max_height_m - self.options.min_height_m + Self::NEAR
    }

    /// Before culling: whether the camera left the map's middle, and if so the top-down view of
    /// the area around it (snapped to whole texels).
    pub(crate) fn update(&mut self, queue: &wgpu::Queue, camera: &Camera) {
        let eye = camera.inverse_view_matrix.to_glam().w_axis;
        let here = glam::Vec2::new(eye.x, eye.z);
        let o = self.options;
        if !self.center.is_none_or(|c| (here - c).abs().max_element() > o.extent_m * o.recenter) {
            return;
        }
        self.tile = Some(0);
        self.pyramid_due = false;
        self.next_layer = None;
        let texel = o.extent_m / self.pyramid.width() as f32;
        let center = (here / texel).round() * texel;
        self.center = Some(center);
        let eye = glam::Vec3::new(center.x, self.eye_y(), center.y);
        let view = glam::Mat4::look_at_rh(eye, eye - glam::Vec3::Y, glam::Vec3::NEG_Z);
        let half = o.extent_m * 0.5;
        let projection = glam::Mat4::orthographic_rh(-half, half, -half, half, Self::NEAR, self.far());
        self.camera.view_matrix = view.into();
        self.camera.inverse_view_matrix = view.inverse().into();
        self.camera.projection_matrix = projection.into();
        self.camera.upload(queue);
    }

    fn tiles(&self) -> u32 {
        self.options.depth_tiles.clamp(1, self.pyramid.width())
    }

    /// Tile `tile`'s texels of the depth map: x from, x to, y from, y to (rows go down the map).
    fn tile_texels(&self, tile: u32) -> [u32; 4] {
        let (n, res) = (self.tiles(), self.pyramid.width());
        let (x, y) = (tile % n, tile / n);
        [x * res / n, (x + 1) * res / n, y * res / n, (y + 1) * res / n]
    }

    /// Tile `tile`'s part of the view in normalized device coordinates: x from, x to, y from, y
    /// to (y up), along its texels' edges.
    fn tile_ndc(&self, tile: u32) -> [f32; 4] {
        let res = self.pyramid.width() as f32;
        let [x0, x1, y0, y1] = self.tile_texels(tile).map(|t| t as f32 / res * 2.0 - 1.0);
        [x0, x1, -y1, -y0]
    }

    /// The top-down view to cull for this frame, while a tile of it is due: that tile's part.
    pub(crate) fn cull_view(&self) -> Option<glam::Mat4> {
        let [x0, x1, y0, y1] = self.tile_ndc(self.tile?);
        Some(crate::reflections::crop(x0, x1, y0, y1) * self.camera.projection_matrix.to_glam() * self.camera.view_matrix.to_glam())
    }

    /// Whether a tile of the top-down pass is due this frame.
    pub(crate) fn pending(&self) -> bool {
        self.tile.is_some()
    }

    /// Whether this frame's tile is the first of a rebuild (its pass clears the map).
    pub(crate) fn first_tile(&self) -> bool {
        self.tile == Some(0)
    }

    /// This frame's tile of the depth map (x, y, width, height in texels), while one is due.
    pub(crate) fn tile_scissor(&self) -> Option<[u32; 4]> {
        let [x0, x1, y0, y1] = self.tile_texels(self.tile?);
        Some([x0, y0, x1 - x0, y1 - y0])
    }

    /// Whether `build` has work this frame: a tile of the top-down pass, the pyramid, or a slab
    /// of the volume.
    pub(crate) fn building(&self) -> bool {
        self.tile.is_some() || self.pyramid_due || self.next_layer.is_some()
    }

    pub(crate) fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth
    }

    pub(crate) fn camera(&self) -> &Camera {
        &self.camera
    }

    /// After a tile of the top-down pass, on to the next (nothing else that frame); on the frame
    /// after the last, the pyramid and the volume's first slab; on the frames after it, the
    /// volume's next slabs. Once the volume is complete, it replaces the one materials read, with
    /// the parameters that place it (written before this frame's submission, so its materials
    /// read the new volume).
    pub(crate) fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let Some(center) = self.center else { return };
        if let Some(tile) = self.tile {
            let last = self.tiles() * self.tiles() - 1;
            self.tile = (tile < last).then_some(tile + 1);
            self.pyramid_due = tile >= last;
            return;
        }
        if !self.building() {
            return;
        }
        let o = self.options;
        let levels = self.pyramid_views.len() as u32;
        let (side, height) = (self.back.width(), self.back.height());
        let first_layer = if self.pyramid_due { 0 } else { self.next_layer.unwrap_or(0) };
        let slab = self.slab;
        let slot = |k: u32| wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.build_params,
            offset: k as u64 * Self::SLOT,
            size: std::num::NonZeroU64::new(std::mem::size_of::<OcclusionBuildGpu>() as u64),
        });
        // with the pyramid, every dispatch's parameters in one write: the pyramid's levels, then
        // the volume's slabs
        if self.pyramid_due {
            let slabs = height.div_ceil(slab);
            let mut data = vec![0u8; ((levels + slabs) as u64 * Self::SLOT) as usize];
            for k in 0..levels + slabs {
                let gpu = OcclusionBuildGpu {
                    center: center.to_array(),
                    extent: o.extent_m,
                    min_y: o.min_height_m,
                    max_y: o.max_height_m,
                    eye_y: self.eye_y(),
                    near: Self::NEAR,
                    far: self.far(),
                    extinction: o.canopy_extinction.max(0.0),
                    levels,
                    level: k.min(levels),
                    first_layer: k.saturating_sub(levels) * slab,
                };
                let at = (k as u64 * Self::SLOT) as usize;
                data[at..at + std::mem::size_of::<OcclusionBuildGpu>()].copy_from_slice(bytemuck::bytes_of(&gpu));
            }
            queue.write_buffer(&self.build_params, 0, &data);
        }
        let group = |layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<_> = entries.iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("SkyOcclusion/BG"), layout, entries: &entries })
        };
        let tex = wgpu::BindingResource::TextureView;
        let stamp = crate::profiling::gpu_pass("SkyOcclusion/Build");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SkyOcclusion/Build"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        if self.pyramid_due {
            let res = self.pyramid.width();
            let top = group(&self.top_bgl, &[(0, slot(0)), (1, tex(&self.depth)), (3, tex(&self.pyramid_views[0]))]);
            pass.set_pipeline(&self.top_pipeline);
            pass.set_bind_group(0, &top, &[]);
            pass.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), 1);
            for level in 1..levels as usize {
                let bg = group(&self.down_bgl, &[(0, slot(level as u32)), (2, tex(&self.pyramid_views[level - 1])), (3, tex(&self.pyramid_views[level]))]);
                let size = (res >> level).max(1);
                pass.set_pipeline(&self.down_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 1);
            }
        }
        let all = self.pyramid.create_view(&Default::default());
        let volume = group(
            &self.volume_bgl,
            &[(0, slot(levels + first_layer / slab)), (4, tex(&all)), (5, wgpu::BindingResource::Sampler(&self.sampler)), (6, tex(&self.back_view))],
        );
        pass.set_pipeline(&self.volume_pipeline);
        pass.set_bind_group(0, &volume, &[]);
        pass.dispatch_workgroups(side.div_ceil(4), slab.min(height - first_layer).div_ceil(4), side.div_ceil(4));
        drop(pass);
        self.pyramid_due = false;
        let next = first_layer + slab;
        if next < height {
            self.next_layer = Some(next);
            return;
        }
        self.next_layer = None;
        encoder.copy_texture_to_texture(self.back.as_image_copy(), self.volume_texture.as_image_copy(), self.back.size());
        let params = SkyOcclusionParamsGpu {
            center: center.to_array(),
            inv_extent: 1.0 / o.extent_m,
            min_y: o.min_height_m,
            inv_height: 1.0 / (o.max_height_m - o.min_height_m).max(1e-3),
            enabled: 1.0,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> [(&'static str, &'static str); 2] {
        [("sky_occlusion_build", BUILD_WGSL), ("sky_occlusion", super::SKY_OCCLUSION_WGSL)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_validate_and_layouts_match() {
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in SkyOcclusion::shader_sources() {
            let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (_, ty) in module.types.iter() {
                if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                    sizes.insert(n.clone(), *span as usize);
                }
            }
        }
        assert_eq!(sizes["OcclusionBuild"], std::mem::size_of::<OcclusionBuildGpu>());
        assert_eq!(sizes["SkyOcclusionParams"], std::mem::size_of::<SkyOcclusionParamsGpu>());
    }

    /// A rebuild draws the top-down pass a tile a frame, the tiles covering the map once and each
    /// culled to its own part of the view, then the pyramid, then the volume a slab a frame.
    #[test]
    fn a_rebuild_is_spread_over_tiles_then_slabs() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let shared = crate::renderers::SharedLayouts::new(&device);
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4096, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
        let options = SkyOcclusionOptions { extent_m: 64.0, resolution: 256, volume_size: (32, 16), frames: 4, depth_tiles: 3, ..Default::default() };
        let mut sky = SkyOcclusion::new(&device, &shared.camera_bgl, &light_buf, options);
        let mut camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        camera.set_position(5.0, 2.0, -3.0);
        camera.look_at(&crate::math::Vec3::new(5.0, 2.0, -4.0));
        camera.update_view_matrix();
        sky.update(&queue, &camera);
        let full = sky.camera().projection_matrix.to_glam() * sky.camera().view_matrix.to_glam();
        let center = sky.center.unwrap();
        let mut covered = vec![0u8; 256 * 256];
        let mut stages = Vec::new();
        let mut encoder = device.create_command_encoder(&Default::default());
        while sky.building() {
            if let Some([x, y, w, h]) = sky.tile_scissor() {
                assert_eq!(sky.first_tile(), stages.is_empty(), "the first tile clears the map");
                for py in y..y + h {
                    for px in x..x + w {
                        covered[(py * 256 + px) as usize] += 1;
                    }
                }
                // the tile's cull view frames exactly its part of the map: the world point at a
                // texel's centre is inside it for the tile's texels and outside for the others
                let cull = sky.cull_view().unwrap();
                for (tx, ty) in [(x, y), (x + w - 1, y + h - 1), (x + w, y), (x.wrapping_sub(1), y + h - 1)] {
                    if tx >= 256 || ty >= 256 {
                        continue;
                    }
                    let world = glam::Vec3::new(
                        center.x + ((tx as f32 + 0.5) / 256.0 - 0.5) * 64.0,
                        0.0,
                        center.y + ((ty as f32 + 0.5) / 256.0 - 0.5) * 64.0,
                    );
                    let ndc = cull.project_point3(world);
                    let inside = ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0;
                    assert_eq!(inside, tx >= x && tx < x + w && ty >= y && ty < y + h, "texel ({tx}, {ty}) and tile at ({x}, {y})");
                    // and the full view puts that texel where the map has it
                    let full_ndc = full.project_point3(world);
                    let texel = ((full_ndc.x * 0.5 + 0.5) * 256.0, (0.5 - full_ndc.y * 0.5) * 256.0);
                    assert!((texel.0 - (tx as f32 + 0.5)).abs() < 1e-2 && (texel.1 - (ty as f32 + 0.5)).abs() < 1e-2, "texel ({tx}, {ty}) seen at {texel:?}");
                }
                stages.push("tile");
            } else {
                stages.push(if sky.pyramid_due { "pyramid" } else { "slab" });
            }
            sky.build(&device, &queue, &mut encoder);
        }
        queue.submit([encoder.finish()]);
        assert!(covered.iter().all(|&c| c == 1), "the tiles cover the map once");
        assert_eq!(stages, [vec!["tile"; 9], vec!["pyramid"], vec!["slab"; 3]].concat(), "the frames of a rebuild");
    }

    /// A crown of full cover, a disc 5 m across its radius with its top at 10 m, over flat ground:
    /// the volume's visibility follows a fine integration of the same height-field model, dark
    /// under the crown, open beside it and above it.
    #[test]
    fn visibility_follows_the_canopy_height_field() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let shared = crate::renderers::SharedLayouts::new(&device);
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4096, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
        let options = SkyOcclusionOptions { extent_m: 64.0, resolution: 512, volume_size: (64, 32), min_height_m: 0.0, max_height_m: 16.0, ..Default::default() };
        let mut sky = SkyOcclusion::new(&device, &shared.camera_bgl, &light_buf, options);
        let mut camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        camera.set_position(0.0, 2.0, 0.0);
        camera.look_at(&crate::math::Vec3::new(0.0, 2.0, -1.0));
        camera.update_view_matrix();
        sky.update(&queue, &camera);
        assert!(sky.pending());
        let (radius, top) = (5.0f32, 10.0f32);
        let height = |x: f32, z: f32| if x * x + z * z < radius * radius { top } else { 0.0 };
        // the depth map, as the top-down pass would draw it
        let res = options.resolution;
        let (eye_y, far) = (sky.eye_y(), sky.far());
        let mut depths = vec![1.0f32; (res * res) as usize];
        for y in 0..res {
            for x in 0..res {
                let (wx, wz) = (((x as f32 + 0.5) / res as f32 - 0.5) * options.extent_m, ((y as f32 + 0.5) / res as f32 - 0.5) * options.extent_m);
                depths[(y * res + x) as usize] = (eye_y - SkyOcclusion::NEAR - height(wx, wz)) / (far - SkyOcclusion::NEAR);
            }
        }
        let buf = { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&depths), usage: wgpu::BufferUsages::STORAGE }) };
        let code = format!("@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
            @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
            @fragment fn fs(@builtin(position) pos : vec4f) -> @builtin(frag_depth) f32 {{ return d[u32(pos.y) * {res}u + u32(pos.x)]; }}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: SkyOcclusion::FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: sky.depth_view(), depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        // every slab of the volume, as the frames after the top-down pass would
        while sky.building() {
            sky.build(&device, &queue, &mut e);
        }
        let (side, levels) = (options.volume_size.0, options.volume_size.1);
        let row = (side * 4).div_ceil(256) * 256;
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * levels * side) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: sky.volume_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(levels) } },
            wgpu::Extent3d { width: side, height: levels, depth_or_array_layers: side },
        );
        queue.submit([e.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = read.slice(..).get_mapped_range();
        let voxel = |x: u32, y: u32, z: u32| data[((z * levels + y) * row + x * 4) as usize] as f32 / 255.0;
        // the same model integrated finely: cosine-weighted directions, dimmed by 0.3 per metre
        // below the crown's top inside its disc
        let reference = |p: glam::Vec3| -> f32 {
            let n = 4000;
            let mut sum = 0.0;
            for i in 0..n {
                let u = (i as f32 + 0.5) / n as f32;
                let (r, phi) = (u.sqrt(), i as f32 * 2.399963);
                let d = glam::Vec3::new(r * phi.cos(), (1.0 - u).sqrt(), r * phi.sin());
                let mut tau = 0.0;
                let dt = 0.05;
                let mut t = 0.0;
                while t < 80.0 {
                    let q = p + d * (t + 0.5 * dt);
                    if q.y > top { break; }
                    if q.y < height(q.x, q.z) { tau += 0.3 * dt; }
                    t += dt;
                }
                sum += (-tau).exp();
            }
            sum / n as f32
        };
        let at = |x: f32, y: f32, z: f32| {
            let f = |v: f32, span: f32, n: u32| ((v / span + 0.5) * n as f32 - 0.5).round().clamp(0.0, n as f32 - 1.0) as u32;
            let (ix, iz) = (f(x, options.extent_m, side), f(z, options.extent_m, side));
            let iy = ((y - options.min_height_m) / (options.max_height_m - options.min_height_m) * levels as f32 - 0.5).round().clamp(0.0, levels as f32 - 1.0) as u32;
            // the voxel's own centre, for the reference
            let c = glam::Vec3::new(
                ((ix as f32 + 0.5) / side as f32 - 0.5) * options.extent_m,
                options.min_height_m + (iy as f32 + 0.5) / levels as f32 * (options.max_height_m - options.min_height_m),
                ((iz as f32 + 0.5) / side as f32 - 0.5) * options.extent_m,
            );
            (voxel(ix, iy, iz), reference(c))
        };
        for (x, y, z, what) in [
            (0.0, 1.0, 0.0, "under the crown"),
            (3.0, 5.0, 0.0, "inside the crown"),
            (7.0, 1.0, 0.0, "beside it"),
            (15.0, 1.0, 0.0, "in the open"),
            (0.0, 11.0, 0.0, "above it"),
        ] {
            let (got, want) = at(x, y, z);
            eprintln!("{what}: {got:.3} (reference {want:.3})");
            assert!((got - want).abs() < 0.08, "{what}: visibility {got}, reference {want}");
        }
        let (under, _) = at(0.0, 1.0, 0.0);
        let (open, _) = at(15.0, 1.0, 0.0);
        assert!(under < 0.5 && open > 0.9, "under {under}, open {open}");
    }

    /// The build's GPU cost (the pyramid and the volume, without the top-down depth pass), by
    /// wall clock: `cargo test -p kansei-core --lib time_sky_occlusion_build -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_sky_occlusion_build() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let shared = crate::renderers::SharedLayouts::new(&device);
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4096, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
        let camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        let wall = |f: &mut dyn FnMut(&mut wgpu::CommandEncoder)| -> f64 {
            let mut e = device.create_command_encoder(&Default::default());
            f(&mut e);
            let t = std::time::Instant::now();
            queue.submit([e.finish()]);
            device.poll(wgpu::Maintain::Wait);
            t.elapsed().as_secs_f64() * 1e3
        };
        let mut empty: Vec<f64> = (0..60).map(|_| wall(&mut |_| {})).collect();
        empty.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (resolution, volume_size) in [(1024, (128, 16)), (1024, (128, 32)), (1024, (160, 32)), (2048, (256, 32))] {
            let mut sky = SkyOcclusion::new(&device, &shared.camera_bgl, &light_buf, SkyOcclusionOptions { resolution, volume_size, ..Default::default() });
            // ten whole builds per submit, so the work stands well above the submit's own latency
            let mut build = |e: &mut wgpu::CommandEncoder| {
                sky.refresh();
                sky.update(&queue, &camera);
                while sky.building() {
                    sky.build(&device, &queue, e);
                }
            };
            let mut times: Vec<f64> = (0..20).map(|_| (wall(&mut |e| for _ in 0..10 { build(e) }) - empty[0]) / 10.0).skip(3).collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("map {resolution}², volume {volume_size:?}: {:.2} ms min, {:.2} ms median per build (over {} frames)", times[0], times[times.len() / 2], sky.options.frames);
        }
    }
}
