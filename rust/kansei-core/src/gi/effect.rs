use bytemuck::{Pod, Zeroable};

use super::cones::gradient_sky_lighting;
use super::volume::{VoxelGiQuality, VoxelVolume};
use crate::cameras::Camera;
use crate::postprocessing::effects::{ScreenSpaceGIEffect, ScreenSpaceGIOptions};
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

pub(crate) const TRACE_WGSL: &str = concat!(
    include_str!("shaders/screen_common.wgsl"),
    include_str!("shaders/screen_normal.wgsl"),
    include_str!("shaders/voxel_volume.wgsl"),
    include_str!("shaders/voxel_cones.wgsl"),
    include_str!("shaders/voxel_irradiance.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("shaders/screen_trace.wgsl"),
);
pub(crate) const TEMPORAL_WGSL: &str = concat!(include_str!("shaders/screen_common.wgsl"), include_str!("shaders/screen_temporal.wgsl"));
pub(crate) const COMPOSITE_WGSL: &str = concat!(
    include_str!("shaders/screen_common.wgsl"),
    include_str!("shaders/screen_normal.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("shaders/screen_composite.wgsl"),
);

/// What `VoxelGIEffect::new` sets up.
#[derive(Clone, Copy, Debug)]
pub struct VoxelGIOptions {
    /// Resolution of the trace (`Low` a quarter each way, else half) and the cones' steps.
    pub quality: VoxelGiQuality,
    /// Scale of the light added (1 is physical).
    pub intensity: f32,
    /// Voxels out along the normal the cones start from, past the surface's own voxels.
    pub start_voxels: f32,
    /// Metres a cone looks.
    pub max_distance_m: f32,
    /// Weight of each new frame in the accumulated result.
    pub temporal_blend: f32,
    /// How much of the materials' own sky ambient the GI replaces (needs `set_sky_lighting`, and
    /// materials lit by `SKY_LIGHTING_WGSL`'s `skyIrradiance`).
    pub material_ambient: f32,
    /// Scale of the sky past the volume.
    pub sky_scale: f32,
    /// Screen-space GI in front of the voxels (`gi=voxel+ssgi`): it brings the light of what it
    /// sees occlude each direction within its radius, at full screen detail, and the voxels light
    /// the rest of the hemisphere. `None`: the voxels alone.
    pub near_field: Option<ScreenSpaceGIOptions>,
}

impl Default for VoxelGIOptions {
    fn default() -> Self {
        Self {
            quality: VoxelGiQuality::Medium,
            intensity: 1.0,
            start_voxels: 1.5,
            max_distance_m: 1e4,
            temporal_blend: 0.1,
            material_ambient: 1.0,
            sky_scale: 1.0,
            near_field: None,
        }
    }
}

/// The WGSL `VoxelGiParams` (screen_common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct VoxelGiParamsGpu {
    inv_proj: [f32; 16],
    inv_view: [f32; 16],
    view: [f32; 16],
    prev_view_proj: [f32; 16],
    full_size: [f32; 2],
    trace_size: [f32; 2],
    near_size: [f32; 2],
    start_voxels: f32,
    max_distance: f32,
    max_steps: u32,
    frame: u32,
    intensity: f32,
    ambient: f32,
    blend: f32,
    history_valid: u32,
    has_sky: u32,
    debug: u32,
    near_field: u32,
    sky_scale: f32,
    _pad: [u32; 2],
}

struct Targets {
    width: u32,
    height: u32,
    trace: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
}

struct Gpu {
    params: wgpu::Buffer,
    trace: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    temporal: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    composite: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The sky past the volume until `set_sky_lighting`: `sky_gradient`.
    gradient_sky: wgpu::Buffer,
    /// Bound as the near field without one.
    no_near: wgpu::TextureView,
    targets: Option<Targets>,
}

/// Diffuse global illumination on screen from a voxel volume of the scene's light (the
/// renderer's `SceneVoxelGi`, `Renderer::enable_voxel_gi`): per pixel, at reduced resolution,
/// six cones over the hemisphere around the surface's normal gather the light the volume holds
/// (off screen and behind the camera too, bounces included) and the sky past it; a temporal filter
/// smooths the cones' per-frame rotation; the composite adds albedo / pi times that irradiance
/// to the scene. With `near_field`, screen-space GI goes first and the voxels light what it
/// leaves open.
///
/// Add it first in the chain, where `ScreenSpaceGIEffect` would go. `enabled = false` skips it at
/// no cost. It adds light only where a material writes the GBuffer's albedo, and replaces the
/// materials' sky ambient as `ScreenSpaceGIEffect` does.
pub struct VoxelGIEffect {
    pub enabled: bool,
    pub quality: VoxelGiQuality,
    pub intensity: f32,
    pub start_voxels: f32,
    pub max_distance_m: f32,
    pub temporal_blend: f32,
    pub material_ambient: f32,
    pub sky_scale: f32,
    /// Debug view: output only the light the GI adds (albedo / pi times its irradiance), black
    /// elsewhere, in place of the lit image.
    pub show_indirect: bool,
    /// The sky past the volume without `set_sky_lighting`: scene radiance straight up and down.
    pub sky_gradient: ([f32; 3], [f32; 3]),
    near_field: Option<ScreenSpaceGIEffect>,
    volume_view: wgpu::TextureView,
    volume_uniform: wgpu::Buffer,
    volume_sampler: wgpu::Sampler,
    sky_lighting: Option<wgpu::Buffer>,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
    frame: u32,
    gpu: Option<Gpu>,
}

impl VoxelGIEffect {
    /// Read `volume` (`SceneVoxelGi::volume`); the effect keeps its own handles to it.
    pub fn new(volume: &VoxelVolume, options: VoxelGIOptions) -> Self {
        Self {
            enabled: true,
            quality: options.quality,
            intensity: options.intensity,
            start_voxels: options.start_voxels,
            max_distance_m: options.max_distance_m,
            temporal_blend: options.temporal_blend,
            material_ambient: options.material_ambient,
            sky_scale: options.sky_scale,
            show_indirect: false,
            sky_gradient: ([0.0; 3], [0.0; 3]),
            near_field: options.near_field.map(ScreenSpaceGIEffect::new),
            volume_view: volume.view().clone(),
            volume_uniform: volume.uniform().clone(),
            volume_sampler: volume.sampler().clone(),
            sky_lighting: None,
            prev_view_proj: None,
            last_camera_frame: None,
            frame: 0,
            gpu: None,
        }
    }

    /// The sky's lighting (`SkyAtmosphereBindings::sky_lighting`): the light past the volume,
    /// and the materials' ambient the GI replaces. Also the near field's.
    pub fn set_sky_lighting(&mut self, sky_lighting: Option<&wgpu::Buffer>) {
        self.sky_lighting = sky_lighting.cloned();
        if let Some(near) = &mut self.near_field {
            near.set_sky_lighting(sky_lighting);
        }
    }

    /// The screen-space GI in front of the voxels, if any (to tune it).
    pub fn near_field_mut(&mut self) -> Option<&mut ScreenSpaceGIEffect> {
        self.near_field.as_mut()
    }

    /// Drop the accumulated frames; call on camera cuts.
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
        if let Some(near) = &mut self.near_field {
            near.reset_history();
        }
    }

    fn trace_scale(&self) -> f32 {
        match self.quality {
            VoxelGiQuality::Low => 0.25,
            VoxelGiQuality::Medium | VoxelGiQuality::High => 0.5,
        }
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture = |binding, filterable, dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: dimension, multisampled: false },
            count: None,
        };
        let d2 = wgpu::TextureViewDimension::D2;
        let depth = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: false },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: d2 },
            count: None,
        };
        let sampler = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let trace_bgl = bgl(
            "VoxelGI/TraceBGL",
            &[uniform(0), depth(1), texture(2, false, d2), uniform(3), texture(4, true, wgpu::TextureViewDimension::D3), sampler(5), uniform(6), storage(7)],
        );
        let temporal_bgl = bgl("VoxelGI/TemporalBGL", &[uniform(0), texture(1, false, d2), texture(2, true, d2), depth(3), storage(4), sampler(5)]);
        let composite_bgl = bgl(
            "VoxelGI/CompositeBGL",
            &[uniform(0), texture(1, false, d2), depth(2), texture(3, false, d2), texture(4, false, d2), texture(5, false, d2), texture(6, false, d2), uniform(7), storage(8)],
        );
        let pipeline = |label: &str, code: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[layout], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let no_near = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VoxelGI/NoNearField"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        self.gpu = Some(Gpu {
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelGI/ScreenParams"),
                size: std::mem::size_of::<VoxelGiParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            trace: pipeline("VoxelGI/Trace", TRACE_WGSL, &trace_bgl),
            trace_bgl,
            temporal: pipeline("VoxelGI/Temporal", TEMPORAL_WGSL, &temporal_bgl),
            temporal_bgl,
            composite: pipeline("VoxelGI/Composite", COMPOSITE_WGSL, &composite_bgl),
            composite_bgl,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("VoxelGI/Linear"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            gradient_sky: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VoxelGI/GradientSky"),
                size: std::mem::size_of::<super::SkyLightingData>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            no_near,
            targets: None,
        });
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let scale = self.trace_scale();
        let (w, h) = (((width as f32 * scale).ceil() as u32).max(1), ((height as f32 * scale).ceil() as u32).max(1));
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_some_and(|t| t.width == w && t.height == h) {
            return;
        }
        let target = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        gpu.targets = Some(Targets { width: w, height: h, trace: target("VoxelGI/Trace"), history: [target("VoxelGI/HistoryA"), target("VoxelGI/HistoryB")] });
        self.prev_view_proj = None;
    }
}

impl PostProcessingEffect for VoxelGIEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
    }

    fn is_active(&self) -> bool {
        self.enabled
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        // the near field's bounce first (its own trace and history)
        let near = self.near_field.as_mut().map(|n| n.encode_trace(device, queue, encoder, gbuffer, input, depth, camera, width, height));
        self.ensure_targets(device, width, height);
        let camera_frame = camera.frame();
        if self.last_camera_frame.is_some_and(|f| camera_frame != f && camera_frame != f.wrapping_add(1)) {
            self.prev_view_proj = None;
        }
        self.last_camera_frame = Some(camera_frame);
        let gpu = self.gpu.as_ref().unwrap();
        let t = gpu.targets.as_ref().unwrap();
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let view_proj = proj * view;
        let params = VoxelGiParamsGpu {
            inv_proj: proj.inverse().to_cols_array(),
            inv_view: view.inverse().to_cols_array(),
            view: view.to_cols_array(),
            prev_view_proj: self.prev_view_proj.unwrap_or(view_proj).to_cols_array(),
            full_size: [width as f32, height as f32],
            trace_size: [t.width as f32, t.height as f32],
            near_size: near.as_ref().map_or([1.0, 1.0], |(_, [w, h])| [*w as f32, *h as f32]),
            start_voxels: self.start_voxels.max(0.0),
            max_distance: self.max_distance_m.max(0.0),
            max_steps: self.quality.cone_steps(),
            frame: self.frame,
            intensity: self.intensity.max(0.0),
            ambient: self.material_ambient.clamp(0.0, 1.0),
            blend: self.temporal_blend.clamp(0.01, 1.0),
            history_valid: self.prev_view_proj.is_some() as u32,
            has_sky: self.sky_lighting.is_some() as u32,
            debug: self.show_indirect as u32,
            near_field: near.is_some() as u32,
            sky_scale: self.sky_scale.max(0.0),
            _pad: [0; 2],
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        if self.sky_lighting.is_none() {
            let (up, down) = self.sky_gradient;
            queue.write_buffer(&gpu.gradient_sky, 0, bytemuck::cast_slice(&gradient_sky_lighting(up, down)));
        }
        let current = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);

        let tex = wgpu::BindingResource::TextureView;
        let group = |label: &str, layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>| {
            let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
        };
        let p = || gpu.params.as_entire_binding();
        let sky = self.sky_lighting.as_ref().unwrap_or(&gpu.gradient_sky);
        let trace = group(
            "VoxelGI/TraceBG",
            &gpu.trace_bgl,
            vec![
                p(),
                tex(depth),
                tex(&gbuffer.normal_view),
                self.volume_uniform.as_entire_binding(),
                tex(&self.volume_view),
                wgpu::BindingResource::Sampler(&self.volume_sampler),
                sky.as_entire_binding(),
                tex(&t.trace),
            ],
        );
        let temporal = group(
            "VoxelGI/TemporalBG",
            &gpu.temporal_bgl,
            vec![p(), tex(&t.trace), tex(&t.history[1 - current]), tex(depth), tex(&t.history[current]), wgpu::BindingResource::Sampler(&gpu.sampler)],
        );
        let near_view = near.as_ref().map_or(&gpu.no_near, |(v, _)| v);
        let composite = group(
            "VoxelGI/CompositeBG",
            &gpu.composite_bgl,
            vec![
                p(),
                tex(input),
                tex(depth),
                tex(&t.history[current]),
                tex(near_view),
                tex(&gbuffer.albedo_view),
                tex(&gbuffer.normal_view),
                sky.as_entire_binding(),
                tex(output),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGI/Screen"), timestamp_writes: crate::profiling::gpu_pass("VoxelGI/Screen").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.trace);
        pass.set_bind_group(0, &trace, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        pass.set_pipeline(&gpu.temporal);
        pass.set_bind_group(0, &temporal, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        pass.set_pipeline(&gpu.composite);
        pass.set_bind_group(0, &composite, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
        if let Some(near) = &mut self.near_field {
            near.destroy();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
