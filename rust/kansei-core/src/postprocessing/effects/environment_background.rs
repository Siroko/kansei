use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::environment::{EnvironmentLighting, EnvironmentMap};
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

pub(crate) fn background_wgsl() -> String {
    format!("{}\n{}", crate::atmosphere::SKY_ENVIRONMENT_WGSL, include_str!("../../shaders/environment_background.wgsl"))
}

/// The WGSL `EnvironmentBackground`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct EnvironmentBackgroundGpu {
    inv_view_proj: [f32; 16],
    params: [f32; 4],
}

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// bound while no map is (`set_environment` with a source that has none)
    no_map: EnvironmentMap,
}

/// The environment as the background: an `environment::EnvironmentMap` (an `HdrEnvironment`'s)
/// wherever the scene left the depth buffer at the far plane, along each pixel's view ray. Put it
/// first in the chain, before the GI, the reflections, the fog and the tonemapper, as the
/// atmosphere's `AtmosphereEffect` goes. A source without a map (`GradientEnvironment`) leaves the
/// pixels as they are: its background is `Material::gradient_sky`.
pub struct EnvironmentBackgroundEffect {
    pub enabled: bool,
    /// Roughness of the map's mip shown (0 sharp, toward 1 a soft blur of it), for a background
    /// that stays out of focus.
    pub blur: f32,
    /// Scale of the background's radiance alone (the lighting keeps the environment's).
    pub intensity: f32,
    map: Option<EnvironmentMap>,
    gpu: Option<Gpu>,
}

impl EnvironmentBackgroundEffect {
    pub fn new(environment: &EnvironmentLighting) -> Self {
        Self { enabled: true, blur: 0.0, intensity: 1.0, map: environment.environment.clone(), gpu: None }
    }

    fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let texture = |sample_type, view_dimension| wgpu::BindingType::Texture { sample_type, view_dimension, multisampled: false };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("EnvironmentBackground/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, texture(wgpu::TextureSampleType::Float { filterable: false }, wgpu::TextureViewDimension::D2)),
                entry(2, texture(wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2)),
                entry(3, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: GBuffer::COLOR_FORMAT, view_dimension: wgpu::TextureViewDimension::D2 }),
                entry(4, texture(wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::Cube)),
                entry(5, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("EnvironmentBackground"), source: wgpu::ShaderSource::Wgsl(background_wgsl().into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("EnvironmentBackground"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("EnvironmentBackground"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        use wgpu::util::DeviceExt;
        let view = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("EnvironmentBackground/NoMap"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[0; 24],
            )
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        let no_map = EnvironmentMap { view, sampler: device.create_sampler(&Default::default()), mips: 1 };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("EnvironmentBackground/Uniform"),
            size: std::mem::size_of::<EnvironmentBackgroundGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu = Some(Gpu { pipeline, bgl, uniform, no_map });
    }
}

impl PostProcessingEffect for EnvironmentBackgroundEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

    /// Its map (none: the pixels pass through).
    fn set_environment(&mut self, environment: &EnvironmentLighting) {
        self.map = environment.environment.clone();
    }

    fn is_active(&self) -> bool {
        self.enabled && self.map.is_some()
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device, queue);
        }
        let gpu = self.gpu.as_ref().unwrap();
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let params = EnvironmentBackgroundGpu { inv_view_proj: view_proj.inverse().to_cols_array(), params: [self.blur.clamp(0.0, 1.0), self.intensity.max(0.0), self.map.is_some() as u32 as f32, 0.0] };
        queue.write_buffer(&gpu.uniform, 0, bytemuck::bytes_of(&params));
        let map = self.map.as_ref().unwrap_or(&gpu.no_map);
        let tex = wgpu::BindingResource::TextureView;
        let group = crate::renderers::cached_bind_group(device, &wgpu::BindGroupDescriptor {
            label: Some("EnvironmentBackground"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gpu.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tex(input) },
                wgpu::BindGroupEntry { binding: 2, resource: tex(depth) },
                wgpu::BindGroupEntry { binding: 3, resource: tex(output) },
                wgpu::BindGroupEntry { binding: 4, resource: tex(&map.view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&map.sampler) },
            ],
        });
        let stamp = crate::profiling::gpu_pass("Environment/Background");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Environment/Background"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_background_shader_validates_and_its_uniform_matches() {
        let code = background_wgsl();
        let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module.types.iter().find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == "EnvironmentBackground" => Some(*span as usize),
            _ => None,
        });
        assert_eq!(span, Some(std::mem::size_of::<EnvironmentBackgroundGpu>()));
    }
}
