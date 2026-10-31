use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = concat!(include_str!("../../atmosphere/shaders/sky_lighting.wgsl"), include_str!("../../shaders/screen_space_glass.wgsl"));

pub struct ScreenSpaceGlassOptions {
    /// The diameter of the ball each pixel's glass is taken for (about the objects' size): how
    /// far the view bends and turns over, and the tint's absorption.
    pub thickness: f32,
    /// How far behind the glass the background is taken to be (m).
    pub behind: f32,
    /// The index of refraction's spread from red to blue, a share of ior - 1 (0: none).
    pub dispersion: f32,
    /// The background's blur at full frost (roughness 1), pixels.
    pub frost_px: f32,
    /// Scale of the sky the glass reflects (`set_sky_lighting`).
    pub sky_scale: f32,
}

impl Default for ScreenSpaceGlassOptions {
    fn default() -> Self {
        Self { thickness: 0.4, behind: 1.5, dispersion: 0.03, frost_px: 12.0, sky_scale: 1.0 }
    }
}

/// `SsGlassParams` in screen_space_glass.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsGlassParams {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    eye: [f32; 3],
    thickness: f32,
    size: [f32; 2],
    dispersion: f32,
    frost_px: f32,
    sky_scale: f32,
    has_sky: u32,
    behind: f32,
    _pad: f32,
}

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Bound when no sky is set.
    no_sky: wgpu::Buffer,
}

/// Glass in screen space, a cheap stand-in for `rt::RtReflectionsEffect`'s ray traced glass on
/// the same GBuffer encoding: the pixels whose material wrote glass (`StandardLitOptions::glass`)
/// show the GBuffer's `background` (the opaque colour copied before the transmissive renderables)
/// refracted through a ball `thickness` across touching each pixel's surface (in through it, out
/// through its far side, so a glass ball turns the background over as a lens does), with
/// dispersion, the tint's absorption and frost, plus their Fresnel reflection of the sky
/// (`set_sky_lighting`). Give the glass's material `MaterialOptions::transmissive`, so the
/// background holds what is behind it. A proxy for the far side and a guessed depth behind
/// (`behind`): flat faces bend too little, no glass behind glass, no reflections of the scene. Put it where
/// the ray traced glass would go: after the GI, before the fog and TAA.
pub struct ScreenSpaceGlassEffect {
    pub options: ScreenSpaceGlassOptions,
    sky: Option<wgpu::Buffer>,
    gpu: Option<Gpu>,
}

impl ScreenSpaceGlassEffect {
    pub fn new(options: ScreenSpaceGlassOptions) -> Self {
        Self { options, sky: None, gpu: None }
    }

    /// The sky the glass reflects (a `SkyLighting` uniform: `SkyAtmosphereBindings::sky_lighting`,
    /// `gi::gradient_sky_lighting`), or none.
    pub fn set_sky_lighting(&mut self, sky: Option<&wgpu::Buffer>) {
        self.sky = sky.cloned();
    }
}

impl PostProcessingEffect for ScreenSpaceGlassEffect {
    fn set_environment(&mut self, environment: &crate::environment::EnvironmentLighting) {
        self.set_sky_lighting(Some(&environment.sky_lighting));
    }

    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_some() {
            return;
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("SsGlass/Shader"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SsGlass/BGL"),
            entries: &[
                uniform(0),
                texture(1, unfiltered),
                texture(2, wgpu::TextureSampleType::Float { filterable: true }),
                texture(3, wgpu::TextureSampleType::Depth),
                texture(4, unfiltered),
                texture(5, unfiltered),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                uniform(8),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("SsGlass/Layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SsGlass/Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, size| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sky_bytes = std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>() as u64;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SsGlass/Linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.gpu = Some(Gpu { pipeline, bgl, params: buffer("SsGlass/Params", std::mem::size_of::<SsGlassParams>() as u64), sampler, no_sky: buffer("SsGlass/NoSky", sky_bytes) });
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
        let Some(gpu) = &self.gpu else { return };
        let o = &self.options;
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        let params = SsGlassParams {
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: view_proj.inverse().to_cols_array(),
            eye: eye.to_array(),
            thickness: o.thickness.max(0.0),
            size: [width as f32, height as f32],
            dispersion: o.dispersion.max(0.0),
            frost_px: o.frost_px.max(0.0),
            sky_scale: o.sky_scale,
            has_sky: self.sky.is_some() as u32,
            behind: o.behind.max(0.0),
            _pad: 0.0,
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let tex = wgpu::BindingResource::TextureView;
        let group = crate::renderers::cached_bind_group(device, &wgpu::BindGroupDescriptor {
            label: Some("SsGlass/BindGroup"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gpu.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tex(input) },
                wgpu::BindGroupEntry { binding: 2, resource: tex(&gbuffer.background_view) },
                wgpu::BindGroupEntry { binding: 3, resource: tex(depth) },
                wgpu::BindGroupEntry { binding: 4, resource: tex(&gbuffer.normal_view) },
                wgpu::BindGroupEntry { binding: 5, resource: tex(&gbuffer.albedo_view) },
                wgpu::BindGroupEntry { binding: 6, resource: tex(output) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: self.sky.as_ref().unwrap_or(&gpu.no_sky).as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SsGlass"),
            timestamp_writes: crate::profiling::gpu_pass("SsGlass").as_ref().map(crate::profiling::PassStamp::compute),
        });
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

    /// screen_space_glass.wgsl validates, and its params and sky are the layouts the effect binds.
    #[test]
    fn shaders_validate_screen_space_glass() {
        let module = naga::front::wgsl::parse_str(WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
        let span = |name: &str| {
            module
                .types
                .iter()
                .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                    (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(span("SsGlassParams"), std::mem::size_of::<SsGlassParams>());
        assert_eq!(span("SkyLighting"), std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>());
    }
}
