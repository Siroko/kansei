//! The studio's ambient occlusion: a game's contact darkening multiplied into the lit image
//! (`ao.wgsl`), where the scene's own occlusion barely shows. The key light is most of a white
//! studio's light and its shadows are ray traced; the hybrid RT GI's occlusion only darkens the
//! sky's fill (about a sixth of the floor's light), and the engine's `SSAOEffect` runs before the
//! direct light is added (it darkens the ambient alone). This one runs after the lighting, the
//! reflections and the glass, before TAA, reading the GBuffer's normals.

use bytemuck::{Pod, Zeroable};

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::GBuffer;

const AO_WGSL: &str = include_str!("ao.wgsl");

/// The WGSL `AoParams` (240 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AoParams {
    view: [f32; 16],
    proj: [f32; 16],
    inv_proj: [f32; 16],
    size: [f32; 2],
    radius: f32,
    intensity: f32,
    falloff: f32,
    samples: u32,
    frame: u32,
    debug: u32,
    white: f32,
    bias: f32,
    _pad: [f32; 2],
}

pub struct AoEffect {
    pub enabled: bool,
    /// Metres the occlusion reaches, its strength (the exponent on the unoccluded share), how
    /// tight round contacts it is (an exponent on the samples' weights), samples a pixel.
    pub radius: f32,
    pub intensity: f32,
    pub falloff: f32,
    pub samples: u32,
    /// The occlusion alone, white `white` (cd/m²) where nothing occludes.
    pub debug: bool,
    pub white: f32,
    frame: u32,
    gpu: Option<(wgpu::ComputePipeline, wgpu::Buffer)>,
}

impl AoEffect {
    pub fn new() -> Self {
        Self { enabled: true, radius: 0.8, intensity: 3.0, falloff: 1.0, samples: 12, debug: false, white: 1.0, frame: 0, gpu: None }
    }
}

impl PostProcessingEffect for AoEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("StudioAO"), source: wgpu::ShaderSource::Wgsl(AO_WGSL.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("StudioAO"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("StudioAO/Params"), size: std::mem::size_of::<AoParams>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.gpu = Some((pipeline, params));
    }

    #[allow(clippy::too_many_arguments)]
    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, gbuffer: &GBuffer, input: &wgpu::TextureView, depth: &wgpu::TextureView, output: &wgpu::TextureView, camera: &Camera, width: u32, height: u32) {
        let Some((pipeline, params_buffer)) = &self.gpu else { return };
        self.frame = self.frame.wrapping_add(1);
        let proj = camera.projection_matrix.to_glam();
        let params = AoParams {
            view: camera.view_matrix.to_glam().to_cols_array(),
            proj: proj.to_cols_array(),
            inv_proj: proj.inverse().to_cols_array(),
            size: [width as f32, height as f32],
            radius: self.radius.max(0.01),
            intensity: self.intensity.max(0.0),
            falloff: self.falloff.max(0.0),
            samples: self.samples.clamp(1, 32),
            frame: self.frame,
            debug: self.debug as u32,
            white: self.white,
            bias: 0.01 * self.radius.max(0.01),
            _pad: [0.0; 2],
        };
        queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StudioAO"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffer.albedo_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&gbuffer.emissive_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(output) },
            ],
        });
        let stamp = kansei_core::profiling::gpu_pass("StudioAO");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("StudioAO"), timestamp_writes: stamp.as_ref().map(kansei_core::profiling::PassStamp::compute) });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn is_active(&self) -> bool {
        self.enabled
    }

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn name(&self) -> &'static str {
        "StudioAO"
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
    fn the_ao_shader_validates_and_its_params_match() {
        let module = naga::front::wgsl::parse_str(AO_WGSL).expect("ao.wgsl parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("ao.wgsl validates");
        let params = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("AoParams")).unwrap().1;
        let naga::TypeInner::Struct { span, .. } = params.inner else { panic!() };
        assert_eq!(span as usize, std::mem::size_of::<AoParams>());
    }
}
