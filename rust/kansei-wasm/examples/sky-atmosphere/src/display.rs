//! A minimal display transform for the example: manual exposure (EV100), the Narkowicz ACES fit
//! and sRGB encoding when the canvas format is not sRGB. kansei's blit copies its input as is.

use bytemuck::{Pod, Zeroable};
use kansei_core::cameras::Camera;
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

const WGSL: &str = r#"
struct Params {
    exposure   : f32,
    encodeSrgb : u32,
    width      : u32,
    height     : u32,
}

@group(0) @binding(0) var inputTex  : texture_2d<f32>;
@group(0) @binding(1) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p : Params;

fn aces(x: vec3f) -> vec3f {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fn srgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let c = aces(textureLoad(inputTex, gid.xy, 0).rgb * p.exposure);
    textureStore(outputTex, gid.xy, vec4f(select(c, srgb(c), p.encodeSrgb != 0u), 1.0));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    exposure: f32,
    encode_srgb: u32,
    width: u32,
    height: u32,
}

pub struct DisplayEffect {
    /// Exposure value at ISO 100: scene luminance L maps to L / (1.2 * 2^EV100) before the curve.
    pub ev100: f32,
    encode_srgb: bool,
    gpu: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer)>,
}

impl DisplayEffect {
    pub fn new(encode_srgb: bool) -> Self {
        Self { ev100: 10.0, encode_srgb, gpu: None }
    }
}

impl PostProcessingEffect for DisplayEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_some() {
            return;
        }
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                entry(2, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Display"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Display"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Display"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Display/Params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu = Some((pipeline, bgl, params));
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        self.initialize(device, gbuffer, camera);
        let (pipeline, bgl, params) = self.gpu.as_ref().unwrap();
        let exposure = 1.0 / (1.2 * 2f32.powf(self.ev100));
        queue.write_buffer(params, 0, bytemuck::bytes_of(&Params { exposure, encode_srgb: self.encode_srgb as u32, width, height }));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display/BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Display"), ..Default::default() });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
