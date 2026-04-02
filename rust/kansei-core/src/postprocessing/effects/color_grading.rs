use crate::cameras::Camera;
use crate::renderers::GBuffer;
use crate::postprocessing::PostProcessingEffect;

pub struct ColorGradingOptions {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub tint: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub black_point: f32,
}

impl Default for ColorGradingOptions {
    fn default() -> Self {
        Self {
            brightness: 0.0, contrast: 1.0, saturation: 1.0,
            temperature: 0.0, tint: 0.0, highlights: 1.0, shadows: 1.0, black_point: 0.0,
        }
    }
}

pub struct ColorGradingEffect {
    pub options: ColorGradingOptions,
    pipeline: Option<wgpu::ComputePipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    initialized: bool,
}

impl ColorGradingEffect {
    pub fn new(options: ColorGradingOptions) -> Self {
        Self { options, pipeline: None, bgl: None, params_buf: None, initialized: false }
    }
}

impl PostProcessingEffect for ColorGradingEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized { return; }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ColorGrading/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/color_grading.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ColorGrading/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });

        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ColorGrading/Pipeline"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColorGrading/Params"), size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.bgl = Some(bgl);
        self.initialized = true;
    }

    fn render(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView, _depth: &wgpu::TextureView, output: &wgpu::TextureView,
        _camera: &Camera, width: u32, height: u32,
    ) {
        let pipeline = match &self.pipeline { Some(p) => p, None => return };

        let o = &self.options;
        let data: [f32; 12] = [
            o.brightness, o.contrast, o.saturation, o.temperature,
            o.tint, o.highlights, o.shadows, o.black_point,
            width as f32, height as f32, 0.0, 0.0,
        ];
        queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(&data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: self.bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_ref().unwrap().as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ColorGrading"), ..Default::default() });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}
    fn destroy(&mut self) { self.initialized = false; }
}
