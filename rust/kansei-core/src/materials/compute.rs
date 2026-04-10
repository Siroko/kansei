use super::binding::{Binding, BindingResource, BindGroupBuilder};

/// A GPU compute pipeline with its bind group layout and cached bind group.
pub struct ComputePass {
    label: String,
    shader_code: String,
    binding_descriptors: Vec<Binding>,
    pipeline: Option<wgpu::ComputePipeline>,
    layout: Option<wgpu::BindGroupLayout>,
    bind_group: Option<wgpu::BindGroup>,
}

impl ComputePass {
    pub fn new(label: &str, shader_code: &str, bindings: Vec<Binding>) -> Self {
        Self {
            label: label.to_string(),
            shader_code: shader_code.to_string(),
            binding_descriptors: bindings,
            pipeline: None,
            layout: None,
            bind_group: None,
        }
    }

    /// Create the pipeline (call once after device is available).
    pub fn initialize(&mut self, device: &wgpu::Device) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}/Shader", self.label)),
            source: wgpu::ShaderSource::Wgsl(self.shader_code.as_str().into()),
        });

        let bgl = BindGroupBuilder::create_layout(device, &format!("{}/BGL", self.label), &self.binding_descriptors);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/PipelineLayout", self.label)),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}/Pipeline", self.label)),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.layout = Some(bgl);
    }

    /// Build or rebuild the bind group from concrete resources.
    pub fn set_bind_group(&mut self, device: &wgpu::Device, entries: &[(u32, BindingResource)]) {
        if let Some(ref layout) = self.layout {
            self.bind_group = Some(BindGroupBuilder::create_bind_group(
                device,
                &format!("{}/BG", self.label),
                layout,
                entries,
            ));
        }
    }

    pub fn pipeline(&self) -> Option<&wgpu::ComputePipeline> {
        self.pipeline.as_ref()
    }

    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    pub fn layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.layout.as_ref()
    }

    pub fn is_initialized(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Record a dispatch into a compute pass encoder.
    pub fn dispatch(&self, pass: &mut wgpu::ComputePass, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        if let (Some(pipeline), Some(bg)) = (&self.pipeline, &self.bind_group) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
    }
}
