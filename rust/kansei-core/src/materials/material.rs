use std::collections::HashMap;
use super::binding::{Binding, BindGroupBuilder, BindingResource};
use crate::renderers::SharedLayouts;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

impl CullMode {
    fn to_wgpu(self) -> Option<wgpu::Face> {
        match self {
            CullMode::None => None,
            CullMode::Front => Some(wgpu::Face::Front),
            CullMode::Back => Some(wgpu::Face::Back),
        }
    }
}

/// Configuration for a render material.
pub struct MaterialOptions {
    pub transparent: bool,
    pub depth_write: Option<bool>,
    pub depth_compare: wgpu::CompareFunction,
    pub cull_mode: CullMode,
    pub topology: wgpu::PrimitiveTopology,
}

impl Default for MaterialOptions {
    fn default() -> Self {
        Self {
            transparent: false,
            depth_write: None,
            depth_compare: wgpu::CompareFunction::Less,
            cull_mode: CullMode::Back,
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }
}

/// Pipeline cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub(crate) color_formats: Vec<wgpu::TextureFormat>,
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
    pub(crate) num_vertex_buffers: usize,
}

/// A render material — shader + pipeline cache + bind group.
pub struct Material {
    pub label: String,
    pub shader_code: String,
    pub options: MaterialOptions,
    pub bindings: Vec<Binding>,
    shader_module: Option<wgpu::ShaderModule>,
    material_bgl: Option<wgpu::BindGroupLayout>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pub(crate) pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    pub initialized: bool,
}

impl Material {
    pub fn new(label: &str, shader_code: &str, bindings: Vec<Binding>, options: MaterialOptions) -> Self {
        Self {
            label: label.to_string(),
            shader_code: shader_code.to_string(),
            options,
            bindings,
            shader_module: None,
            material_bgl: None,
            pipeline_layout: None,
            pipeline_cache: HashMap::new(),
            bind_group: None,
            initialized: false,
        }
    }

    /// Ensure shader module and layouts are created once.
    fn ensure_shared(&mut self, device: &wgpu::Device, shared: &SharedLayouts) {
        if self.shader_module.is_some() {
            return;
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}/Shader", self.label)),
            source: wgpu::ShaderSource::Wgsl(self.shader_code.as_str().into()),
        });

        let material_bgl = BindGroupBuilder::create_layout(
            device,
            &format!("{}/MaterialBGL", self.label),
            &self.bindings,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/PipelineLayout", self.label)),
            bind_group_layouts: &[&material_bgl, &shared.mesh_bgl, &shared.camera_bgl, &shared.shadow_bgl],
            push_constant_ranges: &[],
        });

        self.shader_module = Some(module);
        self.material_bgl = Some(material_bgl);
        self.pipeline_layout = Some(pipeline_layout);
    }

    /// Get or create a pipeline for the given render target config.
    pub fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedLayouts,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        self.ensure_shared(device, shared);

        let key = PipelineKey {
            color_formats: color_formats.to_vec(),
            depth_format,
            sample_count,
            num_vertex_buffers: vertex_layouts.len(),
        };

        if !self.pipeline_cache.contains_key(&key) {
            let targets: Vec<Option<wgpu::ColorTargetState>> = color_formats.iter().enumerate().map(|(i, fmt)| {
                let mut state = wgpu::ColorTargetState {
                    format: *fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                };
                if i == 0 && self.options.transparent {
                    state.blend = Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            operation: wgpu::BlendOperation::Add,
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        },
                        alpha: wgpu::BlendComponent {
                            operation: wgpu::BlendOperation::Add,
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        },
                    });
                }
                Some(state)
            }).collect();

            let depth_write = self.options.depth_write.unwrap_or(!self.options.transparent);
            let cull_mode = if self.options.transparent { None } else { self.options.cull_mode.to_wgpu() };

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}/Pipeline", self.label)),
                layout: self.pipeline_layout.as_ref(),
                vertex: wgpu::VertexState {
                    module: self.shader_module.as_ref().unwrap(),
                    entry_point: Some("vertex_main"),
                    buffers: vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: self.shader_module.as_ref().unwrap(),
                    entry_point: Some("fragment_main"),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: self.options.topology,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: depth_write,
                    depth_compare: self.options.depth_compare,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

            self.pipeline_cache.insert(key.clone(), pipeline);
        }

        self.pipeline_cache.get(&key).unwrap()
    }

    /// Create (or recreate) the material bind group from the given resources.
    pub fn create_bind_group(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedLayouts,
        resources: &[(u32, BindingResource)],
    ) {
        self.ensure_shared(device, shared);
        self.bind_group = Some(BindGroupBuilder::create_bind_group(
            device,
            &format!("{}/BindGroup", self.label),
            self.material_bgl.as_ref().unwrap(),
            resources,
        ));
        self.initialized = true;
    }

    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    pub fn material_bgl(&self) -> Option<&wgpu::BindGroupLayout> {
        self.material_bgl.as_ref()
    }
}
