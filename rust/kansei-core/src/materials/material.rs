use std::collections::HashMap;
use super::binding::{Binding, BindGroupBuilder, BindingResource};
use super::shader_utils::ShaderChunks;
use crate::buffers::{BufferType, GpuBuffer};
use crate::renderers::Renderer;
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
    pub outputs_emissive: bool,
}

impl Default for MaterialOptions {
    fn default() -> Self {
        Self {
            transparent: false,
            depth_write: None,
            depth_compare: wgpu::CompareFunction::Less,
            cull_mode: CullMode::Back,
            topology: wgpu::PrimitiveTopology::TriangleList,
            outputs_emissive: false,
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
    pub shader_chunks: Option<ShaderChunks>,
    pub options: MaterialOptions,
    pub bindings: Vec<Binding>,
    bindables: Vec<(u32, GpuBuffer)>,
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
            shader_chunks: None,
            options,
            bindings,
            bindables: Vec::new(),
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

        let processed_code = if let Some(ref chunks) = self.shader_chunks {
            crate::materials::parse_includes(&self.shader_code, chunks)
        } else {
            self.shader_code.clone()
        };

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}/Shader", self.label)),
            source: wgpu::ShaderSource::Wgsl(processed_code.as_str().into()),
        });

        let material_bgl = BindGroupBuilder::create_layout(
            device,
            &format!("{}/MaterialBGL", self.label),
            &self.bindings,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/PipelineLayout", self.label)),
            bind_group_layouts: &[&material_bgl, &shared.camera_bgl, &shared.mesh_bgl, &shared.shadow_bgl],
            push_constant_ranges: &[],
        });

        self.shader_module = Some(module);
        self.material_bgl = Some(material_bgl);
        self.pipeline_layout = Some(pipeline_layout);
    }

    /// Initialize GPU resources. Called by Renderer during first render.
    pub fn initialize(&mut self, device: &wgpu::Device, shared: &SharedLayouts) {
        self.ensure_shared(device, shared);
    }

    /// Get or create a pipeline for the given render target config.
    pub fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        // Pipeline layout already created during initialize()
        // If not initialized yet, this will fail — Renderer must call initialize() first
        assert!(self.pipeline_layout.is_some(), "Material not initialized — call initialize() first");

        let key = PipelineKey {
            color_formats: color_formats.to_vec(),
            depth_format,
            sample_count,
            num_vertex_buffers: vertex_layouts.len(),
        };

        if !self.pipeline_cache.contains_key(&key) {
            // Number of fragment shader outputs: @location(0) always,
            // @location(1) only if the shader outputs emissive.
            let shader_output_count = if self.options.outputs_emissive { 2 } else { 1 };

            let targets: Vec<Option<wgpu::ColorTargetState>> = color_formats.iter().enumerate().map(|(i, fmt)| {
                let mut state = wgpu::ColorTargetState {
                    format: *fmt,
                    blend: None,
                    // Targets beyond what the shader outputs must have empty write mask,
                    // otherwise WebGPU validation fails ("no corresponding fragment stage output").
                    write_mask: if i < shader_output_count {
                        wgpu::ColorWrites::ALL
                    } else {
                        wgpu::ColorWrites::empty()
                    },
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

    /// Attach/replace a uniform bindable owned by this material.
    pub fn set_uniform_bindable<T: bytemuck::Pod>(&mut self, binding: u32, label: &str, data: &[T]) {
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let buf = GpuBuffer::from_slice(label, BufferType::Uniform, usage, data);
        if let Some((_, slot)) = self.bindables.iter_mut().find(|(b, _)| *b == binding) {
            *slot = buf;
        } else {
            self.bindables.push((binding, buf));
        }
        self.initialized = false;
    }

    /// Get the GPU buffer for a uniform bindable at the given binding index.
    pub fn bindable_buffer(&self, binding: u32) -> Option<&wgpu::Buffer> {
        self.bindables.iter()
            .find(|(b, _)| *b == binding)
            .and_then(|(_, buf)| buf.gpu_buffer())
    }

    /// Ensure this material has a bind group from its owned bindables.
    pub fn ensure_bindables_initialized(&mut self, renderer: &Renderer) {
        if self.bindables.is_empty() || self.initialized {
            return;
        }
        for (_, buf) in &mut self.bindables {
            buf.ensure_ready(renderer.raw_device(), renderer.raw_queue());
        }
        let buffer_handles: Vec<(u32, wgpu::Buffer)> = self
            .bindables
            .iter()
            .map(|(binding, buf)| {
                (
                    *binding,
                    buf.gpu_buffer()
                        .expect("Material bindable buffer should be initialized")
                        .clone(),
                )
            })
            .collect();
        let resources: Vec<(u32, BindingResource)> = buffer_handles
            .iter()
            .map(|(binding, buffer)| {
                (
                    *binding,
                    BindingResource::Buffer {
                        buffer,
                        offset: 0,
                        size: None,
                    },
                )
            })
            .collect();
        self.create_bind_group(renderer.raw_device(), renderer.shared_layouts(), &resources);
    }
}
