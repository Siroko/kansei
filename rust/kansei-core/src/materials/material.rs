use std::collections::HashMap;
use super::binding::{Binding, BindGroupBuilder, BindingResource};
use super::shader_utils::ShaderChunks;
use crate::buffers::{BufferType, ComputeBuffer, Texture, Sampler, Bindable};
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
    /// Number of MRT fragment outputs this shader writes.
    /// None (default) = auto-detect (1 if !outputs_emissive, 2 if outputs_emissive).
    /// Some(N) = shader writes to the first N GBuffer targets.
    pub mrt_output_count: Option<usize>,
    /// Fragment entry point for shadow (depth-only) passes, for alpha-tested casters such as
    /// foliage cards: it runs with no colour targets and should `discard` cut-out texels, and it
    /// must not use group 3. `None` renders shadow depth from `vertex_main` alone.
    pub shadow_fragment_entry: Option<&'static str>,
    /// The fragment shader also writes screen-space motion at @location(4) (see
    /// `cameras::MOTION_VECTORS_WGSL`), for TAA: the renderer redraws the material in a velocity
    /// pass that keeps only that output. Without it, TAA reprojects the material's pixels by
    /// depth, which only follows the camera (fine for static things).
    pub outputs_velocity: bool,
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
            mrt_output_count: None,
            shadow_fragment_entry: None,
            outputs_velocity: false,
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

/// Depth-only (shadow) pipeline cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DepthPipelineKey {
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) num_vertex_buffers: usize,
    pub(crate) bias_constant: i32,
    pub(crate) bias_slope_bits: u32,
}

impl DepthPipelineKey {
    pub(crate) fn new(depth_format: wgpu::TextureFormat, num_vertex_buffers: usize, bias: wgpu::DepthBiasState) -> Self {
        Self { depth_format, num_vertex_buffers, bias_constant: bias.constant, bias_slope_bits: bias.slope_scale.to_bits() }
    }
}

/// A render material — shader + pipeline cache + bind group.
pub struct Material {
    pub label: String,
    pub shader_code: String,
    pub shader_chunks: Option<ShaderChunks>,
    pub options: MaterialOptions,
    pub bindings: Vec<Binding>,
    bindables: Vec<(u32, Box<dyn Bindable>)>,
    shader_module: Option<wgpu::ShaderModule>,
    material_bgl: Option<wgpu::BindGroupLayout>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    /// Groups 0-2 only: shadow passes render into textures that group 3 samples.
    depth_pipeline_layout: Option<wgpu::PipelineLayout>,
    pub(crate) pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    pub(crate) depth_pipeline_cache: HashMap<DepthPipelineKey, wgpu::RenderPipeline>,
    /// Velocity-pass pipelines by vertex-buffer count.
    pub(crate) velocity_pipeline_cache: HashMap<usize, wgpu::RenderPipeline>,
    /// Cluster pipelines (`get_cluster_pipeline`), keyed with no vertex buffers.
    pub(crate) cluster_pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    /// The cluster stage's module for the instance layout it was made for, or why there is none.
    cluster_module: Option<(Option<(u64, Vec<wgpu::VertexAttribute>)>, Result<wgpu::ShaderModule, String>)>,
    cluster_pipeline_layout: Option<wgpu::PipelineLayout>,
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
            depth_pipeline_layout: None,
            pipeline_cache: HashMap::new(),
            depth_pipeline_cache: HashMap::new(),
            velocity_pipeline_cache: HashMap::new(),
            cluster_pipeline_cache: HashMap::new(),
            cluster_module: None,
            cluster_pipeline_layout: None,
            bind_group: None,
            initialized: false,
        }
    }

    /// Ensure shader module and layouts are created once.
    fn ensure_shared(&mut self, device: &wgpu::Device, shared: &SharedLayouts) {
        if self.shader_module.is_some() {
            return;
        }

        let processed_code = self.processed_code();

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

        let depth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/DepthPipelineLayout", self.label)),
            bind_group_layouts: &[&material_bgl, &shared.camera_bgl, &shared.mesh_bgl],
            push_constant_ranges: &[],
        });

        self.shader_module = Some(module);
        self.material_bgl = Some(material_bgl);
        self.pipeline_layout = Some(pipeline_layout);
        self.depth_pipeline_layout = Some(depth_pipeline_layout);
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
            let pipeline = self.create_pipeline(device, self.pipeline_layout.as_ref().unwrap(), self.shader_module.as_ref().unwrap(), "vertex_main", vertex_layouts, color_formats, depth_format, sample_count, "Pipeline");
            self.pipeline_cache.insert(key.clone(), pipeline);
        }

        self.pipeline_cache.get(&key).unwrap()
    }

    /// A render pipeline of this material: `vertex_entry` of `module` over `vertex_layouts`, its
    /// `fragment_main`, into the given targets.
    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        vertex_entry: &str,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
        label: &str,
    ) -> wgpu::RenderPipeline {
        // Number of fragment shader outputs: @location(0) always,
        // @location(1) only if the shader outputs emissive.
        let shader_output_count = self.options.mrt_output_count.unwrap_or_else(||
            if self.options.outputs_emissive { 2 } else { 1 });

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

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{}/{label}", self.label)),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(vertex_entry),
                buffers: vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
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
        })
    }

    /// Get or create the pipeline that draws this material over a cluster draw (the camera's cut
    /// of `Renderable::clusters`): its WGSL with the generated vertex stage for `instances`'
    /// records. The Err says why the stage can't be generated; the renderable then keeps the
    /// ordinary path.
    pub(crate) fn get_cluster_pipeline(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedLayouts,
        instances: Option<&crate::buffers::InstanceBufferLayout>,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Result<&wgpu::RenderPipeline, String> {
        assert!(self.pipeline_layout.is_some(), "Material not initialized — call initialize() first");
        let layout_key = instances.map(|l| (l.stride, l.attributes.clone()));
        if self.cluster_module.as_ref().is_none_or(|(k, _)| *k != layout_key) {
            let module = crate::clusters::cluster_vertex_stage(&self.processed_code(), instances).map(|code| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&format!("{}/ClusterShader", self.label)), source: wgpu::ShaderSource::Wgsl(code.into()) })
            });
            self.cluster_module = Some((layout_key, module));
            self.cluster_pipeline_cache.clear();
        }
        let module = self.cluster_module.as_ref().unwrap().1.clone()?;
        let layout = self
            .cluster_pipeline_layout
            .get_or_insert_with(|| {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}/ClusterPipelineLayout", self.label)),
                    bind_group_layouts: &[self.material_bgl.as_ref().unwrap(), &shared.camera_bgl, &shared.cluster_mesh_bgl, &shared.shadow_bgl],
                    push_constant_ranges: &[],
                })
            })
            .clone();
        let key = PipelineKey { color_formats: color_formats.to_vec(), depth_format, sample_count, num_vertex_buffers: 0 };
        if !self.cluster_pipeline_cache.contains_key(&key) {
            let pipeline = self.create_pipeline(device, &layout, &module, crate::clusters::CLUSTER_VERTEX_ENTRY, &[], color_formats, depth_format, sample_count, "ClusterPipeline");
            self.cluster_pipeline_cache.insert(key.clone(), pipeline);
        }
        Ok(&self.cluster_pipeline_cache[&key])
    }

    /// The cluster pipeline made for a pass with `key` (whatever its vertex buffers), if any.
    pub(crate) fn cluster_pipeline(&self, key: &PipelineKey) -> Option<&wgpu::RenderPipeline> {
        self.cluster_pipeline_cache.get(&PipelineKey { num_vertex_buffers: 0, ..key.clone() })
    }

    /// The WGSL with its includes resolved.
    fn processed_code(&self) -> String {
        match &self.shader_chunks {
            Some(chunks) => crate::materials::parse_includes(&self.shader_code, chunks),
            None => self.shader_code.clone(),
        }
    }

    /// Get or create the depth-only pipeline shadow passes draw this material with: its own
    /// `vertex_main` (so instancing and vertex animation cast matching shadows), with the camera
    /// group bound to the light's view, plus `shadow_fragment_entry` if set.
    pub(crate) fn get_depth_pipeline(
        &mut self,
        device: &wgpu::Device,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        depth_format: wgpu::TextureFormat,
        bias: wgpu::DepthBiasState,
    ) -> &wgpu::RenderPipeline {
        assert!(self.depth_pipeline_layout.is_some(), "Material not initialized — call initialize() first");
        let key = DepthPipelineKey::new(depth_format, vertex_layouts.len(), bias);
        if !self.depth_pipeline_cache.contains_key(&key) {
            let module = self.shader_module.as_ref().unwrap();
            let fragment = self.options.shadow_fragment_entry.map(|entry| wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                targets: &[],
                compilation_options: Default::default(),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}/DepthPipeline", self.label)),
                layout: self.depth_pipeline_layout.as_ref(),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vertex_main"),
                    buffers: vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment,
                primitive: wgpu::PrimitiveState {
                    topology: self.options.topology,
                    cull_mode: self.options.cull_mode.to_wgpu(),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias,
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
            self.depth_pipeline_cache.insert(key.clone(), pipeline);
        }
        self.depth_pipeline_cache.get(&key).unwrap()
    }

    /// Get or create the pipeline of the renderer's velocity pass: this material's shader with
    /// only its @location(4) output kept (a Rg16Float target at index 4, nothing before it),
    /// depth-tested against the GBuffer without writing depth.
    pub(crate) fn get_velocity_pipeline(&mut self, device: &wgpu::Device, vertex_layouts: &[wgpu::VertexBufferLayout]) -> &wgpu::RenderPipeline {
        assert!(self.pipeline_layout.is_some(), "Material not initialized — call initialize() first");
        use crate::renderers::GBuffer;
        let key = vertex_layouts.len();
        if !self.velocity_pipeline_cache.contains_key(&key) {
            let module = self.shader_module.as_ref().unwrap();
            let mut targets: Vec<Option<wgpu::ColorTargetState>> = vec![None; GBuffer::VELOCITY_TARGET];
            targets.push(Some(wgpu::ColorTargetState {
                format: GBuffer::VELOCITY_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}/VelocityPipeline", self.label)),
                layout: self.pipeline_layout.as_ref(),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vertex_main"),
                    buffers: vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fragment_main"),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: self.options.topology,
                    cull_mode: self.options.cull_mode.to_wgpu(),
                    ..Default::default()
                },
                // the GBuffer pass and this one must produce the same depths: mark the position
                // output @invariant; the small bias toward the camera covers compilers that differ
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: GBuffer::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState { constant: -4, slope_scale: -1.0, clamp: 0.0 },
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
            self.velocity_pipeline_cache.insert(key, pipeline);
        }
        self.velocity_pipeline_cache.get(&key).unwrap()
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

    /// Attach any Bindable (Texture, Sampler, ComputeBuffer, …) to a binding
    /// slot. The renderer lazily initializes it before the first draw.
    pub fn set_bindable(&mut self, binding: u32, resource: impl Bindable + 'static) {
        if let Some((_, slot)) = self.bindables.iter_mut().find(|(b, _)| *b == binding) {
            *slot = Box::new(resource);
        } else {
            self.bindables.push((binding, Box::new(resource)));
        }
        self.initialized = false;
    }

    /// Convenience: attach a uniform buffer from typed data.
    pub fn set_uniform_bindable<T: bytemuck::Pod>(&mut self, binding: u32, label: &str, data: &[T]) {
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let buf = ComputeBuffer::from_slice(label, BufferType::Uniform, usage, data);
        self.set_bindable(binding, buf);
    }

    /// Get the GPU buffer for a bindable at the given binding index.
    /// Only works after the bindable has been initialized (i.e. after
    /// the first `renderer.render()` call). Returns None if the binding
    /// doesn't exist or isn't a buffer type.
    pub fn bindable_buffer(&self, binding: u32) -> Option<wgpu::Buffer> {
        self.bindables.iter()
            .find(|(b, _)| *b == binding)
            .and_then(|(_, bindable)| {
                if let Some(BindingResource::Buffer { buffer, .. }) = bindable.binding_resource() {
                    Some(buffer.clone())
                } else {
                    None
                }
            })
    }

    /// Ensure this material has a bind group from its owned bindables.
    /// Called automatically by the renderer before the first draw.
    pub fn ensure_bindables_initialized(&mut self, renderer: &Renderer) {
        if self.bindables.is_empty() || self.initialized {
            return;
        }
        let device = renderer.raw_device();
        let queue = renderer.raw_queue();

        // Initialize all bindables (Texture, Sampler, ComputeBuffer, …)
        for (_, bindable) in &mut self.bindables {
            bindable.ensure_ready(device, queue);
        }

        // Collect binding resources. We must break the borrow on self.bindables
        // before calling create_bind_group(&mut self). BindingResource holds
        // references, so we clone the underlying GPU handles into locals.
        enum OwnedResource {
            Buffer(wgpu::Buffer),
            TextureView(wgpu::TextureView),
            Sampler(wgpu::Sampler),
        }
        let owned: Vec<(u32, OwnedResource)> = self.bindables.iter()
            .filter_map(|(b, bindable)| {
                bindable.binding_resource().map(|r| {
                    let owned = match r {
                        BindingResource::Buffer { buffer, .. } => OwnedResource::Buffer(buffer.clone()),
                        BindingResource::TextureView(v) => OwnedResource::TextureView(v.clone()),
                        BindingResource::Sampler(s) => OwnedResource::Sampler(s.clone()),
                        BindingResource::StorageTexture(v) => OwnedResource::TextureView(v.clone()),
                    };
                    (*b, owned)
                })
            })
            .collect();
        let resources: Vec<(u32, BindingResource)> = owned.iter()
            .map(|(b, r)| match r {
                OwnedResource::Buffer(buf) => (*b, BindingResource::Buffer { buffer: buf, offset: 0, size: None }),
                OwnedResource::TextureView(v) => (*b, BindingResource::TextureView(v)),
                OwnedResource::Sampler(s) => (*b, BindingResource::Sampler(s)),
            })
            .collect();

        self.create_bind_group(device, renderer.shared_layouts(), &resources);
    }
}
