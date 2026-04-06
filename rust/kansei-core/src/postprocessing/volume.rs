use super::PostProcessingEffect;
use crate::cameras::Camera;
use crate::renderers::GBuffer;

/// Orchestrates a chain of post-processing effects with ping-pong textures.
pub struct PostProcessingVolume {
    pub effects: Vec<Box<dyn PostProcessingEffect>>,
    gbuffer: Option<GBuffer>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_sampler: Option<wgpu::Sampler>,
    blit_bgl: Option<wgpu::BindGroupLayout>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    presentation_format: wgpu::TextureFormat,
}

impl PostProcessingVolume {
    /// Create from a Renderer reference (preferred for user-facing code).
    pub fn from_renderer(
        renderer: &crate::renderers::Renderer,
        effects: Vec<Box<dyn PostProcessingEffect>>,
    ) -> Self {
        Self::new(renderer.device(), renderer.queue(), renderer.presentation_format(), effects)
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        presentation_format: wgpu::TextureFormat,
        effects: Vec<Box<dyn PostProcessingEffect>>,
    ) -> Self {
        Self {
            effects,
            gbuffer: None,
            blit_pipeline: None,
            blit_sampler: None,
            blit_bgl: None,
            device: device.clone(),
            queue: queue.clone(),
            presentation_format,
        }
    }

    pub fn gbuffer(&self) -> Option<&GBuffer> {
        self.gbuffer.as_ref()
    }

    /// Lazily create or resize the GBuffer, returning a reference to it.
    pub fn ensure_gbuffer(&mut self, width: u32, height: u32) -> &GBuffer {
        if self.gbuffer.is_none()
            || self
                .gbuffer
                .as_ref()
                .map(|g| g.width != width || g.height != height)
                .unwrap_or(false)
        {
            self.gbuffer = Some(GBuffer::new(&self.device, width, height, 1));
            for effect in &mut self.effects {
                effect.resize(width, height, self.gbuffer.as_ref().unwrap());
            }
        }
        self.gbuffer.as_ref().unwrap()
    }

    /// Lazily initialise the blit render pipeline, bind group layout, and sampler.
    fn initialize_blit(&mut self) {
        let device = &self.device;
        let surface_format = self.presentation_format;
        if self.blit_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/blit.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit PipelineLayout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit RenderPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.blit_bgl = Some(bgl);
        self.blit_pipeline = Some(pipeline);
        self.blit_sampler = Some(sampler);
    }

    /// Render the scene through the post-processing chain and blit to the surface.
    pub fn render(
        &mut self,
        camera: &Camera,
        surface_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        // Lazily create/resize GBuffer
        self.ensure_gbuffer(width, height);

        // Initialise effects that haven't been set up yet
        {
            let gbuffer = self.gbuffer.as_ref().unwrap();
            for effect in &mut self.effects {
                effect.initialize(&self.device, gbuffer, camera);
            }
        }

        // Lazily create blit pipeline
        self.initialize_blit();

        // Run effect chain with ping-pong
        // Track which texture holds the final result after the chain.
        // ping == 0 means last output was written to output_view,
        // ping == 1 means last output was written to ping_pong_view.
        let mut ping = 0u32;
        let mut ran_any_effect = false;

        if !self.effects.is_empty() {
            let gbuffer = self.gbuffer.as_ref().unwrap();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PostProcessingVolume/EffectsEncoder"),
            });

            let mut current_source_is_color = true; // first effect reads from color_view

            for effect in &mut self.effects {
                let (input_view, output_view) = if ping == 0 {
                    if current_source_is_color {
                        (&gbuffer.color_view, &gbuffer.output_view)
                    } else {
                        (&gbuffer.ping_pong_view, &gbuffer.output_view)
                    }
                } else {
                    (&gbuffer.output_view, &gbuffer.ping_pong_view)
                };

                effect.render(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    input_view,
                    &gbuffer.depth_view,
                    output_view,
                    camera,
                    width,
                    height,
                );

                current_source_is_color = false;
                ran_any_effect = true;
                ping = 1 - ping;
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let gbuffer = self.gbuffer.as_ref().unwrap();

        // Determine which texture holds the final result
        let final_view = if !ran_any_effect {
            &gbuffer.color_view
        } else if ping == 1 {
            // Last write went to output_view (ping was 0 before flip)
            &gbuffer.output_view
        } else {
            // Last write went to ping_pong_view (ping was 1 before flip)
            &gbuffer.ping_pong_view
        };

        // Blit the final texture to the surface
        let bgl = self.blit_bgl.as_ref().unwrap();
        let sampler = self.blit_sampler.as_ref().unwrap();
        let pipeline = self.blit_pipeline.as_ref().unwrap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit BindGroup"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(final_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PostProcessingVolume/BlitEncoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
