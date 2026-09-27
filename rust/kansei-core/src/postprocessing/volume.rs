use super::PostProcessingEffect;
use crate::cameras::Camera;
use crate::renderers::GBuffer;

/// Orchestrates a chain of post-processing effects with ping-pong textures.
///
/// The chain starts at the GBuffer's size (the renderer's render size) and ends at the display
/// size of the surface. With a render scale below 1 the first effect that
/// `upscales_to_display` (the TAA resolve) takes it from one to the other, and the effects after
/// it run on the volume's own display-size ping-pong textures; without one, the blit stretches
/// the result to the surface.
pub struct PostProcessingVolume {
    pub effects: Vec<Box<dyn PostProcessingEffect>>,
    gbuffer: Option<GBuffer>,
    /// Ping-pong textures at the display size, for the effects after an upscaler.
    display_targets: Option<DisplayTargets>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_sampler: Option<wgpu::Sampler>,
    blit_bgl: Option<wgpu::BindGroupLayout>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    presentation_format: wgpu::TextureFormat,
}

struct DisplayTargets {
    size: (u32, u32),
    views: [wgpu::TextureView; 2],
}

/// Where an effect of the chain reads or writes.
#[derive(Clone, Copy, PartialEq)]
enum Slot {
    Color,
    Output,
    PingPong,
    Display(usize),
}

fn slot_view<'a>(gbuffer: &'a GBuffer, display: Option<&'a DisplayTargets>, slot: Slot) -> &'a wgpu::TextureView {
    match slot {
        Slot::Color => &gbuffer.color_view,
        Slot::Output => &gbuffer.output_view,
        Slot::PingPong => &gbuffer.ping_pong_view,
        Slot::Display(i) => &display.expect("display targets").views[i],
    }
}

impl PostProcessingVolume {
    pub fn new(
        renderer: &crate::renderers::Renderer,
        effects: Vec<Box<dyn PostProcessingEffect>>,
    ) -> Self {
        Self {
            effects,
            gbuffer: None,
            display_targets: None,
            blit_pipeline: None,
            blit_sampler: None,
            blit_bgl: None,
            device: renderer.device().clone(),
            queue: renderer.queue().clone(),
            presentation_format: renderer.presentation_format(),
        }
    }

    /// Whether any effect wants a jittered projection (the renderer then jitters the camera).
    pub fn wants_jitter(&self) -> bool {
        self.effects.iter().any(|e| e.is_active() && e.wants_jitter())
    }

    pub fn gbuffer(&self) -> Option<&GBuffer> {
        self.gbuffer.as_ref()
    }

    /// Lazily create or resize the GBuffer (at the render size), returning a reference to it.
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

    /// Display-size ping-pong textures, (re)created at `size`.
    fn ensure_display_targets(&mut self, size: (u32, u32)) {
        if self.display_targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let view = |label| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: GBuffer::COLOR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let views = [view("PostProcessingVolume/DisplayA"), view("PostProcessingVolume/DisplayB")];
        self.display_targets = Some(DisplayTargets { size, views });
    }

    /// Render the scene through the post-processing chain and blit to the surface.
    ///
    /// `width` and `height` are the surface's size. The GBuffer is used at the size
    /// `ensure_gbuffer` last gave it (created at the surface's size if it does not exist yet).
    pub fn render(
        &mut self,
        camera: &Camera,
        surface_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        if self.gbuffer.is_none() {
            self.ensure_gbuffer(width, height);
        }
        let render_size = {
            let gbuffer = self.gbuffer.as_ref().unwrap();
            (gbuffer.width, gbuffer.height)
        };
        let display_size = (width, height);
        let upscaling = render_size != display_size && self.effects.iter().any(|e| e.is_active() && e.upscales_to_display());
        if upscaling {
            self.ensure_display_targets(display_size);
        }

        // Initialise effects that haven't been set up yet
        {
            let gbuffer = self.gbuffer.as_ref().unwrap();
            for effect in &mut self.effects {
                effect.initialize(&self.device, gbuffer, camera);
            }
        }

        // Lazily create blit pipeline
        self.initialize_blit();

        // Run the effect chain, each effect reading the previous one's output: at the render
        // size on the GBuffer's output / ping-pong textures, then (from the upscaler on) at the
        // display size on the volume's own pair.
        let mut source = Slot::Color;
        if !self.effects.is_empty() {
            let gbuffer = self.gbuffer.as_ref().unwrap();
            let view = |slot| slot_view(gbuffer, self.display_targets.as_ref(), slot);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PostProcessingVolume/EffectsEncoder"),
            });

            for effect in &mut self.effects {
                if !effect.is_active() {
                    continue;
                }
                let at_display = matches!(source, Slot::Display(_)) || (upscaling && effect.upscales_to_display());
                let target = match source {
                    _ if at_display => Slot::Display(if source == Slot::Display(0) { 1 } else { 0 }),
                    Slot::Output => Slot::PingPong,
                    _ => Slot::Output,
                };
                let (w, h) = if at_display { display_size } else { render_size };

                effect.render(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    gbuffer,
                    view(source),
                    &gbuffer.depth_view,
                    view(target),
                    camera,
                    w,
                    h,
                );
                source = target;
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let final_view = slot_view(self.gbuffer.as_ref().unwrap(), self.display_targets.as_ref(), source);

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
