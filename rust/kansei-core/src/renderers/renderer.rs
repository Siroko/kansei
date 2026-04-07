use crate::math::Vec4;
use crate::cameras::Camera;
use crate::geometries::Vertex;
use crate::lights::{Light, LightUniforms, LIGHT_UNIFORM_BYTES};
use crate::materials::ComputePass;
use crate::objects::Scene;
use super::compute_batch::ComputeBatch;
use super::gbuffer::GBuffer;
use super::shared_layouts::SharedLayouts;

/// Core WebGPU renderer configuration.
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub device_pixel_ratio: f32,
    pub sample_count: u32,
    pub clear_color: Vec4,
    pub present_mode: wgpu::PresentMode,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            device_pixel_ratio: 1.0,
            sample_count: 4,
            clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            present_mode: wgpu::PresentMode::Fifo,
        }
    }
}

/// The main GPU renderer.
pub struct Renderer {
    pub config: RendererConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    presentation_format: wgpu::TextureFormat,
    // Depth texture for the canvas render path
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    // MSAA color texture (if sample_count > 1)
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    // Shared per-object matrix buffers (dynamic offset uniform)
    world_matrices_buf: Option<wgpu::Buffer>,
    normal_matrices_buf: Option<wgpu::Buffer>,
    world_matrices_staging: Vec<f32>,
    normal_matrices_staging: Vec<f32>,
    matrix_alignment: u32,
    last_object_count: usize,
    // Shared bind group layouts
    shared_layouts: Option<SharedLayouts>,
    // Mesh bind group (group 1) — dynamic offset into matrix buffers
    mesh_bind_group: Option<wgpu::BindGroup>,
    // Light uniform buffer (packed into camera bind group, binding 2)
    light_buf: Option<wgpu::Buffer>,
    light_uniforms: LightUniforms,
    // Shadow resources
    shadow_map: Option<crate::shadows::ShadowMap>,
    shadow_uniform_buf: Option<wgpu::Buffer>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    shadow_comparison_sampler: Option<wgpu::Sampler>,
    shadow_dummy_depth_tex: Option<wgpu::Texture>,
    shadow_dummy_depth_view: Option<wgpu::TextureView>,
    cube_dummy_tex: Option<wgpu::Texture>,
    cube_dummy_view: Option<wgpu::TextureView>,
    cube_shadow_sampler: Option<wgpu::Sampler>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_light_vp_bgl: Option<wgpu::BindGroupLayout>,
    shadow_light_vp_bg: Option<wgpu::BindGroup>,
    shadows_enabled: bool,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            device: None,
            queue: None,
            surface: None,
            surface_config: None,
            presentation_format: wgpu::TextureFormat::Bgra8Unorm,
            depth_texture: None,
            depth_view: None,
            msaa_texture: None,
            msaa_view: None,
            world_matrices_buf: None,
            normal_matrices_buf: None,
            world_matrices_staging: Vec::new(),
            normal_matrices_staging: Vec::new(),
            matrix_alignment: 256,
            last_object_count: 0,
            shared_layouts: None,
            mesh_bind_group: None,
            light_buf: None,
            light_uniforms: LightUniforms::new(),
            shadow_map: None,
            shadow_uniform_buf: None,
            shadow_bind_group: None,
            shadow_comparison_sampler: None,
            shadow_dummy_depth_tex: None,
            shadow_dummy_depth_view: None,
            cube_dummy_tex: None,
            cube_dummy_view: None,
            cube_shadow_sampler: None,
            shadow_pipeline: None,
            shadow_light_vp_bgl: None,
            shadow_light_vp_bg: None,
            shadows_enabled: false,
        }
    }

    /// Initialize with a wgpu surface (from winit window or web canvas).
    pub async fn initialize(&mut self, surface: wgpu::Surface<'static>, adapter: &wgpu::Adapter) {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Kansei Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            }, None)
            .await
            .expect("Failed to create device");

        self.matrix_alignment = device.limits().min_uniform_buffer_offset_alignment;

        let surface_caps = surface.get_capabilities(adapter);
        let format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Pick best available present mode: prefer requested, fallback to Mailbox, then Fifo
        let present_mode = if surface_caps.present_modes.contains(&self.config.present_mode) {
            self.config.present_mode
        } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            log::info!("PresentMode::{:?} not supported, using Mailbox", self.config.present_mode);
            wgpu::PresentMode::Mailbox
        } else {
            log::info!("Using PresentMode::Fifo (vsync)");
            wgpu::PresentMode::Fifo
        };
        log::info!("Present mode: {:?} (available: {:?})", present_mode, surface_caps.present_modes);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: self.config.width,
            height: self.config.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        self.presentation_format = format;
        self._create_depth_texture(&device);

        // Create shared bind group layouts
        let shared = SharedLayouts::new(&device);

        // Create light uniform buffer
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/Lights"),
            size: LIGHT_UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shadow resources — dummy depth texture + comparison sampler + uniform buffer
        let dummy_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/DummyDepth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth.create_view(&Default::default());

        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/ShadowSampler"),
            compare: Some(wgpu::CompareFunction::Less),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Dummy cubemap distance texture (1x1x6 r32float)
        let cube_dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/DummyCubeShadow"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cube_dummy_view = cube_dummy_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let cube_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/CubeShadowSampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shadow_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/ShadowUniforms"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shadow bind group with dummy depth texture (shadows disabled by default)
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/ShadowBG"),
            layout: &shared.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&comparison_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shadow_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&cube_dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&cube_shadow_sampler),
                },
            ],
        });

        self.shared_layouts = Some(shared);
        self.light_buf = Some(light_buf);
        self.shadow_bind_group = Some(shadow_bind_group);
        self.shadow_dummy_depth_tex = Some(dummy_depth);
        self.shadow_dummy_depth_view = Some(dummy_depth_view);
        self.shadow_comparison_sampler = Some(comparison_sampler);
        self.shadow_uniform_buf = Some(shadow_uniform_buf);
        self.cube_dummy_tex = Some(cube_dummy_tex);
        self.cube_dummy_view = Some(cube_dummy_view);
        self.cube_shadow_sampler = Some(cube_shadow_sampler);

        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
    }

    /// Returns a reference to the underlying wgpu device.
    ///
    /// Prefer using higher-level APIs (e.g. `material.set_uniform_bindable()`)
    /// instead of accessing the device directly. This accessor will be removed
    /// in a future release once all subsystems manage their own GPU resources.
    #[doc(hidden)]
    pub fn device(&self) -> &wgpu::Device {
        self.device.as_ref().expect("Renderer not initialized")
    }

    /// Returns a reference to the underlying wgpu queue.
    ///
    /// Prefer using higher-level APIs instead of accessing the queue directly.
    /// This accessor will be removed in a future release.
    #[doc(hidden)]
    pub fn queue(&self) -> &wgpu::Queue {
        self.queue.as_ref().expect("Renderer not initialized")
    }

    pub(crate) fn raw_device(&self) -> &wgpu::Device {
        self.device()
    }

    pub(crate) fn raw_queue(&self) -> &wgpu::Queue {
        self.queue()
    }

    pub fn create_command_encoder(&self, desc: &wgpu::CommandEncoderDescriptor) -> wgpu::CommandEncoder {
        self.device().create_command_encoder(desc)
    }

    pub fn submit(&self, command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.queue().submit(command_buffers);
    }

    pub fn compute(&self, pass: &ComputePass, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        ComputeBatch::submit(self.device(), self.queue(), &[(pass, workgroups_x, workgroups_y, workgroups_z)]);
    }

    pub fn compute_batch(&self, passes: &[(&ComputePass, u32, u32, u32)]) {
        ComputeBatch::submit(self.device(), self.queue(), passes);
    }

    pub(crate) fn shared_layouts(&self) -> &SharedLayouts {
        self.shared_layouts.as_ref().expect("Renderer not initialized")
    }

    pub fn presentation_format(&self) -> wgpu::TextureFormat {
        self.presentation_format
    }

    pub fn surface(&self) -> Option<&wgpu::Surface<'static>> {
        self.surface.as_ref()
    }

    pub fn width(&self) -> u32 { self.config.width }
    pub fn height(&self) -> u32 { self.config.height }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        if self.device.is_none() { return; }
        // Configure surface
        if let (Some(ref surface), Some(ref mut config)) = (&self.surface, &mut self.surface_config) {
            config.width = width;
            config.height = height;
            surface.configure(self.device.as_ref().unwrap(), config);
        }
        // Recreate depth/MSAA (separate borrow scope)
        self._recreate_size_dependent();
    }

    fn _recreate_size_dependent(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let sc = self.config.sample_count;
        let fmt = self.presentation_format;
        let device = self.device.as_ref().unwrap();

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        if sc > 1 {
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/MSAA"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: sc,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
            });
            self.msaa_view = Some(msaa.create_view(&Default::default()));
            self.msaa_texture = Some(msaa);
        }
    }

    fn _create_depth_texture(&mut self, device: &wgpu::Device) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Depth"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.config.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        if self.config.sample_count > 1 {
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/MSAA"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.config.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.presentation_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = Some(msaa.create_view(&Default::default()));
            self.msaa_texture = Some(msaa);
        }
    }

    /// Ensure shared per-object matrix buffers are large enough.
    fn _ensure_matrix_buffers(&mut self, count: usize) {
        if count <= self.last_object_count && self.world_matrices_buf.is_some() {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let alignment = self.matrix_alignment as usize;
        let floats_per_slot = alignment / 4;
        let total_floats = count * floats_per_slot;
        let total_bytes = (total_floats * 4) as u64;

        self.world_matrices_staging.resize(total_floats, 0.0);
        self.normal_matrices_staging.resize(total_floats, 0.0);

        let world_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/WorldMatrices"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let normal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/NormalMatrices"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Rebuild mesh bind group (group 1) to point at the new buffers
        let shared = self.shared_layouts.as_ref().unwrap();
        self.mesh_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/MeshBG"),
            layout: &shared.mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &normal_buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &world_buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
            ],
        }));

        self.world_matrices_buf = Some(world_buf);
        self.normal_matrices_buf = Some(normal_buf);
        self.last_object_count = count;
    }

    /// Upload camera + per-object matrices to GPU.
    fn upload_all(&mut self, scene: &Scene, camera: &Camera) {
        let queue = self.queue.as_ref().unwrap();

        // Upload camera matrices (camera owns its buffers)
        camera.upload(queue);

        // Upload lights
        let lights_vec: Vec<&Light> = scene.lights().collect();
        self.light_uniforms.pack_refs(&lights_vec);
        if let Some(ref buf) = self.light_buf {
            queue.write_buffer(buf, 0, self.light_uniforms.as_bytes());
        }

        // Upload per-object matrices
        let count = scene.len();
        self._ensure_matrix_buffers(count);

        let alignment = self.matrix_alignment as usize;
        let floats_per_slot = alignment / 4;

        for (i, idx) in scene.ordered_indices().enumerate() {
            if let Some(renderable) = scene.get_renderable(idx) {
                let offset = i * floats_per_slot;
                self.world_matrices_staging[offset..offset + 16]
                    .copy_from_slice(renderable.world_matrix.as_slice());
                self.normal_matrices_staging[offset..offset + 16]
                    .copy_from_slice(renderable.normal_matrix.as_slice());
            }
        }

        let queue = self.queue.as_ref().unwrap();
        if count > 0 {
            if let Some(ref buf) = self.world_matrices_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.world_matrices_staging));
            }
            if let Some(ref buf) = self.normal_matrices_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.normal_matrices_staging));
            }
        }
    }

    /// Upload all scene object matrices to shared GPU buffers.
    /// Public wrapper around upload_all for backward compatibility.
    pub fn upload_matrices(&mut self, scene: &Scene, camera: &Camera) {
        self.upload_all(scene, camera);
    }

    /// Enable directional shadow mapping.
    pub fn enable_shadows(&mut self, resolution: u32) {
        let device = self.device.as_ref().unwrap();
        let mut sm = crate::shadows::ShadowMap::new(resolution);
        sm.initialize(device);

        // Rebuild shadow bind group with real depth texture
        let shared = self.shared_layouts.as_ref().unwrap();
        self.shadow_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/ShadowBG"),
            layout: &shared.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(sm.depth_view.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.shadow_comparison_sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.shadow_uniform_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        self.cube_dummy_view.as_ref().unwrap()
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        self.cube_shadow_sampler.as_ref().unwrap()
                    ),
                },
            ],
        }));

        // Shadow pipeline (depth-only, no fragment)
        let shadow_light_vp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/LightVPBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow/PipelineLayout"),
            bind_group_layouts: &[&shadow_light_vp_bgl, &shared.camera_bgl, &shared.mesh_bgl],
            push_constant_ranges: &[],
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_vs.wgsl").into()),
        });

        self.shadow_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow/Pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("shadow_vs"),
                buffers: &[crate::geometries::Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        }));

        // Light VP bind group
        self.shadow_light_vp_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow/LightVPBG"),
            layout: &shadow_light_vp_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sm.light_vp_buf.as_ref().unwrap().as_entire_binding(),
            }],
        }));

        self.shadow_light_vp_bgl = Some(shadow_light_vp_bgl);
        self.shadow_map = Some(sm);
        self.shadows_enabled = true;
    }

    /// Render the scene to the canvas surface (simple path).
    pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera) {
        // Phase 0: Update transforms, prepare scene
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Phase 0.5: Initialize geometries, instance buffers, and pre-warm pipelines
        {
            let device = self.device.as_ref().unwrap();
            let queue = self.queue.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            let format = self.presentation_format;
            let sample_count = self.config.sample_count;
            let depth_format = wgpu::TextureFormat::Depth24Plus;

            let ordered_indices: Vec<usize> = scene.ordered_indices().collect();
            for idx in ordered_indices {
                let r = scene.get_renderable_mut(idx).expect("ordered scene index should exist");
                if !r.geometry.initialized {
                    r.geometry.initialize(device);
                }
                for ib in &mut r.instance_buffers {
                    if !ib.initialized {
                        ib.initialize(device, queue);
                    }
                }
                r.material.ensure_bindables_initialized(self);
                r.material.initialize(device, shared);

                // Build combined vertex layouts (base + instance buffers)
                let instance_layouts: Vec<_> = r.instance_buffers.iter()
                    .map(|ib| ib.vertex_layout())
                    .collect();
                let mut layouts = vec![Vertex::LAYOUT];
                for il in &instance_layouts {
                    layouts.push(il.as_layout());
                }

                r.material.get_pipeline(
                    device, &layouts,
                    &[format], depth_format, sample_count,
                );
            }
        }

        // Initialize camera GPU resources if needed
        if !camera.initialized {
            let device = self.device.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            camera.gpu_initialize(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap());
        }

        // Phase 1: Upload camera + per-object matrices
        self.upload_all(scene, camera);

        // Shadow pass
        if self.shadows_enabled {
            if let Some(ref mut sm) = self.shadow_map {
                // Find first directional light
                let dir_light_dir = scene.lights().find_map(|l| {
                    if let crate::lights::Light::Directional(dl) = l { Some(dl.direction) } else { None }
                });

                if let Some(light_dir) = dir_light_dir {
                    sm.compute_light_vp(camera, &light_dir);
                    sm.upload(self.queue.as_ref().unwrap());

                    // Upload shadow uniforms (96 bytes)
                    let mut shadow_data = [0.0f32; 24];
                    shadow_data[..16].copy_from_slice(sm.light_vp.as_slice());
                    shadow_data[16] = sm.bias;
                    shadow_data[17] = sm.normal_bias;
                    shadow_data[18] = 1.0; // shadowEnabled
                    shadow_data[19] = 0.0; // pointShadowEnabled
                    shadow_data[20] = 0.0; // pointLightPos.x
                    shadow_data[21] = 0.0; // pointLightPos.y
                    shadow_data[22] = 0.0; // pointLightPos.z
                    shadow_data[23] = 0.0; // pointShadowFar
                    if let Some(ref buf) = self.shadow_uniform_buf {
                        self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
                    }

                    // Render shadow depth pass
                    let device = self.device.as_ref().unwrap();
                    let mut shadow_encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = shadow_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Renderer/ShadowPass"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: sm.depth_view.as_ref().unwrap(),
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });

                        pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
                        pass.set_bind_group(0, self.shadow_light_vp_bg.as_ref().unwrap(), &[]);

                        let alignment = self.matrix_alignment;
                        for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                            let r = scene.get_renderable(scene_idx).unwrap();
                            if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                                continue;
                            }

                            let offset = (draw_idx as u32) * alignment;
                            pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                            pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                            pass.set_index_buffer(r.geometry.index_buffer.as_ref().unwrap().slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                        }
                    }
                    self.queue.as_ref().unwrap().submit(std::iter::once(shadow_encoder.finish()));
                }
            }
        }

        // Upload disabled shadow uniforms when shadows are off
        if !self.shadows_enabled {
            let shadow_data = [0.0f32; 24];
            if let Some(ref buf) = self.shadow_uniform_buf {
                self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
            }
        }

        // Phase 2+3: Create render pass and draw
        let surface = self.surface.as_ref().unwrap();
        let output = surface.get_current_texture().expect("Failed to get surface texture");
        let canvas_view = output.texture.create_view(&Default::default());

        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/RenderEncoder"),
        });

        let cc = &self.config.clear_color;
        let format = self.presentation_format;
        let sample_count = self.config.sample_count;
        let depth_format = wgpu::TextureFormat::Depth24Plus;
        let alignment = self.matrix_alignment;

        {
            let color_view = if sample_count > 1 {
                self.msaa_view.as_ref().unwrap()
            } else {
                &canvas_view
            };
            let resolve = if sample_count > 1 { Some(&canvas_view) } else { None };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/MainPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Set camera bind group (shared across all objects)
            pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);
            pass.set_bind_group(3, self.shadow_bind_group.as_ref().unwrap(), &[]);

            for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                let r = scene.get_renderable(scene_idx).unwrap();
                if !r.visible || !r.geometry.initialized { continue; }

                // Per-renderable pipeline key (includes num_vertex_buffers)
                let num_vb = 1 + r.instance_buffers.len();
                let key = crate::materials::PipelineKey {
                    color_formats: vec![format],
                    depth_format,
                    sample_count,
                    num_vertex_buffers: num_vb,
                };

                // Get the pre-warmed pipeline
                let pipeline = match r.material.pipeline_cache.get(&key) {
                    Some(p) => p,
                    None => continue,
                };

                pass.set_pipeline(pipeline);

                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }

                let offset = (draw_idx as u32) * alignment;
                pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                // Slot 0: base geometry vertex buffer
                pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                // Slot 1+: instance buffers
                for (i, ib) in r.instance_buffers.iter().enumerate() {
                    if let Some(ref buf) = ib.gpu_buffer {
                        pass.set_vertex_buffer((i + 1) as u32, buf.slice(..));
                    }
                }

                pass.set_index_buffer(
                    r.geometry.index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
            }
        }

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Render scene with post-processing effects.
    pub fn render_with_postprocessing(
        &mut self,
        scene: &mut Scene,
        camera: &mut Camera,
        volume: &mut crate::postprocessing::PostProcessingVolume,
    ) {
        let width = self.config.width;
        let height = self.config.height;
        volume.ensure_gbuffer(width, height);

        // Render scene to GBuffer (actually draw into it)
        {
            let gbuffer = volume.gbuffer().unwrap();
            self.render_scene_to_gbuffer(scene, camera, gbuffer);
        }

        // Get surface texture for blit
        let surface = self.surface.as_ref().unwrap();
        let output = surface.get_current_texture().expect("Surface texture");
        let canvas_view = output.texture.create_view(&Default::default());

        // Run post-processing chain + blit
        volume.render(camera, &canvas_view, width, height);

        output.present();
    }

    /// Private: draw scene into GBuffer MRT (non-MSAA, sample_count=1).
    fn render_scene_to_gbuffer(
        &mut self,
        scene: &mut Scene,
        camera: &mut Camera,
        gbuffer: &GBuffer,
    ) {
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Initialize camera GPU resources if needed
        if !camera.initialized {
            let device = self.device.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            camera.gpu_initialize(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap());
        }

        // Initialize geometries + pre-warm pipelines for GBuffer formats
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        let depth_format = GBuffer::DEPTH_FORMAT;
        let sample_count = gbuffer.sample_count;

        let ordered_indices: Vec<usize> = scene.ordered_indices().collect();
        for idx in ordered_indices {
            let r = scene.get_renderable_mut(idx).expect("ordered scene index should exist");
            if !r.geometry.initialized {
                r.geometry.initialize(device);
            }
            for ib in &mut r.instance_buffers {
                if !ib.initialized {
                    ib.initialize(device, queue);
                }
            }
            r.material.ensure_bindables_initialized(self);
            r.material.initialize(device, shared);

            let instance_layouts: Vec<_> = r.instance_buffers.iter()
                .map(|ib| ib.vertex_layout())
                .collect();
            let mut layouts = vec![Vertex::LAYOUT];
            for il in &instance_layouts {
                layouts.push(il.as_layout());
            }

            r.material.get_pipeline(
                device, &layouts,
                &GBuffer::MRT_FORMATS, depth_format, sample_count,
            );
        }

        // Upload camera + per-object matrices
        self.upload_all(scene, camera);

        // Shadow pass (if enabled)
        if self.shadows_enabled {
            if let Some(ref mut sm) = self.shadow_map {
                let dir_light_dir = scene.lights().find_map(|l| {
                    if let crate::lights::Light::Directional(dl) = l { Some(dl.direction) } else { None }
                });

                if let Some(light_dir) = dir_light_dir {
                    sm.compute_light_vp(camera, &light_dir);
                    sm.upload(self.queue.as_ref().unwrap());

                    let mut shadow_data = [0.0f32; 24];
                    shadow_data[..16].copy_from_slice(sm.light_vp.as_slice());
                    shadow_data[16] = sm.bias;
                    shadow_data[17] = sm.normal_bias;
                    shadow_data[18] = 1.0; // shadowEnabled
                    if let Some(ref buf) = self.shadow_uniform_buf {
                        self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
                    }

                    let device = self.device.as_ref().unwrap();
                    let mut shadow_encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = shadow_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Shadow/GBufferPath"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: sm.depth_view.as_ref().unwrap(),
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });

                        pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
                        pass.set_bind_group(0, self.shadow_light_vp_bg.as_ref().unwrap(), &[]);

                        let alignment = self.matrix_alignment;
                        for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                            let r = scene.get_renderable(scene_idx).unwrap();
                            if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                                continue;
                            }

                            let offset = (draw_idx as u32) * alignment;
                            pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);
                            pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                            pass.set_index_buffer(r.geometry.index_buffer.as_ref().unwrap().slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                        }
                    }
                    self.queue.as_ref().unwrap().submit(std::iter::once(shadow_encoder.finish()));
                }
            }
        } else {
            let shadow_data = [0.0f32; 24];
            if let Some(ref buf) = self.shadow_uniform_buf {
                self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
            }
        }

        // GBuffer MRT render pass (non-MSAA)
        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/GBufferDraw"),
        });
        let cc = &self.config.clear_color;
        let clear = wgpu::Color { r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64 };
        let black = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferDrawPass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.emissive_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.albedo_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);
            pass.set_bind_group(3, self.shadow_bind_group.as_ref().unwrap(), &[]);

            let alignment = self.matrix_alignment;

            for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                let r = scene.get_renderable(scene_idx).unwrap();
                if !r.visible || !r.geometry.initialized { continue; }

                let num_vb = 1 + r.instance_buffers.len();
                let key = crate::materials::PipelineKey {
                    color_formats: GBuffer::MRT_FORMATS.to_vec(),
                    depth_format: GBuffer::DEPTH_FORMAT,
                    sample_count: gbuffer.sample_count,
                    num_vertex_buffers: num_vb,
                };

                let pipeline = match r.material.pipeline_cache.get(&key) {
                    Some(p) => p,
                    None => continue,
                };

                pass.set_pipeline(pipeline);

                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }

                let offset = (draw_idx as u32) * alignment;
                pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                for (i, ib) in r.instance_buffers.iter().enumerate() {
                    if let Some(ref buf) = ib.gpu_buffer {
                        pass.set_vertex_buffer((i + 1) as u32, buf.slice(..));
                    }
                }

                pass.set_index_buffer(
                    r.geometry.index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
            }
        }

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
    }

    /// Render scene into a GBuffer for post-processing.
    pub fn render_to_gbuffer(&mut self, scene: &mut Scene, camera: &mut Camera, gbuffer: &GBuffer) {
        camera.update_view_matrix();
        scene.prepare(camera.position());
        self.upload_all(scene, camera);

        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/GBufferEncoder"),
        });

        let cc = &self.config.clear_color;
        let clear = wgpu::Color { r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64 };
        let black = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

        if gbuffer.sample_count > 1 {
            // MSAA path — clear only for now (GBuffer draw loop comes in Plan 2)
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferPass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.color_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.color_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Discard },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.emissive_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.emissive_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.normal_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.normal_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.albedo_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.albedo_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: gbuffer.depth_msaa_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        } else {
            // Non-MSAA path — clear only for now (GBuffer draw loop comes in Plan 2)
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferPass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.emissive_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.albedo_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
    }
}
