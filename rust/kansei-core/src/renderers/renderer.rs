use crate::math::Vec4;
use crate::cameras::Camera;
use crate::geometries::Vertex;
use crate::objects::Scene;
use super::gbuffer::GBuffer;
use super::shared_layouts::SharedLayouts;

/// Core WebGPU renderer configuration.
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub device_pixel_ratio: f32,
    pub sample_count: u32,
    pub clear_color: Vec4,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            device_pixel_ratio: 1.0,
            sample_count: 4,
            clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
    // Camera uniform buffers + bind group (group 2)
    camera_view_buf: Option<wgpu::Buffer>,
    camera_proj_buf: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
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
            camera_view_buf: None,
            camera_proj_buf: None,
            camera_bind_group: None,
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

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: self.config.width,
            height: self.config.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        self.presentation_format = format;
        self._create_depth_texture(&device);

        // Create shared bind group layouts
        let shared = SharedLayouts::new(&device);

        // Create camera uniform buffers (64 bytes = one mat4x4<f32>)
        let camera_view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/CameraViewBuf"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_proj_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/CameraProjBuf"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create camera bind group (group 2)
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/CameraBG"),
            layout: &shared.camera_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_view_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_proj_buf.as_entire_binding(),
                },
            ],
        });

        self.shared_layouts = Some(shared);
        self.camera_view_buf = Some(camera_view_buf);
        self.camera_proj_buf = Some(camera_proj_buf);
        self.camera_bind_group = Some(camera_bind_group);

        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
    }

    pub fn device(&self) -> &wgpu::Device {
        self.device.as_ref().expect("Renderer not initialized")
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.queue.as_ref().expect("Renderer not initialized")
    }

    pub fn shared_layouts(&self) -> &SharedLayouts {
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

        // Upload camera matrices
        if let Some(ref buf) = self.camera_view_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(camera.view_matrix.as_slice()));
        }
        if let Some(ref buf) = self.camera_proj_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(camera.projection_matrix.as_slice()));
        }

        // Upload per-object matrices
        let count = scene.len();
        self._ensure_matrix_buffers(count);

        let alignment = self.matrix_alignment as usize;
        let floats_per_slot = alignment / 4;

        for (i, idx) in scene.ordered_indices().enumerate() {
            if let Some(renderable) = scene.get(idx) {
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

    /// Render the scene to the canvas surface (simple path).
    pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera) {
        // Phase 0: Update transforms, prepare scene
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Phase 0.5: Initialize geometries, instance buffers, and pre-warm pipelines
        {
            let device = self.device.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            let format = self.presentation_format;
            let sample_count = self.config.sample_count;
            let depth_format = wgpu::TextureFormat::Depth24Plus;

            for r in scene.renderables_mut() {
                if !r.geometry.initialized {
                    r.geometry.initialize(device);
                }
                for ib in &mut r.instance_buffers {
                    if !ib.initialized {
                        ib.initialize(device);
                    }
                }

                // Build combined vertex layouts (base + instance buffers)
                let instance_layouts: Vec<_> = r.instance_buffers.iter()
                    .map(|ib| ib.vertex_layout())
                    .collect();
                let mut layouts = vec![Vertex::LAYOUT];
                for il in &instance_layouts {
                    layouts.push(il.as_layout());
                }

                r.material.get_pipeline(
                    device, shared, &layouts,
                    &[format], depth_format, sample_count,
                );
            }
        }

        // Phase 1: Upload camera + per-object matrices
        self.upload_all(scene, camera);

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
            pass.set_bind_group(2, self.camera_bind_group.as_ref().unwrap(), &[]);

            for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                let r = &scene.renderables()[scene_idx];
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
                pass.set_bind_group(1, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

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
