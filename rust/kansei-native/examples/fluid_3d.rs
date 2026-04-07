use std::sync::Arc;
use std::time::Instant;
use kansei_core::math::{Mat4, Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::objects::Scene;
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    FluidSimulation, FluidSimulationOptions, FluidDensityField, DensityFieldOptions, FluidSurfaceRenderer,
    FluidParticleRenderer as FluidParticleRendererCore,
    FullscreenBlit as FullscreenBlitCore,
};

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

// ── App ──
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    camera_controls: CameraControls,
    camera_dragging: bool,
    last_mouse: Option<(f64, f64)>,
    sim: Option<FluidSimulation>,
    density_field: Option<FluidDensityField>,
    surface_renderer: Option<FluidSurfaceRenderer>,
    particle_renderer: Option<FluidParticleRendererCore>,
    blit: Option<FullscreenBlitCore>,
    color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    output_view: Option<wgpu::TextureView>,
    color_texture: Option<wgpu::Texture>,
    surface_bg: Option<wgpu::BindGroup>,
    blit_bg: Option<wgpu::BindGroup>,
    // egui
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    // state
    show_particles: bool,
    particle_size: f32,
    last_time: Option<Instant>,
    // mouse for fluid interaction
    mouse_vectors: MouseVectors,
    mouse_pressed: bool,
    enable_mouse_fluid_interaction: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None, renderer: None, scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 1000.0, 1.0),
            camera_controls: {
                let mut c = CameraControls::new(Vec3::new(0.0, 3.0, 0.0), 75.0);
                c.rotate(0.0, 0.3);
                c
            },
            camera_dragging: false,
            last_mouse: None,
            sim: None, density_field: None, surface_renderer: None,
            particle_renderer: None, blit: None,
            color_view: None, depth_view: None, output_view: None, color_texture: None,
            surface_bg: None, blit_bg: None,
            egui_ctx: egui::Context::default(),
            egui_state: None, egui_renderer: None,
            show_particles: true, particle_size: 0.15,
            last_time: None,
            mouse_vectors: MouseVectors::new(), mouse_pressed: false, enable_mouse_fluid_interaction: true,
        }
    }

    fn rebuild_offscreen(&mut self, w: u32, h: u32) {
        let renderer = self.renderer.as_ref().unwrap();
        let mk = |label: &str, fmt: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            let tex = renderer.device().create_texture(&wgpu::TextureDescriptor {
                label: Some(label), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: fmt, usage, view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };
        let cu = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let (ct, cv) = mk("Color", wgpu::TextureFormat::Rgba16Float, cu);
        let (_dt, dv) = mk("Depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let (_ot, ov) = mk("Output", wgpu::TextureFormat::Rgba16Float, cu);
        self.color_texture = Some(ct); self.color_view = Some(cv);
        self.depth_view = Some(dv); self.output_view = Some(ov);
        // Rebuild bind groups
        if let (Some(sr), Some(df), Some(blit)) = (&self.surface_renderer, &self.density_field, &self.blit) {
            self.surface_bg = Some(sr.create_bind_group(self.color_view.as_ref().unwrap(), self.depth_view.as_ref().unwrap(), self.output_view.as_ref().unwrap(), &df.density_view));
            self.blit_bg = Some(blit.create_bind_group(renderer, self.output_view.as_ref().unwrap()));
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let window = Arc::new(el.create_window(Window::default_attributes()
            .with_title("Kansei — Fluid 3D").with_inner_size(winit::dpi::LogicalSize::new(1280, 720))).unwrap());
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() })).unwrap();
        let mut renderer = Renderer::new(RendererConfig { width: size.width, height: size.height, sample_count: 1, clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0), ..Default::default() });
        pollster::block_on(renderer.initialize(surface, &adapter));
        let format = renderer.presentation_format();

        // Particles
        let count = 10000usize;
        let radius = 5.0f32;
        let mut positions = vec![0.0f32; count * 4];
        let mut rng: u64 = 12345;
        for i in 0..count {
            loop {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let x = ((rng as f32 / u64::MAX as f32) * 2.0 - 1.0) * radius;
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let y = ((rng as f32 / u64::MAX as f32) * 2.0 - 1.0) * radius;
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                let z = ((rng as f32 / u64::MAX as f32) * 2.0 - 1.0) * radius;
                if x*x + y*y + z*z <= radius * radius { positions[i*4]=x; positions[i*4+1]=y; positions[i*4+2]=z; positions[i*4+3]=1.0; break; }
            }
        }

        let mut sim = FluidSimulation::new(&renderer, FluidSimulationOptions {
            max_particles: count as u32, dimensions: 3, smoothing_radius: 1.0, pressure_multiplier: 46.5,
            near_pressure_multiplier: 20.0, density_target: 8.6, viscosity: 0.36, damping: 0.998,
            gravity: [0.0, -9.8, 0.0], mouse_force: 1520.0, substeps: 3, world_bounds_padding: 0.3,
            ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
        }, &positions);
        sim.world_bounds_min = [-12.0, -8.0, -8.0];
        sim.world_bounds_max = [12.0, 32.0, 8.0];
        sim.rebuild_grid();

        self.density_field = Some(FluidDensityField::new(&renderer, sim.positions_buffer().unwrap(), sim.world_bounds_min, sim.world_bounds_max, DensityFieldOptions { resolution: 64, kernel_scale: 3.7 }));
        self.surface_renderer = Some(FluidSurfaceRenderer::new(&renderer));
        self.particle_renderer = Some(FluidParticleRendererCore::new(
            &renderer,
            sim.positions_buffer().unwrap(),
            count as u32,
            format,
            1,
            None,
        ));
        self.blit = Some(FullscreenBlitCore::new(&renderer, format));
        self.sim = Some(sim);
        self.renderer = Some(renderer);

        // egui
        self.egui_state = Some(egui_winit::State::new(self.egui_ctx.clone(), egui::ViewportId::ROOT, &*window, Some(window.scale_factor() as f32), None, None));
        self.egui_renderer = Some(self.renderer.as_ref().unwrap().egui_create_renderer(format, 1));

        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();
        self.window = Some(window);
        self.last_time = Some(Instant::now());
        self.rebuild_offscreen(size.width, size.height);
        log::info!("Fluid 3D — {} particles", count);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui consume events first
        if let Some(ref mut state) = self.egui_state {
            let response = state.on_window_event(self.window.as_ref().unwrap(), &event);
            if response.consumed { return; }
        }

        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                if self.renderer.is_some() {
                    self.renderer.as_mut().unwrap().resize(s.width, s.height);
                    self.camera.aspect = s.width as f32 / s.height as f32;
                    self.camera.update_projection_matrix();
                    self.color_texture = None; // invalidate
                }
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state: s, .. } => {
                self.camera_dragging = s == ElementState::Pressed;
                self.mouse_pressed = s == ElementState::Pressed;
            }
            WindowEvent::MouseInput { button: MouseButton::Right, state: s, .. } => {
                self.mouse_pressed = s == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let first_mouse = self.last_mouse.is_none();
                if self.camera_dragging {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = (position.x - lx) as f32 * 0.005;
                        let dy = (position.y - ly) as f32 * 0.005;
                        self.camera_controls.rotate(-dx, dy);
                    }
                }
                self.last_mouse = Some((position.x, position.y));
                // Track mouse in NDC for fluid interaction
                if let Some(ref w) = self.window {
                    let s = w.inner_size();
                    if first_mouse {
                        self.mouse_vectors.set_position_from_screen(position.x as f32, position.y as f32, s.width as f32, s.height as f32);
                    } else {
                        self.mouse_vectors.set_target_from_screen(position.x as f32, position.y as f32, s.width as f32, s.height as f32);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.camera_controls.radius = (self.camera_controls.radius - dy * 0.5).clamp(2.0, 100.0);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self.last_time.map(|t| now.duration_since(t).as_secs_f32()).unwrap_or(1.0/60.0).min(1.0/30.0);
                self.last_time = Some(now);
                let size = self.window.as_ref().unwrap().inner_size();

                if self.color_texture.is_none() { self.rebuild_offscreen(size.width, size.height); }

                let renderer = self.renderer.as_ref().unwrap();

                // Camera orbit update via engine CameraControls
                self.camera_controls.update(&mut self.camera, dt);
                let aspect = size.width as f32 / size.height as f32;
                self.camera.aspect = aspect;
                self.camera.update_projection_matrix();
                let eye = *self.camera.position();
                let view = self.camera.view_matrix;
                let proj = self.camera.projection_matrix;
                let inv_view = self.camera.inverse_view_matrix;
                let inv_vp = (proj * view).inverse();

                // Mouse interaction
                self.mouse_vectors.update(dt);
                let (mouse_dir, mouse_pos, mouse_strength) = if self.enable_mouse_fluid_interaction {
                    (
                        [self.mouse_vectors.direction.x, self.mouse_vectors.direction.y],
                        [self.mouse_vectors.position.x, self.mouse_vectors.position.y],
                        self.mouse_vectors.strength,
                    )
                } else {
                    ([0.0, 0.0], [0.0, 0.0], 0.0)
                };

                // Upload camera matrices + run sim
                if let Some(ref mut sim) = self.sim {
                    let identity = Mat4::identity();
                    sim.set_camera_matrices(
                        view.as_slice(), proj.as_slice(),
                        inv_view.as_slice(), identity.as_slice());
                    sim.update(dt, mouse_strength, mouse_pos, mouse_dir);
                }

                let surface = renderer.surface().unwrap();
                let output = surface.get_current_texture().expect("Surface texture");
                let canvas_view = output.texture.create_view(&Default::default());
                let mut encoder = renderer.create_command_encoder(&Default::default());

                if self.show_particles {
                    if let Some(ref pr) = self.particle_renderer {
                        pr.upload(renderer, &view, &proj, self.particle_size);
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &canvas_view, resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }), store: wgpu::StoreOp::Store },
                            })], ..Default::default()
                        });
                        pass.set_pipeline(&pr.pipeline);
                        pass.set_bind_group(0, &pr.bind_group, &[]);
                        pass.draw(0..pr.count * 6, 0..1);
                    }
                } else {
                    let sim = self.sim.as_ref().unwrap();
                    if let Some(ref mut df) = self.density_field {
                        df.update_with_encoder(&mut encoder, sim.world_bounds_min, sim.world_bounds_max, sim.particle_count(), sim.params.smoothing_radius);
                    }
                    { let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: self.color_view.as_ref().unwrap(), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: self.depth_view.as_ref().unwrap(), depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                        ..Default::default()
                    }); }
                    if let Some(ref sr) = self.surface_renderer {
                        sr.render(&mut encoder, self.surface_bg.as_ref().unwrap(), &inv_vp, [eye.x, eye.y, eye.z], sim.world_bounds_min, sim.world_bounds_max, size.width, size.height);
                    }
                    { let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &canvas_view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })], ..Default::default()
                    });
                    if let Some(ref blit) = self.blit { pass.set_pipeline(&blit.pipeline); pass.set_bind_group(0, self.blit_bg.as_ref().unwrap(), &[]); pass.draw(0..3, 0..1); } }
                }

                // egui
                let raw_input = self.egui_state.as_mut().unwrap().take_egui_input(self.window.as_ref().unwrap());
                let full_output = self.egui_ctx.run(raw_input, |ctx| {
                    egui::SidePanel::right("controls").min_width(220.0).show(ctx, |ui| {
                        ui.heading("Fluid 3D");
                        ui.separator();
                        ui.checkbox(&mut self.show_particles, "Show particles");
                        ui.add(egui::Slider::new(&mut self.particle_size, 0.01..=0.5).text("Particle size"));
                        ui.separator();
                        if let Some(ref mut sim) = self.sim {
                            ui.label("SPH");
                            ui.add(egui::Slider::new(&mut sim.params.pressure_multiplier, 0.0..=100.0).text("Pressure"));
                            ui.add(egui::Slider::new(&mut sim.params.near_pressure_multiplier, 0.0..=100.0).text("Near pressure"));
                            ui.add(egui::Slider::new(&mut sim.params.density_target, 0.0..=20.0).text("Density target"));
                            ui.add(egui::Slider::new(&mut sim.params.viscosity, 0.0..=1.0).text("Viscosity"));
                            ui.add(egui::Slider::new(&mut sim.params.damping, 0.9..=1.0).text("Damping"));
                            ui.separator();
                            ui.label("Forces");
                            ui.add(egui::Slider::new(&mut sim.params.gravity[1], -20.0..=0.0).text("Gravity Y"));
                            ui.checkbox(&mut self.enable_mouse_fluid_interaction, "Mouse affects fluid");
                        }
                        if let Some(ref mut sr) = self.surface_renderer {
                            ui.separator();
                            ui.label("Surface");
                            ui.add(egui::Slider::new(&mut sr.density_scale, 0.1..=10.0).text("Density scale"));
                            ui.add(egui::Slider::new(&mut sr.density_threshold, 0.01..=5.0).text("Threshold"));
                            ui.add(egui::Slider::new(&mut sr.absorption, 0.0..=10.0).text("Absorption"));
                        }
                        ui.separator();
                        ui.label(format!(
                            "Camera: dist={:.1} az={:.2} el={:.2}",
                            self.camera_controls.radius,
                            self.camera_controls.azimuth(),
                            self.camera_controls.elevation()
                        ));
                    });
                });

                // Submit scene encoder first
                renderer.submit(std::iter::once(encoder.finish()));

                // egui in a separate encoder
                let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                let screen_descriptor = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.width, size.height], pixels_per_point: full_output.pixels_per_point };

                // Take egui renderer out of self to avoid borrow conflicts
                let mut er = self.egui_renderer.take().unwrap();
                renderer.egui_upload(&mut er, &full_output.textures_delta, &paint_jobs, &screen_descriptor);
                renderer.egui_render(&mut er, &paint_jobs, &screen_descriptor, &canvas_view);
                renderer.egui_free_textures(&mut er, &full_output.textures_delta);
                self.egui_renderer = Some(er);
                self.egui_state.as_mut().unwrap().handle_platform_output(self.window.as_ref().unwrap(), full_output.platform_output);
                output.present();
                if let Some(ref w) = self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Kansei — Fluid 3D");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
