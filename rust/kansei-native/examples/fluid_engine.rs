use std::sync::Arc;
use std::time::Instant;
use kansei_core::math::{Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::objects::Scene;
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    FluidSimulation, FluidSimulationOptions, FluidDensityField, DensityFieldOptions, FluidSurfaceRenderer,
};

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

// ── Orbit camera ──
struct OrbitCamera {
    target: glam::Vec3,
    distance: f32,
    azimuth: f32,   // radians
    elevation: f32,  // radians
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

impl OrbitCamera {
    fn new(target: glam::Vec3, distance: f32) -> Self {
        Self { target, distance, azimuth: 0.0, elevation: 0.3, dragging: false, last_mouse: None }
    }

    fn eye(&self) -> glam::Vec3 {
        let x = self.target.x + self.distance * self.azimuth.sin() * self.elevation.cos();
        let y = self.target.y + self.distance * self.elevation.sin();
        let z = self.target.z + self.distance * self.azimuth.cos() * self.elevation.cos();
        glam::Vec3::new(x, y, z)
    }

    fn on_mouse_move(&mut self, x: f64, y: f64) {
        if self.dragging {
            if let Some((lx, ly)) = self.last_mouse {
                let dx = (x - lx) as f32 * 0.005;
                let dy = (y - ly) as f32 * 0.005;
                self.azimuth -= dx;
                self.elevation = (self.elevation + dy).clamp(-1.5, 1.5);
            }
        }
        self.last_mouse = Some((x, y));
    }

    fn on_scroll(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.5).clamp(2.0, 100.0);
    }
}

// ── Blit pipeline ──
struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    bgl: wgpu::BindGroupLayout,
}

impl BlitPipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit"), source: wgpu::ShaderSource::Wgsl(r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4<f32>(pos[vi], 0.0, 1.0);
}
struct FOut { @location(0) color: vec4<f32>, }
@fragment fn fs(@builtin(position) frag: vec4<f32>) -> FOut {
    let dims = vec2<f32>(textureDimensions(src));
    var out: FOut; out.color = textureSample(src, samp, frag.xy / dims); return out;
}
"#.into()) });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("Blit/BGL"), entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ]});
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        Self { pipeline, sampler, bgl }
    }
    fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
        ]})
    }
}

// ── Particle renderer ──
struct ParticleRenderer { pipeline: wgpu::RenderPipeline, bind_group: wgpu::BindGroup, params_buf: wgpu::Buffer, count: u32 }

impl ParticleRenderer {
    fn new(device: &wgpu::Device, pos_buf: &wgpu::Buffer, count: u32, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Particles"), source: wgpu::ShaderSource::Wgsl(r#"
struct P { view: mat4x4<f32>, proj: mat4x4<f32>, size: f32, }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p: P;
struct V { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32>, }
const Q: array<vec2<f32>,6> = array(vec2(-1.,-1.),vec2(1.,-1.),vec2(1.,1.),vec2(-1.,-1.),vec2(1.,1.),vec2(-1.,1.));
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    let pid=vi/6u; let c=Q[vi%6u]; let pos=positions[pid]; let s=p.size;
    let r=vec3<f32>(p.view[0][0],p.view[1][0],p.view[2][0]);
    let u=vec3<f32>(p.view[0][1],p.view[1][1],p.view[2][1]);
    let wp=pos.xyz+r*c.x*s+u*c.y*s;
    var o:V; o.pos=p.proj*p.view*vec4<f32>(wp,1.);
    let t=clamp((pos.y+8.)/16.,0.,1.);
    o.col=mix(vec3<f32>(0.1,0.3,0.8),vec3<f32>(0.8,0.95,1.0),t); return o;
}
@fragment fn fs(v:V)->@location(0) vec4<f32>{return vec4<f32>(v.col,1.);}
"#.into()) });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 144, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particles"), layout: None,
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let bgl = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
        ]});
        Self { pipeline, bind_group, params_buf, count }
    }
    fn upload(&self, queue: &wgpu::Queue, view: &glam::Mat4, proj: &glam::Mat4, size: f32) {
        let mut d = [0.0f32; 36]; d[..16].copy_from_slice(&view.to_cols_array()); d[16..32].copy_from_slice(&proj.to_cols_array()); d[32] = size;
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&d));
    }
}

// ── App ──
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    #[allow(dead_code)]
    scene: Scene,
    camera: Camera,
    orbit: OrbitCamera,
    sim: Option<FluidSimulation>,
    density_field: Option<FluidDensityField>,
    surface_renderer: Option<FluidSurfaceRenderer>,
    particle_renderer: Option<ParticleRenderer>,
    blit: Option<BlitPipeline>,
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
    // FPS counter
    frame_count: u64,
    frame_time_accum: f64,
    current_fps: f64,
    current_frame_ms: f64,
    // mouse for fluid interaction
    mouse_ndc: [f32; 2],
    mouse_prev_ndc: [f32; 2],
    mouse_pressed: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None, renderer: None, scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 1000.0, 1.0),
            orbit: OrbitCamera::new(glam::Vec3::new(0.0, 3.0, 0.0), 75.0),
            sim: None, density_field: None, surface_renderer: None,
            particle_renderer: None, blit: None,
            color_view: None, depth_view: None, output_view: None, color_texture: None,
            surface_bg: None, blit_bg: None,
            egui_ctx: egui::Context::default(),
            egui_state: None, egui_renderer: None,
            show_particles: true, particle_size: 0.15,
            last_time: None,
            frame_count: 0, frame_time_accum: 0.0, current_fps: 0.0, current_frame_ms: 0.0,
            mouse_ndc: [0.0; 2], mouse_prev_ndc: [0.0; 2], mouse_pressed: false,
        }
    }

    fn rebuild_offscreen(&mut self, w: u32, h: u32) {
        let device = self.renderer.as_ref().unwrap().device();
        let mk = |label: &str, fmt: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
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
            self.surface_bg = Some(sr.create_bind_group(device, self.color_view.as_ref().unwrap(), self.depth_view.as_ref().unwrap(), self.output_view.as_ref().unwrap(), &df.density_view));
            self.blit_bg = Some(blit.bind(device, self.output_view.as_ref().unwrap()));
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let window = Arc::new(el.create_window(Window::default_attributes()
            .with_title("Kansei — Fluid Engine").with_inner_size(winit::dpi::LogicalSize::new(1280, 720))).unwrap());
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() })).unwrap();
        let mut renderer = Renderer::new(RendererConfig { width: size.width, height: size.height, sample_count: 1, clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0), present_mode: wgpu::PresentMode::Immediate, ..Default::default() });
        pollster::block_on(renderer.initialize(surface, &adapter));
        let device = renderer.device();
        let format = renderer.presentation_format();

        // Particles
        let count = 50000usize;
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

        let mut sim = FluidSimulation::new(FluidSimulationOptions {
            max_particles: count as u32, dimensions: 3, smoothing_radius: 1.0, pressure_multiplier: 46.5,
            near_pressure_multiplier: 20.0, density_target: 8.6, viscosity: 0.36, damping: 0.998,
            gravity: [0.0, -9.8, 0.0], mouse_force: 1520.0, substeps: 3, world_bounds_padding: 0.3,
            ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
        });
        sim.initialize(&positions, device);
        sim.world_bounds_min = [-12.0, -8.0, -8.0];
        sim.world_bounds_max = [12.0, 32.0, 8.0];
        sim.rebuild_grid(device);

        self.density_field = Some(FluidDensityField::new(device, sim.positions_buffer().unwrap(), sim.world_bounds_min, sim.world_bounds_max, DensityFieldOptions { resolution: 64, kernel_scale: 3.7 }));
        self.surface_renderer = Some(FluidSurfaceRenderer::new(device));
        self.particle_renderer = Some(ParticleRenderer::new(device, sim.positions_buffer().unwrap(), count as u32, format));
        self.blit = Some(BlitPipeline::new(device, format));
        self.sim = Some(sim);
        self.renderer = Some(renderer);

        // egui
        self.egui_state = Some(egui_winit::State::new(self.egui_ctx.clone(), egui::ViewportId::ROOT, &*window, Some(window.scale_factor() as f32), None, None));
        self.egui_renderer = Some(egui_wgpu::Renderer::new(self.renderer.as_ref().unwrap().device(), format, None, 1, false));

        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();
        self.window = Some(window);
        self.last_time = Some(Instant::now());
        self.rebuild_offscreen(size.width, size.height);
        log::info!("Fluid Engine — {} particles", count);
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
                self.orbit.dragging = s == ElementState::Pressed;
                self.mouse_pressed = s == ElementState::Pressed;
            }
            WindowEvent::MouseInput { button: MouseButton::Right, state: s, .. } => {
                self.mouse_pressed = s == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.orbit.on_mouse_move(position.x, position.y);
                // Track mouse in NDC for fluid interaction
                if let Some(ref w) = self.window {
                    let s = w.inner_size();
                    self.mouse_prev_ndc = self.mouse_ndc;
                    self.mouse_ndc = [
                        (position.x as f32 / s.width as f32) * 2.0 - 1.0,
                        (position.y as f32 / s.height as f32) * 2.0 - 1.0,
                    ];
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.orbit.on_scroll(dy);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self.last_time.map(|t| now.duration_since(t).as_secs_f32()).unwrap_or(1.0/60.0).min(1.0/30.0);
                self.last_time = Some(now);
                let size = self.window.as_ref().unwrap().inner_size();

                if self.color_texture.is_none() { self.rebuild_offscreen(size.width, size.height); }

                let renderer = self.renderer.as_ref().unwrap();
                let device = renderer.device();
                let queue = renderer.queue();

                // Camera — use engine Camera with orbit eye position
                let eye = self.orbit.eye();
                self.camera.set_position(eye.x, eye.y, eye.z);
                self.camera.look_at(&Vec3::new(self.orbit.target.x, self.orbit.target.y, self.orbit.target.z));
                let aspect = size.width as f32 / size.height as f32;
                self.camera.aspect = aspect;
                self.camera.update_projection_matrix();
                let view = self.camera.view_matrix.to_glam();
                let proj = self.camera.projection_matrix.to_glam();
                let inv_view = view.inverse();
                let inv_vp = (proj * view).inverse();

                // Mouse interaction
                let mouse_dir = [
                    -(self.mouse_ndc[0] - self.mouse_prev_ndc[0]),
                    -(self.mouse_ndc[1] - self.mouse_prev_ndc[1]),
                ];
                let mouse_strength = (mouse_dir[0] * mouse_dir[0] + mouse_dir[1] * mouse_dir[1]).sqrt().min(1.0);

                // Upload camera matrices + run sim
                if let Some(ref mut sim) = self.sim {
                    let identity = glam::Mat4::IDENTITY.to_cols_array();
                    sim.set_camera_matrices(queue,
                        &view.to_cols_array(), &proj.to_cols_array(),
                        &inv_view.to_cols_array(), &identity);
                    sim.update(device, queue, dt, mouse_strength, self.mouse_ndc, mouse_dir);
                }

                let surface = renderer.surface().unwrap();
                let output = surface.get_current_texture().expect("Surface texture");
                let canvas_view = output.texture.create_view(&Default::default());
                let mut encoder = device.create_command_encoder(&Default::default());

                if self.show_particles {
                    if let Some(ref pr) = self.particle_renderer {
                        pr.upload(queue, &view, &proj, self.particle_size);
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
                        df.update(&mut encoder, queue, sim.world_bounds_min, sim.world_bounds_max, sim.particle_count(), sim.params.smoothing_radius);
                    }
                    { let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: self.color_view.as_ref().unwrap(), resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }), store: wgpu::StoreOp::Store } })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: self.depth_view.as_ref().unwrap(), depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                        ..Default::default()
                    }); }
                    if let Some(ref sr) = self.surface_renderer {
                        sr.render(&mut encoder, queue, self.surface_bg.as_ref().unwrap(), &inv_vp, [eye.x, eye.y, eye.z], sim.world_bounds_min, sim.world_bounds_max, size.width, size.height);
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
                        ui.heading("Fluid Engine");
                        ui.label(format!("{:.1} FPS  ({:.2} ms)", self.current_fps, self.current_frame_ms));
                        ui.label(format!("Particles: {}", 50000));
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
                        }
                        if let Some(ref mut sr) = self.surface_renderer {
                            ui.separator();
                            ui.label("Surface");
                            ui.add(egui::Slider::new(&mut sr.density_scale, 0.1..=10.0).text("Density scale"));
                            ui.add(egui::Slider::new(&mut sr.density_threshold, 0.01..=5.0).text("Threshold"));
                            ui.add(egui::Slider::new(&mut sr.absorption, 0.0..=10.0).text("Absorption"));
                        }
                        ui.separator();
                        ui.label(format!("Camera: dist={:.1} az={:.2} el={:.2}", self.orbit.distance, self.orbit.azimuth, self.orbit.elevation));
                    });
                });

                // Submit scene encoder first
                queue.submit(std::iter::once(encoder.finish()));

                // egui in a separate encoder
                let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                let screen_descriptor = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.width, size.height], pixels_per_point: full_output.pixels_per_point };

                // Take egui renderer out of self to avoid borrow conflicts
                let mut er = self.egui_renderer.take().unwrap();
                for (id, delta) in &full_output.textures_delta.set {
                    er.update_texture(device, queue, *id, delta);
                }
                // Egui: upload buffers in one encoder, render in another
                let mut upload_enc = device.create_command_encoder(&Default::default());
                er.update_buffers(device, queue, &mut upload_enc, &paint_jobs, &screen_descriptor);
                queue.submit([upload_enc.finish()]);

                let mut enc = device.create_command_encoder(&Default::default());
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &canvas_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                }).forget_lifetime();
                er.render(&mut pass, &paint_jobs, &screen_descriptor);
                drop(pass);
                queue.submit([enc.finish()]);

                for id in &full_output.textures_delta.free { er.free_texture(id); }
                self.egui_renderer = Some(er);
                self.egui_state.as_mut().unwrap().handle_platform_output(self.window.as_ref().unwrap(), full_output.platform_output);
                // FPS counter — measure work time before present (excludes vsync wait)
                let work_time = now.elapsed().as_secs_f64();

                output.present();
                self.frame_time_accum += work_time;
                self.frame_count += 1;
                if self.frame_count % 60 == 0 {
                    let avg = self.frame_time_accum / 60.0;
                    self.current_frame_ms = avg * 1000.0;
                    self.current_fps = 1.0 / avg;
                    self.frame_time_accum = 0.0;
                }

                if let Some(ref w) = self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Kansei — Fluid Engine");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
