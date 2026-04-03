use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::math::Vec3;
use kansei_core::simulations::fluid::{
    FluidSimulation, FluidSimulationOptions, FluidDensityField, DensityFieldOptions, FluidSurfaceRenderer,
};

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("Kansei WASM initialized");
}

// ── Blit shader (fullscreen triangle) ──
const BLIT_WGSL: &str = r#"
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
"#;

// ── Particle billboard shader ──
const PARTICLE_WGSL: &str = r#"
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
"#;

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU, ..Default::default()
    });
    let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface), ..Default::default()
    }).await.ok_or("No adapter")?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Kansei"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
        memory_hints: wgpu::MemoryHints::default(),
    }, None).await.map_err(|e| JsValue::from_str(&format!("{e}")))?;

    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats[0];
    surface.configure(&device, &wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
        width, height, present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2,
    });

    // ── Particles ──
    let count = 50000usize;
    let radius = 10.0f32;
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
            if x*x + y*y + z*z <= radius * radius {
                positions[i*4]=x; positions[i*4+1]=y; positions[i*4+2]=z; positions[i*4+3]=1.0; break;
            }
        }
    }

    // ── Sim ──
    let mut sim = FluidSimulation::new(FluidSimulationOptions {
        max_particles: count as u32, dimensions: 3, smoothing_radius: 1.0,
        pressure_multiplier: 46.5, near_pressure_multiplier: 20.0, density_target: 8.6,
        viscosity: 0.36, damping: 0.998, gravity: [0.0, -9.8, 0.0],
        mouse_force: 1520.0, substeps: 3, world_bounds_padding: 0.3,
        ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
    });
    sim.initialize(&positions, &device);
    sim.world_bounds_min = [-25.0, -8.0, -16.0];
    sim.world_bounds_max = [25.0, 30.0, 16.0];
    sim.rebuild_grid(&device);

    // ── Density field + surface renderer ──
    let density_field = FluidDensityField::new(&device, sim.positions_buffer().unwrap(),
        sim.world_bounds_min, sim.world_bounds_max, DensityFieldOptions { resolution: 128, kernel_scale: 3.7 });
    let surface_renderer = FluidSurfaceRenderer::new(&device);

    // ── Particle pipeline ──
    let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Particles"), source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
    });
    let particle_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: 144, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Particles"), layout: None,
        vertex: wgpu::VertexState { module: &particle_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &particle_shader, entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default() }),
        primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
    });
    let particle_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &particle_pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: sim.positions_buffer().unwrap().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: particle_params_buf.as_entire_binding() },
        ],
    });

    // ── Blit pipeline ──
    let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Blit"), source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
    });
    let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ],
    });
    let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blit"), layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&blit_bgl], push_constant_ranges: &[] })),
        vertex: wgpu::VertexState { module: &blit_shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &blit_shader, entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default() }),
        primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
    });
    let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });

    // ── Offscreen textures ──
    let mk_tex = |label: &str, fmt: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label), size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: fmt, usage, view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    };
    let cu = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
    let (_color_tex, color_view) = mk_tex("Color", wgpu::TextureFormat::Rgba16Float, cu);
    let (_depth_tex, depth_view) = mk_tex("Depth", wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
    let (_output_tex, output_view) = mk_tex("Output", wgpu::TextureFormat::Rgba16Float, cu);

    // ── Bind groups for surface renderer + blit ──
    let surface_bg = surface_renderer.create_bind_group(&device, &color_view, &depth_view, &output_view, &density_field.density_view);
    let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &blit_bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&output_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&blit_sampler) },
        ],
    });

    let perf_now = window.performance().map(|p| p.now()).unwrap_or(0.0);
    let state = Rc::new(RefCell::new(State {
        device, queue, surface, format,
        sim, density_field, surface_renderer,
        particle_pipeline, particle_bg, particle_params_buf,
        blit_pipeline, blit_bg, surface_bg,
        color_view, depth_view, output_view,
        count: count as u32, width, height,
        mouse_ndc: [0.0; 2], mouse_prev_ndc: [0.0; 2],
        azimuth: 0.0, elevation: 0.3, distance: 75.0,
        dragging: false, last_mouse: None,
        particle_size: 0.15, show_particles: true,
        camera: Camera::new(45.0, 0.1, 1000.0, width as f32 / height as f32),
        frame_count: 0, frame_time_sum: 0.0, last_perf_time: perf_now,
        current_fps: 0.0, current_frame_ms: 0.0,
    }));

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    // Mouse events
    { let s = state.clone();
      let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
        let mut st = s.borrow_mut();
        let nx = (e.offset_x() as f32 / st.width as f32) * 2.0 - 1.0;
        let ny = (e.offset_y() as f32 / st.height as f32) * 2.0 - 1.0;
        st.mouse_prev_ndc = st.mouse_ndc; st.mouse_ndc = [nx, ny];
        if st.dragging {
            if let Some((lx, ly)) = st.last_mouse {
                st.azimuth -= (e.offset_x() as f32 - lx) * 0.005;
                st.elevation = (st.elevation + (e.offset_y() as f32 - ly) * 0.005).clamp(-1.5, 1.5);
            }
        }
        st.last_mouse = Some((e.offset_x() as f32, e.offset_y() as f32));
      });
      canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())?; cb.forget(); }
    { let s = state.clone();
      let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| { s.borrow_mut().dragging = e.buttons() & 1 != 0; });
      canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())?; cb.forget(); }
    { let s = state.clone();
      let cb = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |_: web_sys::MouseEvent| { s.borrow_mut().dragging = false; });
      canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())?; cb.forget(); }
    { let s = state.clone();
      let cb = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
        let mut st = s.borrow_mut();
        st.distance = (st.distance + e.delta_y() as f32 * 0.01).clamp(2.0, 100.0);
      });
      canvas.add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())?; cb.forget(); }

    // Animation loop
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone(); let s = state.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        s.borrow_mut().render_frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    log::info!("Kansei WASM — {} particles, [toggle via tweakpane]", count);
    Ok(())
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

struct State {
    device: wgpu::Device, queue: wgpu::Queue, surface: wgpu::Surface<'static>, format: wgpu::TextureFormat,
    sim: FluidSimulation, density_field: FluidDensityField, surface_renderer: FluidSurfaceRenderer,
    particle_pipeline: wgpu::RenderPipeline, particle_bg: wgpu::BindGroup, particle_params_buf: wgpu::Buffer,
    blit_pipeline: wgpu::RenderPipeline, blit_bg: wgpu::BindGroup, surface_bg: wgpu::BindGroup,
    color_view: wgpu::TextureView, depth_view: wgpu::TextureView, output_view: wgpu::TextureView,
    count: u32, width: u32, height: u32,
    mouse_ndc: [f32; 2], mouse_prev_ndc: [f32; 2],
    azimuth: f32, elevation: f32, distance: f32,
    dragging: bool, last_mouse: Option<(f32, f32)>,
    particle_size: f32, show_particles: bool,
    camera: Camera,
    frame_count: u32, frame_time_sum: f64, last_perf_time: f64,
    current_fps: f64, current_frame_ms: f64,
}

impl State {
    fn render_frame(&mut self) {
        // Wall clock frame timing (honest FPS)
        let perf = web_sys::window().unwrap().performance().unwrap();
        let now = perf.now();
        let frame_ms = now - self.last_perf_time;
        self.last_perf_time = now;
        self.frame_time_sum += frame_ms;
        self.frame_count += 1;
        if self.frame_count % 60 == 0 {
            let avg = self.frame_time_sum / 60.0;
            self.current_frame_ms = avg;
            self.current_fps = 1000.0 / avg;
            self.frame_time_sum = 0.0;
        }

        let dt = 1.0 / 60.0;
        let target = glam::Vec3::new(0.0, 3.0, 0.0);
        let eye_offset = glam::Vec3::new(
            self.distance * self.azimuth.sin() * self.elevation.cos(),
            self.distance * self.elevation.sin(),
            self.distance * self.azimuth.cos() * self.elevation.cos(),
        );
        let eye = target + eye_offset;
        self.camera.set_position(eye.x, eye.y, eye.z);
        self.camera.look_at(&Vec3::new(target.x, target.y, target.z));
        self.camera.aspect = self.width as f32 / self.height as f32;
        self.camera.update_projection_matrix();
        let view = self.camera.view_matrix.to_glam();
        let proj = self.camera.projection_matrix.to_glam();
        let inv_view = self.camera.inverse_view_matrix.to_glam();
        let inv_vp = (proj * view).inverse();

        let mouse_dir = [
            -(self.mouse_ndc[0] - self.mouse_prev_ndc[0]),
            -(self.mouse_ndc[1] - self.mouse_prev_ndc[1]),
        ];
        let mouse_strength = (mouse_dir[0]*mouse_dir[0] + mouse_dir[1]*mouse_dir[1]).sqrt().min(1.0);

        let identity = glam::Mat4::IDENTITY.to_cols_array();
        self.sim.set_camera_matrices(&self.queue, &view.to_cols_array(), &proj.to_cols_array(), &inv_view.to_cols_array(), &identity);
        self.sim.update(&self.device, &self.queue, dt, mouse_strength, self.mouse_ndc, mouse_dir);

        let output = self.surface.get_current_texture().unwrap();
        let canvas_view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        if self.show_particles {
            // Upload particle params
            let mut data = [0.0f32; 36];
            data[..16].copy_from_slice(&view.to_cols_array());
            data[16..32].copy_from_slice(&proj.to_cols_array());
            data[32] = self.particle_size;
            self.queue.write_buffer(&self.particle_params_buf, 0, bytemuck::cast_slice(&data));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &canvas_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }), store: wgpu::StoreOp::Store },
                })], ..Default::default()
            }).forget_lifetime();
            pass.set_pipeline(&self.particle_pipeline);
            pass.set_bind_group(0, &self.particle_bg, &[]);
            pass.draw(0..self.count * 6, 0..1);
            drop(pass);
        } else {
            // Density field update
            self.density_field.update(&mut encoder, &self.queue,
                self.sim.world_bounds_min, self.sim.world_bounds_max,
                self.sim.particle_count(), self.sim.params.smoothing_radius);

            // Clear offscreen color + depth
            {
                let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.color_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }), ..Default::default()
                });
            }

            // Ray-march
            self.surface_renderer.render(&mut encoder, &self.queue, &self.surface_bg,
                &inv_vp, [eye.x, eye.y, eye.z],
                self.sim.world_bounds_min, self.sim.world_bounds_max,
                self.width, self.height);

            // Blit to canvas
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &canvas_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })], ..Default::default()
            }).forget_lifetime();
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.blit_bg, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.mouse_prev_ndc = self.mouse_ndc;
    }
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) { GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } }); }

#[wasm_bindgen]
pub fn get_fps() -> f64 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() { rc.borrow().current_fps } else { 0.0 }
    })
}

#[wasm_bindgen]
pub fn get_frame_time() -> f64 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() { rc.borrow().current_frame_ms } else { 0.0 }
    })
}

#[wasm_bindgen] pub fn set_pressure(v: f32) { with_state(|s| s.sim.params.pressure_multiplier = v); }
#[wasm_bindgen] pub fn set_near_pressure(v: f32) { with_state(|s| s.sim.params.near_pressure_multiplier = v); }
#[wasm_bindgen] pub fn set_density_target(v: f32) { with_state(|s| s.sim.params.density_target = v); }
#[wasm_bindgen] pub fn set_viscosity(v: f32) { with_state(|s| s.sim.params.viscosity = v); }
#[wasm_bindgen] pub fn set_damping(v: f32) { with_state(|s| s.sim.params.damping = v); }
#[wasm_bindgen] pub fn set_gravity_y(v: f32) { with_state(|s| s.sim.params.gravity[1] = v); }
#[wasm_bindgen] pub fn set_mouse_force(v: f32) { with_state(|s| s.sim.params.mouse_force = v); }
#[wasm_bindgen] pub fn set_mouse_radius(v: f32) { with_state(|s| s.sim.params.mouse_radius = v); }
#[wasm_bindgen] pub fn set_particle_size(v: f32) { with_state(|s| s.particle_size = v); }
#[wasm_bindgen] pub fn set_substeps(v: u32) { with_state(|s| s.sim.params.substeps = v); }
#[wasm_bindgen] pub fn set_show_particles(v: bool) { with_state(|s| s.show_particles = v); }
#[wasm_bindgen] pub fn set_density_scale(v: f32) { with_state(|s| s.surface_renderer.density_scale = v); }
#[wasm_bindgen] pub fn set_density_threshold(v: f32) { with_state(|s| s.surface_renderer.density_threshold = v); }
#[wasm_bindgen] pub fn set_absorption(v: f32) { with_state(|s| s.surface_renderer.absorption = v); }
#[wasm_bindgen] pub fn set_step_count(v: u32) { with_state(|s| s.surface_renderer.step_count = v); }
#[wasm_bindgen] pub fn set_kernel_scale(v: f32) { with_state(|s| s.density_field.kernel_scale = v); }
#[wasm_bindgen] pub fn set_density_resolution(v: u32) {
    with_state(|s| {
        s.density_field = FluidDensityField::new(&s.device, s.sim.positions_buffer().unwrap(),
            s.sim.world_bounds_min, s.sim.world_bounds_max,
            DensityFieldOptions { resolution: v, kernel_scale: s.density_field.kernel_scale });
        s.surface_bg = s.surface_renderer.create_bind_group(&s.device,
            &s.color_view, &s.depth_view, &s.output_view, &s.density_field.density_view);
    });
}
#[wasm_bindgen] pub fn set_bounds(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) {
    with_state(|s| {
        s.sim.world_bounds_min = [min_x, min_y, min_z];
        s.sim.world_bounds_max = [max_x, max_y, max_z];
        s.sim.rebuild_grid(&s.device);
        // Rebuild density field for new bounds
        s.density_field = FluidDensityField::new(&s.device, s.sim.positions_buffer().unwrap(),
            s.sim.world_bounds_min, s.sim.world_bounds_max,
            DensityFieldOptions { resolution: 128, kernel_scale: s.density_field.kernel_scale });
        s.surface_bg = s.surface_renderer.create_bind_group(&s.device,
            &s.color_view, &s.depth_view, &s.output_view, &s.density_field.density_view);
    });
}
