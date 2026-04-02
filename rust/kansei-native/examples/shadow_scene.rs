use std::sync::Arc;
use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::materials::{Binding, BindingResource, Material, MaterialOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene};
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const LIT_WGSL: &str = include_str!("../../kansei-core/src/shaders/basic_lit.wgsl");

// ── Orbit camera ──

struct OrbitCamera {
    target: glam::Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

impl OrbitCamera {
    fn new(target: glam::Vec3, distance: f32, elevation: f32) -> Self {
        Self {
            target,
            distance,
            azimuth: 0.0,
            elevation,
            dragging: false,
            last_mouse: None,
        }
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
        self.distance = (self.distance - delta * 2.0).clamp(4.0, 50.0);
    }
}

// ── Helper: create a lit material with color + specular ──

fn create_lit_material(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shared: &kansei_core::renderers::SharedLayouts,
    label: &str,
    color: [f32; 4],
    specular: [f32; 4],
) -> (Material, wgpu::Buffer) {
    let mat_data: [f32; 8] = [
        color[0], color[1], color[2], color[3],
        specular[0], specular[1], specular[2], specular[3],
    ];
    let mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&mat_buf, 0, bytemuck::cast_slice(&mat_data));

    let mut material = Material::new(
        label,
        LIT_WGSL,
        vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
        MaterialOptions::default(),
    );
    material.create_bind_group(device, shared, &[(
        0,
        BindingResource::Buffer {
            buffer: &mat_buf,
            offset: 0,
            size: None,
        },
    )]);

    (material, mat_buf)
}

// ── App ──

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    orbit: OrbitCamera,
    start_time: Instant,
    // Keep material buffers alive
    _mat_bufs: Vec<wgpu::Buffer>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 100.0, 1280.0 / 720.0),
            orbit: OrbitCamera::new(glam::Vec3::new(0.0, 1.0, 0.0), 15.0, 0.5),
            start_time: Instant::now(),
            _mat_bufs: Vec::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Kansei \u{2014} Shadow Scene")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            )
            .unwrap(),
        );
        let size = window.inner_size();

        // wgpu instance, surface, adapter
        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            }))
            .unwrap();

        // Renderer with dark background
        let mut renderer = Renderer::new(RendererConfig {
            width: size.width,
            height: size.height,
            sample_count: 4,
            clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0),
            ..Default::default()
        });
        pollster::block_on(renderer.initialize(surface, &adapter));

        // Enable shadow mapping (2048x2048 shadow map)
        renderer.enable_shadows(2048);

        let device = renderer.device();
        let queue = renderer.queue();
        let shared = renderer.shared_layouts();

        // ── Floor ──
        let floor_geo = PlaneGeometry::new(30.0, 30.0);
        let (floor_mat, floor_buf) = create_lit_material(
            device, queue, shared,
            "FloorMat",
            [0.6, 0.6, 0.6, 1.0],        // light grey
            [0.15, 0.15, 0.15, 0.05],     // low specular, very low shininess
        );
        let mut floor = Renderable::new(floor_geo, floor_mat);
        floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
        floor.object.update_model_matrix();
        floor.cast_shadow = false; // floor should not cast shadows
        self._mat_bufs.push(floor_buf);

        // ── Box ──
        let box_geo = BoxGeometry::new(2.0, 2.0, 2.0);
        let (box_mat, box_buf) = create_lit_material(
            device, queue, shared,
            "BoxMat",
            [0.85, 0.25, 0.2, 1.0],       // reddish
            [0.5, 0.5, 0.5, 0.3],         // medium specular
        );
        let mut box_obj = Renderable::new(box_geo, box_mat);
        box_obj.object.set_position(0.0, 1.0, 0.0);
        box_obj.object.update_model_matrix();
        self._mat_bufs.push(box_buf);

        // ── Sphere ──
        let sphere_geo = SphereGeometry::new(1.2, 32, 16);
        let (sphere_mat, sphere_buf) = create_lit_material(
            device, queue, shared,
            "SphereMat",
            [0.2, 0.4, 0.9, 1.0],         // blue-ish
            [0.8, 0.8, 0.8, 0.6],         // high specular
        );
        let mut sphere = Renderable::new(sphere_geo, sphere_mat);
        sphere.object.set_position(3.0, 1.2, -1.0);
        sphere.object.update_model_matrix();
        self._mat_bufs.push(sphere_buf);

        // Add objects to scene
        self.scene.add(floor);
        self.scene.add(box_obj);
        self.scene.add(sphere);

        // ── Lights ──
        // Directional: warm white from upper-left
        let mut dir_light = DirectionalLight::new(
            Vec3::new(-0.3, -1.0, -0.5),
            Vec3::new(1.0, 0.95, 0.8),
            1.0,
        );
        dir_light.cast_shadow = true;
        self.scene.add_light(Light::Directional(dir_light));

        // Point: soft warm above-left
        self.scene.add_light(Light::Point(PointLight::new(
            Vec3::new(-3.0, 4.0, 2.0),
            Vec3::new(1.0, 0.8, 0.5),
            2.0,
            20.0,
        )));

        // Camera
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();

        let eye = self.orbit.eye();
        self.camera.set_position(eye.x, eye.y, eye.z);
        self.camera.look_at(&Vec3::new(0.0, 1.0, 0.0));

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start_time = Instant::now();

        log::info!("Kansei \u{2014} Shadow Scene ready");
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::Resized(s) => {
                if let Some(ref mut r) = self.renderer {
                    r.resize(s.width, s.height);
                    self.camera.aspect = s.width as f32 / s.height as f32;
                    self.camera.update_projection_matrix();
                }
            }

            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: s,
                ..
            } => {
                self.orbit.dragging = s == ElementState::Pressed;
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.orbit.on_mouse_move(position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.orbit.on_scroll(dy);
            }

            WindowEvent::RedrawRequested => {
                let t = self.start_time.elapsed().as_secs_f32();

                // Rotate the box slowly on Y
                if let Some(r) = self.scene.get_mut(1) {
                    r.object.rotation.y = t * 0.3;
                    r.object.update_model_matrix();
                }

                // Update normal matrices for all objects
                for i in 0..self.scene.len() {
                    if let Some(r) = self.scene.get_mut(i) {
                        r.object.update_normal_matrix(&self.camera.view_matrix);
                    }
                }

                // Update camera from orbit state
                let eye = self.orbit.eye();
                self.camera.set_position(eye.x, eye.y, eye.z);
                self.camera.look_at(&Vec3::new(0.0, 1.0, 0.0));

                // Render
                if let Some(ref mut renderer) = self.renderer {
                    renderer.render(&mut self.scene, &mut self.camera);
                }

                if let Some(ref w) = self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Kansei \u{2014} Shadow Scene");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
