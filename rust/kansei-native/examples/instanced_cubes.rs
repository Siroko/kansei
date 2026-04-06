use std::sync::Arc;
use std::time::Instant;

use kansei_core::buffers::InstanceBuffer;
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Binding, Material, MaterialOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const INSTANCED_WGSL: &str = include_str!("../../kansei-core/src/shaders/basic_instanced.wgsl");

const GRID_SIZE: usize = 10;
const INSTANCE_COUNT: usize = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const SPACING: f32 = 3.0;

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
    fn new(target: glam::Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            azimuth: 0.4,
            elevation: 0.3,
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
        self.distance = (self.distance - delta * 2.0).clamp(10.0, 150.0);
    }
}

// ── Instance matrix generation ──

fn build_instance_matrices(time: f32) -> Vec<f32> {
    let mut data = vec![0.0f32; INSTANCE_COUNT * 16];
    let offset = (GRID_SIZE as f32 - 1.0) * SPACING * 0.5;
    let mut idx = 0;
    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let px = x as f32 * SPACING - offset;
                let py = y as f32 * SPACING - offset;
                let pz = z as f32 * SPACING - offset;
                let angle = time * 0.5 + (x + y + z) as f32 * 0.3;
                let m = glam::Mat4::from_translation(glam::Vec3::new(px, py, pz))
                    * glam::Mat4::from_rotation_y(angle);
                data[idx..idx + 16].copy_from_slice(&m.to_cols_array());
                idx += 16;
            }
        }
    }
    data
}

// ── App ──

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    orbit: OrbitCamera,
    start_time: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 500.0, 1280.0 / 720.0),
            orbit: OrbitCamera::new(glam::Vec3::ZERO, 50.0),
            start_time: Instant::now(),
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
                    .with_title("Kansei \u{2014} Instanced Cubes (1000)")
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

        // Renderer
        let mut renderer = Renderer::new(RendererConfig {
            width: size.width,
            height: size.height,
            sample_count: 4,
            clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0),
            ..Default::default()
        });
        pollster::block_on(renderer.initialize(surface, &adapter));

        // Geometry — 1x1x1 box
        let geometry = BoxGeometry::new(1.0, 1.0, 1.0);

        // Material with color uniform — blue-ish
        let color_data: [f32; 4] = [0.4, 0.7, 0.9, 1.0];
        let mut material = Material::new(
            "InstancedMaterial",
            INSTANCED_WGSL,
            vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
            MaterialOptions::default(),
        );
        material.set_uniform_bindable(0, "ColorUniform", &color_data);

        // Instance buffer with mat4 per instance (locations 3-6)
        let matrices = build_instance_matrices(0.0);
        let instance_buf = InstanceBuffer::from_mat4("InstanceMatrices", &matrices, 3);

        // Instanced renderable
        let renderable = Renderable::new_instanced(
            geometry,
            material,
            INSTANCE_COUNT as u32,
            vec![instance_buf],
        );
        self.scene.add(SceneNode::Renderable(renderable));

        // Camera
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();

        let eye = self.orbit.eye();
        self.camera.set_position(eye.x, eye.y, eye.z);
        self.camera.look_at(&Vec3::ZERO);

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start_time = Instant::now();

        log::info!(
            "Kansei \u{2014} Instanced Cubes: {} instances in a {}x{}x{} grid",
            INSTANCE_COUNT,
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE
        );
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

                // Update instance matrices
                if let Some(r) = self.scene.get_renderable_mut(0) {
                    let matrices = build_instance_matrices(t);
                    if let Some(ib) = r.instance_buffers.get(0) {
                        if let Some(ref renderer) = self.renderer {
                            ib.update(
                                renderer.queue(),
                                bytemuck::cast_slice(&matrices),
                            );
                        }
                    }
                }

                // Update camera from orbit state
                let eye = self.orbit.eye();
                self.camera.set_position(eye.x, eye.y, eye.z);
                self.camera.look_at(&Vec3::ZERO);

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
    log::info!("Kansei \u{2014} Instanced Cubes");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
