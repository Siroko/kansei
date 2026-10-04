//! Instancing (native twin of the WASM example): a 10 x 10 x 10 grid of cubes drawn in one call,
//! each cube's model matrix an instance attribute of one `InstancedGeometry`, drawn with the
//! stock `Material::basic_instanced`; the CPU rewrites the matrices each frame to turn the cubes.
//! Drag to orbit, wheel to zoom.
//!
//!   cargo run -p kansei-native --example instancing

use std::sync::Arc;
use std::time::Instant;

use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry};
use kansei_core::materials::Material;
use kansei_core::math::{Mat4, Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const N: usize = 10;
const SPACING: f32 = 3.0;

/// The cubes' model matrices (column-major, 16 floats each) at `time` seconds.
fn instance_matrices(time: f32) -> Vec<f32> {
    let offset = (N as f32 - 1.0) * SPACING * 0.5;
    let mut data = Vec::with_capacity(N * N * N * 16);
    for x in 0..N {
        for y in 0..N {
            for z in 0..N {
                let position = Vec3::new(x as f32 * SPACING - offset, y as f32 * SPACING - offset, z as f32 * SPACING - offset);
                let angle = time * 0.5 + (x + y + z) as f32 * 0.3;
                data.extend_from_slice(&(Mat4::from_translation(position) * Mat4::from_rotation_y(angle)).to_cols_array());
            }
        }
    }
    data
}

struct Running {
    window: Arc<Window>,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    cubes: usize,
}

#[derive(Default)]
struct App {
    running: Option<Running>,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
    start: Option<Instant>,
}

fn build(window: Arc<Window>) -> Running {
    let size = window.inner_size();
    let mut renderer = Renderer::new(RendererConfig { width: size.width, height: size.height, clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0), ..Default::default() });
    pollster::block_on(renderer.initialize_with_target(window.clone()));

    // one vertex buffer of matrices, read per instance at locations 3-6
    let matrices = ComputeBuffer::from_slice("Instances", BufferType::Storage, BufferUsage::VERTEX, &instance_matrices(0.0)).with_vertex_mat4(3);
    let geometry = InstancedGeometry::new(BoxGeometry::new(1.0, 1.0, 1.0), (N * N * N) as u32, vec![matrices]);
    let mut scene = Scene::new();
    let cubes = scene.add(SceneNode::Renderable(Renderable::new(geometry, Material::basic_instanced("Cubes", [0.4, 0.7, 0.9, 1.0]))));

    let camera = Camera::new(45.0, 0.1, 500.0, size.width as f32 / size.height.max(1) as f32);
    let mut controls = CameraControls::new(Vec3::ZERO, N as f32 * SPACING * 1.6);
    controls.set_elevation(0.35);
    Running { window, renderer, scene, camera, controls, cubes }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.running.is_none() {
            let attributes = Window::default_attributes().with_title("Kansei \u{2014} Instancing").with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            self.running = Some(build(Arc::new(el.create_window(attributes).unwrap())));
            self.start = Some(Instant::now());
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(app) = &mut self.running else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                app.renderer.resize(s.width.max(1), s.height.max(1));
                app.camera.aspect = s.width.max(1) as f32 / s.height.max(1) as f32;
                app.camera.update_projection_matrix();
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => self.dragging = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => {
                if let (true, Some((x, y))) = (self.dragging, self.last_mouse) {
                    app.controls.rotate(-(position.x - x) as f32 * 0.005, (position.y - y) as f32 * 0.005);
                }
                self.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => app.controls.zoom(match delta {
                MouseScrollDelta::LineDelta(_, y) => y * 2.0,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
            }),
            WindowEvent::RedrawRequested => {
                let t = self.start.map_or(0.0, |s| s.elapsed().as_secs_f32());
                let matrices = instance_matrices(t);
                if let Some(buffer) = app.scene.get_renderable_mut(app.cubes).and_then(|r| r.geometry.instance_buffers.first_mut()) {
                    buffer.write(&matrices);
                }
                app.controls.update(&mut app.camera, 0.0);
                app.renderer.render(&mut app.scene, &mut app.camera);
                app.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::default()).unwrap();
}
