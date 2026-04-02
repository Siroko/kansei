use std::sync::Arc;
use std::time::Instant;

use kansei_core::math::{Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Binding, BindingResource, Material, MaterialOptions};
use kansei_core::objects::{Renderable, Scene};
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const BASIC_WGSL: &str = include_str!("../../kansei-core/src/shaders/basic.wgsl");

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    start_time: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 100.0, 1280.0 / 720.0),
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
                    .with_title("Kansei \u{2014} Spinning Box")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            )
            .unwrap(),
        );
        let size = window.inner_size();

        // wgpu instance, surface, adapter
        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
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

        let device = renderer.device();
        let queue = renderer.queue();
        let shared = renderer.shared_layouts();

        // Geometry
        let geometry = BoxGeometry::new(2.0, 2.0, 2.0);

        // Material
        let mut material = Material::new(
            "BasicMaterial",
            BASIC_WGSL,
            vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
            MaterialOptions::default(),
        );

        // Color uniform buffer — reddish color
        let color_data: [f32; 4] = [0.9, 0.3, 0.2, 1.0];
        let color_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColorUniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&color_buf, 0, bytemuck::cast_slice(&color_data));

        // Create bind group
        material.create_bind_group(device, shared, &[(
            0,
            BindingResource::Buffer {
                buffer: &color_buf,
                offset: 0,
                size: None,
            },
        )]);

        // Renderable + Scene
        let renderable = Renderable::new(geometry, material);
        self.scene.add(renderable);

        // Camera
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.set_position(0.0, 2.0, 6.0);
        self.camera.look_at(&Vec3::ZERO);
        self.camera.update_projection_matrix();

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start_time = Instant::now();

        log::info!("Kansei — Spinning Box ready");
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
            WindowEvent::RedrawRequested => {
                let t = self.start_time.elapsed().as_secs_f32();

                // Rotate the box
                if let Some(r) = self.scene.get_mut(0) {
                    r.object.rotation.y = t * 0.8;
                    r.object.rotation.x = t * 0.3;
                    r.object.update_model_matrix();
                    r.object.update_normal_matrix(&self.camera.view_matrix);
                }

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
    log::info!("Kansei — Spinning Box");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
