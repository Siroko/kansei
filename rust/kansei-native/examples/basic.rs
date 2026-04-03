use std::sync::Arc;
use kansei_core::math::Vec4;
use kansei_core::cameras::Camera;
use kansei_core::objects::Scene;
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
}

impl App {
    fn new() -> Self {
        let camera = Camera::new(45.0, 0.1, 1000.0, 1.0);
        Self { window: None, renderer: None, scene: Scene::new(), camera }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let window = Arc::new(event_loop.create_window(
            Window::default_attributes().with_title("Kansei — Basic").with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        ).unwrap());
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface), ..Default::default()
        })).unwrap();
        let mut renderer = Renderer::new(RendererConfig {
            width: size.width, height: size.height, sample_count: 1,
            clear_color: Vec4::new(0.1, 0.2, 0.3, 1.0), ..Default::default()
        });
        pollster::block_on(renderer.initialize(surface, &adapter));
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => { if let Some(r) = &mut self.renderer { r.resize(s.width, s.height); } }
            WindowEvent::RedrawRequested => {
                if let Some(r) = &mut self.renderer { r.render(&mut self.scene, &mut self.camera); }
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
