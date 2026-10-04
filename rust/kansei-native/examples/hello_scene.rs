//! Hello scene (native twin of the WASM example): a floor, a spinning box and a sphere in the
//! stock lit material, a sun and two point lights, shadows from the sun and one lamp, and an
//! orbit camera (drag to orbit, wheel to zoom).
//!
//!   cargo run -p kansei-native --example hello_scene [-- --post] [-- --no-shadows]
//!
//! `--post` renders through a post-processing volume (bloom, then a colour grade).

use std::sync::Arc;
use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::materials::Material;
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{BloomEffect, BloomOptions, ColorGradingEffect, ColorGradingOptions};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct Running {
    window: Arc<Window>,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: Option<PostProcessingVolume>,
    cube: usize,
}

#[derive(Default)]
struct App {
    running: Option<Running>,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
    start: Option<Instant>,
}

fn build(window: Arc<Window>) -> Running {
    let post = std::env::args().any(|a| a == "--post");
    let shadows = !std::env::args().any(|a| a == "--no-shadows");
    let size = window.inner_size();
    // the post-processing GBuffer is not multisampled
    let mut renderer = Renderer::new(RendererConfig {
        width: size.width,
        height: size.height,
        sample_count: if post { 1 } else { 4 },
        clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0),
        ..Default::default()
    });
    pollster::block_on(renderer.initialize_with_target(window.clone()));
    if shadows {
        renderer.enable_shadows(2048);
        renderer.enable_point_shadows(512, 1);
    }

    let mut scene = Scene::new();
    let mut floor = Renderable::new(PlaneGeometry::new(20.0, 20.0), Material::basic_lit("Floor", [0.6, 0.6, 0.6, 1.0], [0.2, 0.2, 0.2, 0.1]));
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.cast_shadow = false;
    scene.add(SceneNode::Renderable(floor));
    let mut cube = Renderable::new(BoxGeometry::new(1.6, 1.6, 1.6), Material::basic_lit("Box", [1.6, 0.35, 0.2, 1.0], [0.5, 0.5, 0.5, 0.4]));
    cube.object.set_position(-1.2, 0.8, 0.0);
    let cube = scene.add(SceneNode::Renderable(cube));
    let mut sphere = Renderable::new(SphereGeometry::new(1.0, 48, 24), Material::basic_lit("Sphere", [0.2, 0.4, 0.9, 1.0], [0.8, 0.8, 0.8, 0.6]));
    sphere.object.set_position(1.4, 1.0, 0.4);
    scene.add(SceneNode::Renderable(sphere));

    let mut sun = DirectionalLight::new(Vec3::new(-0.4, -1.0, -0.5), Vec3::new(0.75, 0.72, 0.65), 1.0);
    sun.cast_shadow = shadows;
    scene.add(SceneNode::Light(Light::Directional(sun)));
    let mut lamp = PointLight::new(Vec3::new(-3.0, 3.5, 2.5), Vec3::new(0.7, 0.5, 0.3), 2.0, 14.0);
    lamp.cast_shadow = shadows;
    scene.add(SceneNode::Light(Light::Point(lamp)));
    scene.add(SceneNode::Light(Light::Point(PointLight::new(Vec3::new(3.5, 2.0, -2.5), Vec3::new(0.25, 0.4, 0.75), 1.5, 12.0))));

    let camera = Camera::new(45.0, 0.1, 100.0, size.width as f32 / size.height.max(1) as f32);
    let mut controls = CameraControls::new(Vec3::new(0.0, 0.8, 0.0), 11.0);
    controls.set_elevation(0.35);
    let volume = post.then(|| {
        PostProcessingVolume::new(
            &renderer,
            vec![
                Box::new(BloomEffect::new(BloomOptions { threshold: 0.9, intensity: 0.6, ..Default::default() })),
                Box::new(ColorGradingEffect::new(ColorGradingOptions { contrast: 1.15, temperature: 0.1, ..Default::default() })),
            ],
        )
    });
    Running { window, renderer, scene, camera, controls, volume, cube }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.running.is_none() {
            let attributes = Window::default_attributes().with_title("Kansei \u{2014} Hello scene").with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
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
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
            }),
            WindowEvent::RedrawRequested => {
                let t = self.start.map_or(0.0, |s| s.elapsed().as_secs_f32());
                if let Some(r) = app.scene.get_renderable_mut(app.cube) {
                    r.object.rotation.y = t * 0.6;
                }
                app.controls.update(&mut app.camera, 0.0);
                match &mut app.volume {
                    Some(volume) => app.renderer.render_with_postprocessing(&mut app.scene, &mut app.camera, volume),
                    None => app.renderer.render(&mut app.scene, &mut app.camera),
                }
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
