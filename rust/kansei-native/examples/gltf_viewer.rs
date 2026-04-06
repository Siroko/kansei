use std::sync::Arc;
use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::loaders::{GLTFLoader, GLTFResult};
use kansei_core::materials::{Binding, BindingResource, CullMode, Material, MaterialOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig, SharedLayouts};

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
        self.distance = (self.distance - delta * 2.0).clamp(1.0, 200.0);
    }
}

// ── Helper: create a lit material with color + specular ──

fn create_lit_material(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shared: &SharedLayouts,
    label: &str,
    color: [f32; 4],
    specular: [f32; 4],
    cull_mode: CullMode,
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
        MaterialOptions {
            cull_mode,
            ..Default::default()
        },
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

// ── Load glTF scene ──

fn load_gltf_scene(
    path: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shared: &SharedLayouts,
    scene: &mut Scene,
) -> (Vec<wgpu::Buffer>, glam::Vec3, f32) {
    let result: GLTFResult = GLTFLoader::load(path).unwrap_or_else(|e| {
        log::error!("Failed to load glTF: {}", e);
        std::process::exit(1);
    });

    log::info!(
        "Loaded glTF: {} renderables, {} materials",
        result.renderables.len(),
        result.materials.len()
    );

    let mut mat_bufs = Vec::new();

    // Compute bounding box for auto-framing
    let mut min_pos = glam::Vec3::splat(f32::MAX);
    let mut max_pos = glam::Vec3::splat(f32::MIN);

    for (i, gr) in result.renderables.into_iter().enumerate() {
        let mat_info = &result.materials[gr.material_index.min(result.materials.len().saturating_sub(1))];

        let cull = if mat_info.double_sided {
            CullMode::None
        } else {
            CullMode::Back
        };

        let (material, mat_buf) = create_lit_material(
            device,
            queue,
            shared,
            &format!("GLTF/Material{}", i),
            mat_info.base_color,
            [0.4, 0.4, 0.4, 0.3],
            cull,
        );

        let p = glam::Vec3::new(gr.position.x, gr.position.y, gr.position.z);
        min_pos = min_pos.min(p);
        max_pos = max_pos.max(p);

        let mut renderable = Renderable::new(gr.geometry, material);
        renderable.object.position = gr.position;
        renderable.object.rotation = gr.rotation;
        renderable.object.scale = gr.scale;
        renderable.object.update_model_matrix();
        renderable.object.update_world_matrix(None);
        scene.add(SceneNode::Renderable(renderable));
        mat_bufs.push(mat_buf);
    }

    // Compute framing
    let center = (min_pos + max_pos) * 0.5;
    let extent = (max_pos - min_pos).length();
    let distance = extent.max(5.0) * 1.5;

    (mat_bufs, center, distance)
}

// ── Build fallback scene (same as shadow_scene) ──

fn build_fallback_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shared: &SharedLayouts,
    scene: &mut Scene,
) -> Vec<wgpu::Buffer> {
    let mut mat_bufs = Vec::new();

    // Floor
    let floor_geo = PlaneGeometry::new(30.0, 30.0);
    let (floor_mat, floor_buf) = create_lit_material(
        device, queue, shared,
        "FloorMat",
        [0.6, 0.6, 0.6, 1.0],
        [0.15, 0.15, 0.15, 0.05],
        CullMode::Back,
    );
    let mut floor = Renderable::new(floor_geo, floor_mat);
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.object.update_model_matrix();
    floor.object.update_world_matrix(None);
    floor.cast_shadow = false;
    mat_bufs.push(floor_buf);

    // Box
    let box_geo = BoxGeometry::new(2.0, 2.0, 2.0);
    let (box_mat, box_buf) = create_lit_material(
        device, queue, shared,
        "BoxMat",
        [0.85, 0.25, 0.2, 1.0],
        [0.5, 0.5, 0.5, 0.3],
        CullMode::Back,
    );
    let mut box_obj = Renderable::new(box_geo, box_mat);
    box_obj.object.set_position(0.0, 1.0, 0.0);
    box_obj.object.update_model_matrix();
    box_obj.object.update_world_matrix(None);
    mat_bufs.push(box_buf);

    // Sphere
    let sphere_geo = SphereGeometry::new(1.2, 32, 16);
    let (sphere_mat, sphere_buf) = create_lit_material(
        device, queue, shared,
        "SphereMat",
        [0.2, 0.4, 0.9, 1.0],
        [0.8, 0.8, 0.8, 0.6],
        CullMode::Back,
    );
    let mut sphere = Renderable::new(sphere_geo, sphere_mat);
    sphere.object.set_position(3.0, 1.2, -1.0);
    sphere.object.update_model_matrix();
    sphere.object.update_world_matrix(None);
    mat_bufs.push(sphere_buf);

    scene.add(SceneNode::Renderable(floor));
    scene.add(SceneNode::Renderable(box_obj));
    scene.add(SceneNode::Renderable(sphere));

    mat_bufs
}

// ── App ──

struct App {
    gltf_path: Option<String>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    orbit: OrbitCamera,
    start_time: Instant,
    _mat_bufs: Vec<wgpu::Buffer>,
}

impl App {
    fn new(gltf_path: Option<String>) -> Self {
        Self {
            gltf_path,
            window: None,
            renderer: None,
            scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 500.0, 1280.0 / 720.0),
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

        let title = match &self.gltf_path {
            Some(p) => format!("Kansei \u{2014} glTF Viewer: {}", p),
            None => "Kansei \u{2014} glTF Viewer (fallback scene)".to_string(),
        };

        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title(&title)
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            )
            .unwrap(),
        );
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            }))
            .unwrap();

        let mut renderer = Renderer::new(RendererConfig {
            width: size.width,
            height: size.height,
            sample_count: 4,
            clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0),
            ..Default::default()
        });
        pollster::block_on(renderer.initialize(surface, &adapter));
        renderer.enable_shadows(2048);

        let device = renderer.device();
        let queue = renderer.queue();
        let shared = renderer.shared_layouts();

        // Build scene
        let (orbit_target, orbit_distance) = if let Some(ref path) = self.gltf_path {
            let (bufs, center, distance) =
                load_gltf_scene(path, device, queue, shared, &mut self.scene);
            self._mat_bufs = bufs;
            (center, distance)
        } else {
            self._mat_bufs = build_fallback_scene(device, queue, shared, &mut self.scene);
            (glam::Vec3::new(0.0, 1.0, 0.0), 15.0)
        };

        // Directional light (warm white from upper-left)
        let mut dir_light = DirectionalLight::new(
            Vec3::new(-0.3, -1.0, -0.5),
            Vec3::new(1.0, 0.95, 0.8),
            1.0,
        );
        dir_light.cast_shadow = true;
        self.scene.add(SceneNode::Light(Light::Directional(dir_light)));

        // Point light (soft warm)
        self.scene.add(SceneNode::Light(Light::Point(PointLight::new(
            Vec3::new(-3.0, 4.0, 2.0),
            Vec3::new(1.0, 0.8, 0.5),
            2.0,
            20.0,
        ))));

        // Configure orbit camera from bounding box
        self.orbit = OrbitCamera::new(orbit_target, orbit_distance, 0.5);

        // Camera
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();

        let eye = self.orbit.eye();
        self.camera.set_position(eye.x, eye.y, eye.z);
        let t = self.orbit.target;
        self.camera.look_at(&Vec3::new(t.x, t.y, t.z));

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start_time = Instant::now();

        log::info!("Kansei \u{2014} glTF Viewer ready");
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
                // Update normal matrices for all objects
                for i in 0..self.scene.children_len() {
                    if let Some(r) = self.scene.get_renderable_mut(i) {
                        r.object.update_normal_matrix();
                    }
                }

                // Update camera from orbit state
                let eye = self.orbit.eye();
                self.camera.set_position(eye.x, eye.y, eye.z);
                let t = self.orbit.target;
                self.camera.look_at(&Vec3::new(t.x, t.y, t.z));

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

    let gltf_path = std::env::args().nth(1);
    if let Some(ref path) = gltf_path {
        log::info!("Loading glTF: {}", path);
    } else {
        log::info!("No glTF path given, using fallback scene");
    }

    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new(gltf_path)).unwrap();
}
