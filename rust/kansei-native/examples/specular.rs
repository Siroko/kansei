//! Specular factor (native twin of the WASM example): the same pieces twice, side by side under
//! a low sun ahead of the camera, on the left with the default specular factor (1: a
//! dielectric's Fresnel, a highlight on the balls and a sheen on the board), on the right with a
//! factor of 0, matte (Lambert alone). Drag to orbit, wheel to zoom.
//!
//! ```text
//! cargo run -p kansei-native --example specular [-- --rt] [-- --matte <factor>]
//! cargo run --release -p kansei-native --example specular -- --bench [--size WxH] [--rounds N] [--frames N]
//! ```
//!
//! `--rt` leaves the sun's light to `RtShadowsEffect` (ray-traced shadows, the factor read from the
//! GBuffer) instead of the stock material's own lighting and shadow map; `--matte` sets the right
//! side's factor. `--bench` renders headless, with nothing on screen: both lightings in turn,
//! blocks of frames alternating between every surface at the default factor and the right side
//! matte, and prints each one's GBuffer pass and `RtShadowsEffect` passes (GPU time, the
//! renderer's profiler) and the frame's wall time.

use std::sync::Arc;
use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, Geometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{ToneMapEffect, ToneMapOptions};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtGridOptions, RtShadowsEffect, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const WOOD: [f32; 3] = [0.55, 0.3, 0.17];
const GROUND: [f32; 3] = [0.16, 0.16, 0.17];
const SKY_UP: [f32; 3] = [0.22, 0.27, 0.36];
const SKY_DOWN: [f32; 3] = [0.05, 0.045, 0.04];

/// A surface whose sun light `RtShadowsEffect` adds (the WASM example's).
fn rt_material(color: [f32; 3], roughness: f32, specular: f32) -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
{RT_SHADOWS_GBUFFER_WGSL}
struct Surface {{ albedo: vec4f, skyUp: vec4f, skyDown: vec4f }};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4f;
struct VOut {{ @builtin(position) clip: vec4f, @location(0) normal: vec3f }};
@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f) -> VOut {{
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.normal = (normal_matrix * vec4f(normal, 0.0)).xyz;
    return out;
}}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {{
    let n = normalize(in.normal);
    let base = surface.albedo.rgb;
    let sky = mix(surface.skyDown.rgb, surface.skyUp.rgb, n.y * 0.5 + 0.5);
    // roughness and specular factor in the albedo's and the sky's alphas
    let out = kansei_gbuffer_out_rt_lit(base * sky, vec3f(0.0), n, base, surface.albedo.a, 0.04, false);
    return kansei_gbuffer_with_specular(out, surface.skyUp.a);
}}
"#
    );
    let mut m = Material::new("RtLit", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    let ([r, g, b], [ur, ug, ub], [dr, dg, db]) = (color, SKY_UP, SKY_DOWN);
    m.set_uniform_bindable(0, "RtLit/Surface", &[r, g, b, roughness, ur, ug, ub, specular, dr, dg, db, 0.0]);
    m
}

fn standard_material(color: [f32; 3], roughness: f32, specular: f32) -> Material {
    Material::standard_lit("Standard", &StandardLitOptions { base_color: color, roughness, specular, sky_up: SKY_UP, sky_down: SKY_DOWN, ..Default::default() })
}

/// The WASM example's scene: the default on the left, `matte`'s factor on the right.
fn scene(rt: bool, matte: f32) -> Scene {
    let mut scene = Scene::new();
    let add = |scene: &mut Scene, geometry: Geometry, color: [f32; 3], roughness: f32, specular: f32, at: [f32; 3], tilt: f32| {
        let material = if rt { rt_material(color, roughness, specular) } else { standard_material(color, roughness, specular) };
        let mut r = Renderable::new(geometry, material).with_rt(RtSurface::new(color));
        r.object.set_position(at[0], at[1], at[2]);
        r.object.rotation.x = tilt;
        scene.add(SceneNode::Renderable(r));
    };
    add(&mut scene, BoxGeometry::new(12.0, 0.1, 12.0), GROUND, 0.9, 1.0, [0.0, -0.05, -1.0], 0.0);
    for (x, specular) in [(-1.45, 1.0), (1.45, matte)] {
        add(&mut scene, BoxGeometry::new(2.5, 0.12, 4.6), WOOD, 0.6, specular, [x, 0.06, -1.0], 0.0);
        for (k, roughness) in [0.35, 0.6, 0.85].into_iter().enumerate() {
            add(&mut scene, SphereGeometry::new(0.36, 48, 24), WOOD, roughness, specular, [x - 0.55, 0.48, 0.4 - 1.3 * k as f32], 0.0);
        }
        add(&mut scene, BoxGeometry::new(0.7, 0.08, 3.2), WOOD, 0.6, specular, [x + 0.6, 0.45, -1.0], -0.22);
    }
    let mut sun = DirectionalLight::new(Vec3::new(-0.25, -0.32, 1.0), Vec3::new(1.0, 0.95, 0.88), 3.0);
    sun.cast_shadow = !rt;
    scene.add(SceneNode::Light(Light::Directional(sun)));
    scene
}

fn grid_options() -> SceneRtGridOptions {
    SceneRtGridOptions { grid: RtGridOptions { dims: [128, 32, 128], cell: 12.0 / 128.0, fixed_origin: Some(glam::Vec3::new(-6.0, -0.2, -7.0)), ..Default::default() }, ..Default::default() }
}

fn arg<T: std::str::FromStr>(name: &str) -> Option<T> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
}

struct Running {
    window: Arc<Window>,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
}

#[derive(Default)]
struct App {
    running: Option<Running>,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

fn build(window: Arc<Window>) -> Running {
    let rt = std::env::args().any(|a| a == "--rt");
    let size = window.inner_size();
    let mut renderer = Renderer::new(RendererConfig { width: size.width, height: size.height, sample_count: 1, clear_color: Vec4::new(0.02, 0.02, 0.03, 1.0), ..Default::default() });
    pollster::block_on(renderer.initialize_with_target(window.clone()));
    let adapter = renderer.adapter_info();
    println!("adapter: {} ({:?}, driver {} {})", adapter.name, adapter.backend, adapter.driver, adapter.driver_info);
    if rt {
        renderer.enable_rt_grid(grid_options());
    } else {
        renderer.enable_shadows(2048);
    }
    let scene = scene(rt, arg("--matte").unwrap_or(0.0));
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(grid) = renderer.rt_grid() {
        let mut shadows = RtShadowsEffect::new(grid.handle());
        shadows.update_lights(scene.lights());
        effects.push(Box::new(shadows));
    }
    effects.push(Box::new(ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()))));
    let volume = PostProcessingVolume::new(&renderer, effects);
    let camera = Camera::new(40.0, 0.1, 100.0, size.width as f32 / size.height.max(1) as f32);
    let mut controls = CameraControls::new(Vec3::new(0.0, 0.3, -0.8), 7.5);
    controls.set_elevation(0.32);
    Running { window, renderer, scene, camera, controls, volume }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.running.is_none() {
            let attributes = Window::default_attributes().with_title("Kansei \u{2014} Specular factor").with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            self.running = Some(build(Arc::new(el.create_window(attributes).unwrap())));
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
                app.controls.update(&mut app.camera, 0.0);
                app.renderer.render_with_postprocessing(&mut app.scene, &mut app.camera, &mut app.volume);
                app.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// The passes `--bench` reports.
const PASSES: [&str; 8] = ["Renderer/ShadowPass", "Renderer/GBufferOpaquePass", "Renderer/GBufferPass", "RtShadows/Trace", "RtShadows/Temporal", "RtShadows/Atrous1", "RtShadows/Atrous2", "RtShadows/Composite"];

fn bench() {
    let (w, h) = arg::<String>("--size").and_then(|s| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))).unwrap_or((1920u32, 1080u32));
    let rounds = arg("--rounds").unwrap_or(6usize);
    let frames = arg("--frames").unwrap_or(48usize);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no adapter");
    let info = adapter.get_info();
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    renderer.enable_shadows(2048);
    renderer.set_profiling(true);
    let gbuffer = GBuffer::new(renderer.device(), w, h, 1);
    let output = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("Bench/Out"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let output = output.create_view(&Default::default());
    let mut camera = Camera::new(40.0, 0.1, 100.0, w as f32 / h as f32);
    camera.set_position(0.0, 2.6, 6.4);
    camera.look_at(&Vec3::new(0.0, 0.3, -0.8));
    camera.update_projection_matrix();
    // (lighting, right side's factor): every surface at the default, then the right side matte
    let cases = [(false, 1.0), (false, 0.0), (true, 1.0), (true, 0.0)];
    let mut scenes: Vec<Scene> = cases.iter().map(|&(rt, matte)| scene(rt, matte)).collect();
    // per case: each pass's mean ms per block, and the frames' wall times
    let mut passes: Vec<Vec<(&'static str, Vec<f64>)>> = cases.iter().map(|_| PASSES.iter().map(|&p| (p, Vec::new())).collect()).collect();
    let mut wall: Vec<Vec<f64>> = cases.iter().map(|_| Vec::new()).collect();
    let warmup = 16;
    for round in 0..rounds {
        for (c, &(rt, _)) in cases.iter().enumerate() {
            let mut effect = if rt {
                renderer.enable_rt_grid(grid_options());
                let mut e = RtShadowsEffect::new(renderer.rt_grid().unwrap().handle());
                e.update_lights(scenes[c].lights());
                Some(e)
            } else {
                renderer.disable_rt_grid();
                None
            };
            for k in 0..warmup + frames {
                if k == warmup {
                    let _ = renderer.take_profile();
                }
                let t0 = Instant::now();
                renderer.render_scene_offscreen(&mut scenes[c], &mut camera, &gbuffer);
                if let Some(e) = &mut effect {
                    let mut encoder = renderer.device().create_command_encoder(&Default::default());
                    e.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output, &camera, w, h);
                    renderer.queue().submit(Some(encoder.finish()));
                }
                renderer.end_profiled_frame();
                let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
                if k >= warmup {
                    wall[c].push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let profile = renderer.take_profile();
            for (label, times) in &mut passes[c] {
                let ms: Vec<f64> = profile.gpu.iter().filter(|p| p.label == *label).map(|p| p.busy_ms).collect();
                if !ms.is_empty() {
                    times.push(ms.iter().sum::<f64>() / ms.len() as f64);
                }
            }
        }
        eprintln!("  round {} of {rounds}", round + 1);
    }
    println!("\n## Specular factor: {} ({:?}), {w}x{h}, {rounds} rounds of {frames} frames a case, medians of the rounds' means", info.name, info.backend);
    let name = |&(rt, matte): &(bool, f32)| format!("{}, right side {}", if rt { "RtShadowsEffect" } else { "standard_lit" }, if matte == 0.0 { "matte" } else { "default" });
    println!("| | {} |", cases.iter().map(name).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---|".repeat(cases.len()));
    println!("| frame, wall | {} |", wall.iter().map(|t| format!("{:.3} ms", median(t))).collect::<Vec<_>>().join(" | "));
    for (i, label) in PASSES.iter().enumerate() {
        if passes.iter().all(|p| p[i].1.is_empty()) {
            continue;
        }
        let cells: Vec<String> = passes.iter().map(|p| if p[i].1.is_empty() { "—".into() } else { format!("{:.3} ms", median(&p[i].1)) }).collect();
        println!("| {label} | {} |", cells.join(" | "));
    }
    // (no teardown: on Vulkan a device didn't finish dropping in rt_backends)
    std::mem::forget(renderer);
    std::process::exit(0);
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn main() {
    env_logger::init();
    if std::env::args().any(|a| a == "--bench") {
        return bench();
    }
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::default()).unwrap();
}
