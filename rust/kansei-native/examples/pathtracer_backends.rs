//! The path tracer's two traversals side by side, headless: its BVH4s walked in compute
//! (software, the web's only one) against hardware ray queries over acceleration structures of
//! the same scene (`BVHBuilder::set_backend`), on the web demo's and `pathtracer_test`'s Cornell box
//! (walls, two boxes, a glass Stanford dragon, the sun) at its settings (8 bounces):
//!
//! - `cornell`: the decimated dragon (`stanford_dragon_pbr.glb`, 19k triangles);
//! - `full`: the whole scan (`scene.gltf`, 871k triangles).
//!
//! Each backend traces in turn, blocks of frames alternating (other work on the GPU then weighs on
//! both alike), the camera still (the samples accumulate, each frame costs the same). Reported:
//! the frame's wall time (the CPU waiting for each frame's GPU work), the trace pass's GPU time
//! (the profiler's `PathTracer/Trace`), the rays a frame casts (each closest-hit and any-hit query,
//! counted in separate frames by `PathTracer::set_count_rays`), and rays per second over the trace
//! pass's time.
//!
//! ```text
//! cargo run --release -p kansei-native --example pathtracer_backends -- [cornell|full|all] [--backend both|software|hardware] [--size WxH] [--spp N] [--bounces N] [--rounds N] [--frames N]
//! ```

use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::Material;
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pathtracer::{BVHBuilder, PathTracer, PathTracerMaterial, TLASBuilder};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::RtBackend;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Cornell,
    Full,
}

struct Args {
    scenes: Vec<Kind>,
    backends: Vec<RtBackend>,
    size: (u32, u32),
    spp: u32,
    bounces: u32,
    rounds: usize,
    frames: usize,
}

fn args() -> Args {
    let mut a = Args { scenes: vec![Kind::Cornell, Kind::Full], backends: vec![RtBackend::Software, RtBackend::Hardware], size: (1920, 1080), spp: 1, bounces: 8, rounds: 4, frames: 24 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "cornell" => a.scenes = vec![Kind::Cornell],
            "full" => a.scenes = vec![Kind::Full],
            "all" => {}
            "--backend" => {
                a.backends = match it.next().as_deref() {
                    Some("software") => vec![RtBackend::Software],
                    Some("hardware") => vec![RtBackend::Hardware],
                    _ => vec![RtBackend::Software, RtBackend::Hardware],
                }
            }
            "--spp" => a.spp = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.spp),
            "--bounces" => a.bounces = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.bounces),
            "--rounds" => a.rounds = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.rounds),
            "--frames" => a.frames = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.frames),
            "--size" => {
                if let Some((w, h)) = it.next().and_then(|v| v.split_once('x').map(|(w, h)| (w.parse().unwrap_or(1920), h.parse().unwrap_or(1080)))) {
                    a.size = (w, h);
                }
            }
            other => eprintln!("unknown argument {other}"),
        }
    }
    a
}

/// `pathtracer_test`'s Cornell box with `kind`'s dragon in glass, and the tracer's materials.
fn cornell(kind: Kind) -> (Scene, Vec<PathTracerMaterial>, usize) {
    let mut scene = Scene::new();
    let mut materials = Vec::new();
    let mut add = |scene: &mut Scene, geometry, m: PathTracerMaterial, at: [f32; 3]| {
        let mut r = Renderable::new(geometry, Material::basic_lit("Surface", [0.8, 0.8, 0.8, 1.0], [0.15, 0.15, 0.15, 0.5]));
        r.object.set_position(at[0], at[1], at[2]);
        scene.add(SceneNode::Renderable(r));
        materials.push(m);
    };
    let diffuse = |albedo| PathTracerMaterial { albedo, ..Default::default() };
    add(&mut scene, BoxGeometry::new(8.0, 0.1, 8.0), diffuse([0.73; 3]), [0.0, -0.05, 0.0]);
    add(&mut scene, BoxGeometry::new(8.0, 0.1, 8.0), diffuse([0.73; 3]), [0.0, 6.05, 0.0]);
    add(&mut scene, BoxGeometry::new(8.0, 6.0, 0.1), diffuse([0.73; 3]), [0.0, 3.0, -4.05]);
    add(&mut scene, BoxGeometry::new(0.1, 6.0, 8.0), diffuse([0.65, 0.05, 0.05]), [-4.05, 3.0, 0.0]);
    add(&mut scene, BoxGeometry::new(0.1, 6.0, 8.0), diffuse([0.12, 0.45, 0.15]), [4.05, 3.0, 0.0]);
    add(&mut scene, BoxGeometry::new(1.0, 2.0, 1.0), diffuse([0.9, 0.2, 0.2]), [-1.5, 1.0, 0.0]);
    add(&mut scene, BoxGeometry::new(1.0, 1.0, 1.0), diffuse([0.2, 0.2, 0.9]), [1.5, 0.5, 0.0]);
    let assets = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/assets/geom");
    let dragon = match kind {
        Kind::Cornell => GLTFLoader::load(&format!("{assets}/stanford_dragon_pbr.glb")),
        Kind::Full => GLTFLoader::load(&format!("{assets}/scene.gltf")),
    }
    .expect("the Stanford dragon");
    // about 3 m long, standing on the floor in front of the boxes, as pathtracer_test's
    let geometry = dragon.merged_geometry("Dragon").fit(glam::Vec3::new(3.0, f32::INFINITY, 3.0));
    let triangles = geometry.indices.len() / 3;
    add(&mut scene, geometry, PathTracerMaterial::glass(1.5), [0.0, 0.0, 2.0]);
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(-0.5, -1.0, -0.3).normalize(), Vec3::new(1.0, 0.95, 0.9), 3.0))));
    (scene, materials, triangles)
}

#[derive(Default)]
struct Totals {
    wall: Vec<f64>,
    trace_ms: f64,
    traces: usize,
    rays: Vec<u32>,
}

fn run(kind: Kind, args: &Args, adapter: &wgpu::Adapter) {
    let (w, h) = args.size;
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(adapter));
    let (mut scene, materials, dragon) = cornell(kind);
    let mut camera = Camera::new(45.0, 0.1, 100.0, w as f32 / h as f32);
    camera.set_position(0.0, 3.0, 14.0);
    camera.look_at(&Vec3::new(0.0, 3.0, 0.0));
    camera.update_projection_matrix();
    camera.update_view_matrix();
    scene.prepare(&camera.position());

    // the scene ready for each backend
    let scenes: Vec<_> = args
        .backends
        .iter()
        .map(|&backend| {
            let t0 = Instant::now();
            let mut bvh = BVHBuilder::new();
            bvh.set_backend(backend);
            let mut tlas = TLASBuilder::new(&renderer);
            let data = bvh.build_full(&renderer, &scene, &mut tlas);
            let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
            eprintln!("  {backend:?}: scene ready in {:.0} ms (traced by {:?})", t0.elapsed().as_secs_f64() * 1e3, data.backend());
            (data, tlas)
        })
        .collect();
    let mut tracer = PathTracer::new(&renderer);
    tracer.resize(w, h);
    tracer.set_spp(args.spp);
    tracer.set_max_bounces(args.bounces);
    tracer.set_materials(&materials);
    let lights = tracer.set_lights_from_scene(&scene);

    renderer.set_profiling(true);
    let mut totals: Vec<Totals> = args.backends.iter().map(|_| Totals::default()).collect();
    let warmup = 8;
    for round in 0..args.rounds {
        for (b, (data, tlas)) in scenes.iter().enumerate() {
            for k in 0..warmup + args.frames {
                let measured = k >= warmup;
                if k == warmup {
                    let _ = renderer.take_profile();
                }
                let t0 = Instant::now();
                tracer.trace_frame(&renderer, data, tlas, &camera, lights);
                renderer.end_profiled_frame();
                let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
                if measured {
                    totals[b].wall.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let profile = renderer.take_profile();
            for p in profile.gpu.iter().filter(|p| p.label == "PathTracer/Trace") {
                totals[b].trace_ms += p.busy_ms;
                totals[b].traces += 1;
            }
            let t = &totals[b];
            eprintln!("  block {round} {:?}: wall {:.2} ms", args.backends[b], median(&t.wall[t.wall.len() - args.frames..]));
        }
    }
    // the rays a frame casts, counted apart (the counter slows the trace)
    tracer.set_count_rays(true);
    for (b, (data, tlas)) in scenes.iter().enumerate() {
        let _ = tracer.take_ray_count(&renderer);
        for _ in 0..8 {
            tracer.trace_frame(&renderer, data, tlas, &camera, lights);
            totals[b].rays.push(tracer.take_ray_count(&renderer));
        }
    }

    let info = renderer.adapter_info();
    let name = match kind {
        Kind::Cornell => "cornell (the decimated dragon in glass)",
        Kind::Full => "full (the whole dragon scan in glass)",
    };
    println!("\n## {name}\n{} ({:?}), {w}x{h}, {} spp, {} bounces, {} rounds of {} frames a backend", info.name, info.backend, args.spp, args.bounces, args.rounds, args.frames);
    let header: Vec<String> = args.backends.iter().map(|b| format!("{b:?}")).collect();
    println!("| | {} |", header.join(" | "));
    println!("|---|{}", "---|".repeat(args.backends.len()));
    println!("| triangles (dragon) | {} |", scenes.iter().map(|(d, _)| format!("{} ({dragon})", d.triangle_count)).collect::<Vec<_>>().join(" | "));
    println!("| frame, wall (median) | {} |", totals.iter().map(|t| format!("{:.2} ms", median(&t.wall))).collect::<Vec<_>>().join(" | "));
    let trace = |t: &Totals| t.trace_ms / t.traces.max(1) as f64;
    println!("| PathTracer/Trace (GPU) | {} |", totals.iter().map(|t| format!("{:.3} ms", trace(t))).collect::<Vec<_>>().join(" | "));
    let rays = |t: &Totals| t.rays.iter().map(|&r| r as f64).sum::<f64>() / t.rays.len().max(1) as f64;
    println!("| rays a frame | {} |", totals.iter().map(|t| format!("{:.2} M", rays(t) / 1e6)).collect::<Vec<_>>().join(" | "));
    println!("| rays per second (trace pass) | {} |", totals.iter().map(|t| format!("{:.0} M", rays(t) / trace(t) / 1e3)).collect::<Vec<_>>().join(" | "));
    println!("| rays per second (frame) | {} |", totals.iter().map(|t| format!("{:.0} M", rays(t) / median(&t.wall) / 1e3)).collect::<Vec<_>>().join(" | "));
    // (no teardown: on Vulkan a device didn't always finish dropping; see rt_backends)
    std::mem::forget(scenes);
    std::mem::forget(tracer);
    std::mem::forget(renderer);
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn main() {
    env_logger::init();
    let args = args();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, driver {} {}); hardware ray queries: {}", info.name, info.backend, info.driver, info.driver_info, adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY));
    // a thread a scene: the profiler is the thread's, for one device
    let args = std::sync::Arc::new(args);
    for kind in args.scenes.clone() {
        let (args, adapter) = (args.clone(), adapter.clone());
        std::thread::spawn(move || run(kind, &args, &adapter)).join().expect("scene");
    }
    std::process::exit(0);
}
