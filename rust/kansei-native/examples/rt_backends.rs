//! The ray tracing grid's two backends side by side, headless: software (the grid's cells walked
//! in compute, the web's only one) against hardware ray queries (`rt::RtBackend`), on scenes set up
//! as the RT examples set theirs (they run on the web only, where there is no hardware backend):
//!
//! - `room`: gi-box's room (its 96³ grid of 4.8 cm cells, the walls, blocks, rug, a chrome and a
//!   glass ball, the lamp) with a dense mesh where its dragon stands (`--dense <subdivisions>`: an
//!   icosphere of 20 × 4^n triangles, 5 by default, about the light dragon's 19k; 8 is 1.3 M, past
//!   the full scan's 871k), hybrid diffuse GI and reflections with glass, static: built once.
//! - `drive`: outdoor-gi's grid (128 × 64 × 128 half-metre cells round the eye) over a field of
//!   blocks, the camera driving through it, so the grid rebuilds as its box steps along.
//! - `particles`: glass-particles' room (96 × 64 × 96 cells) with a few hundred glass beads that
//!   move every frame, rebuilt every frame, reflections and their glass pass.
//! - `rolling`: the room with its glass ball moving (`dynamic`): the grid's moving layer holds it,
//!   rebuilt every frame, the static layer built once (`--moving-layer off|both` also measures the
//!   grid without the moving layer, rebuilt whole every frame, `SceneRtGrid::set_moving_layer`).
//!
//! Each backend renders in turn, blocks of frames alternating (other work on the GPU then weighs
//! on both alike), and the profiler's pass times and the frame's wall time (the CPU waiting for
//! each frame's GPU work, Metal's waits for the BLAS included) are averaged per backend. The room,
//! static, is measured again rebuilding every frame: the difference is a rebuild's cost (the AS
//! builds have no pass for the profiler to time; `Rt/BlasInput` is the BLAS's vertices).
//!
//! ```text
//! cargo run --release -p kansei-native --example rt_backends -- [room|room-static|room-rebuilt|drive|particles|rolling|all] [--backend both|software|hardware] [--moving-layer on|off|both] [--dense N] [--size WxH] [--rounds N] [--frames N]
//! ```

use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry, PlaneGeometry};
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtBackend, RtDiffuseGiEffect, RtDiffuseGiOptions, RtGlass, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions, RtSurface, SceneRtGridOptions};

/// The passes reported, in frame order.
const PASSES: [&str; 16] = [
    "Rt/Gather",
    "Rt/GridCount",
    "Rt/GridScan",
    "Rt/GridFill",
    "Rt/GatherMoving",
    "Rt/MovingCount",
    "Rt/MovingScan",
    "Rt/MovingFill",
    "Rt/BlasInput",
    "RtGi/Trace",
    "RtGi/Temporal",
    "RtGi/Variance",
    "RtGi/Composite",
    "Rt/Trace",
    "Rt/Glass",
    "Rt/Resolve",
];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Room,
    /// the room, its grid rebuilt every frame
    RoomRebuilt,
    Drive,
    Particles,
    /// the room, its glass ball moving
    Rolling,
}

struct Args {
    scenes: Vec<Kind>,
    backends: Vec<RtBackend>,
    /// with the grid's moving layer, without, or both
    layers: Vec<bool>,
    dense: u32,
    size: (u32, u32),
    rounds: usize,
    frames: usize,
}

fn args() -> Args {
    let mut a = Args { scenes: vec![Kind::Room, Kind::RoomRebuilt, Kind::Drive, Kind::Particles], backends: vec![RtBackend::Software, RtBackend::Hardware], layers: vec![true], dense: 5, size: (1920, 1080), rounds: 4, frames: 24 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "room" => a.scenes = vec![Kind::Room, Kind::RoomRebuilt],
            "room-static" => a.scenes = vec![Kind::Room],
            "room-rebuilt" => a.scenes = vec![Kind::RoomRebuilt],
            "drive" => a.scenes = vec![Kind::Drive],
            "particles" => a.scenes = vec![Kind::Particles],
            "rolling" => a.scenes = vec![Kind::Rolling],
            "--moving-layer" => {
                a.layers = match it.next().as_deref() {
                    Some("off") => vec![false],
                    Some("both") => vec![true, false],
                    _ => vec![true],
                }
            }
            "all" => {}
            "--backend" => {
                a.backends = match it.next().as_deref() {
                    Some("software") => vec![RtBackend::Software],
                    Some("hardware") => vec![RtBackend::Hardware],
                    _ => vec![RtBackend::Software, RtBackend::Hardware],
                }
            }
            "--dense" => a.dense = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.dense),
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

fn lit(color: [f32; 3], roughness: f32) -> Material {
    Material::standard_lit("Lit", &StandardLitOptions { base_color: color, roughness, ..Default::default() })
}

/// A static, lit, voxelized, ray traced block.
fn block(scene: &mut Scene, size: [f32; 3], at: [f32; 3], color: [f32; 3]) {
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), lit(color, 0.6)).with_gi(GiSurface::new(color)).with_rt(RtSurface::new(color));
    r.object.set_position(at[0], at[1], at[2]);
    scene.add(SceneNode::Renderable(r));
}

fn glass_ball(radius: f32, subdivisions: u32) -> Renderable {
    let tint = [0.92, 0.96, 1.0];
    let mut r = Renderable::new(IcosphereGeometry::new(radius, subdivisions), Material::standard_lit("Glass", &StandardLitOptions::glass(tint, 1.5, 0.0))).with_rt(RtSurface::glass(tint));
    r.cast_shadow = false;
    r
}

/// The scene, the grid round it, and where the camera goes on frame `f`.
struct Setup {
    scene: Scene,
    grid: SceneRtGridOptions,
    camera: Box<dyn Fn(usize) -> ([f32; 3], [f32; 3])>,
    /// beads that move each frame
    beads: Vec<(usize, [f32; 3])>,
    /// the room's glass ball
    ball: Option<usize>,
    glass: bool,
    gi: bool,
}

fn room(renderer: &mut Renderer, dense: u32) -> Setup {
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-2.3, -0.3, -4.3], bounds_max: [2.3, 4.3, 0.3], ..Default::default() });
    let mut scene = Scene::new();
    let (white, red, green) = ([0.73; 3], [0.63, 0.065, 0.05], [0.14, 0.45, 0.09]);
    // the room: 4 m wide, high and deep, open toward the camera, walls 0.2 m thick
    block(&mut scene, [4.4, 0.2, 4.2], [0.0, -0.1, -2.0], white);
    block(&mut scene, [4.4, 0.2, 4.2], [0.0, 4.1, -2.0], white);
    block(&mut scene, [4.4, 4.0, 0.2], [0.0, 2.0, -4.1], white);
    block(&mut scene, [0.2, 4.0, 4.2], [-2.1, 2.0, -2.0], red);
    block(&mut scene, [0.2, 4.0, 4.2], [2.1, 2.0, -2.0], green);
    block(&mut scene, [1.2, 2.4, 1.2], [-0.8, 1.2, -2.6], white);
    block(&mut scene, [1.2, 1.2, 1.2], [0.8, 0.6, -1.5], white);
    block(&mut scene, [2.4, 0.02, 0.8], [0.0, 0.01, -0.55], [0.8, 0.5, 0.2]);
    // where the dragon stands: a dense mesh
    let mut dense_mesh = Renderable::new(IcosphereGeometry::new(0.45, dense), lit([0.75, 0.6, 0.35], 0.5)).with_gi(GiSurface::new([0.75, 0.6, 0.35])).with_rt(RtSurface::new([0.75, 0.6, 0.35]));
    dense_mesh.object.set_position(-0.8, 2.85, -2.6);
    scene.add(SceneNode::Renderable(dense_mesh));
    let mut chrome = Renderable::new(IcosphereGeometry::new(0.3, 4), Material::standard_lit("Chrome", &StandardLitOptions::mirror([0.9; 3], 0.05))).with_gi(GiSurface::new([0.5; 3])).with_rt(RtSurface::new([0.9; 3]).with_smooth_normals().with_specular(0.9));
    chrome.object.set_position(0.8, 1.5, -1.5);
    scene.add(SceneNode::Renderable(chrome));
    let mut glass = glass_ball(0.3, 4);
    glass.object.set_position(-0.1, 0.33, -0.45);
    let ball = scene.add(SceneNode::Renderable(glass));
    lamp(&mut scene);
    Setup {
        scene,
        grid: SceneRtGridOptions { grid: RtGridOptions { dims: [96, 96, 96], cell: 4.6 / 96.0, fixed_origin: Some(glam::Vec3::new(-2.3, -0.3, -4.3)), ..Default::default() }, ..Default::default() },
        camera: Box::new(|_| ([0.0, 2.0, 4.2], [0.0, 1.6, -2.0])),
        beads: Vec::new(),
        ball: Some(ball),
        glass: true,
        gi: true,
    }
}

fn lamp(scene: &mut Scene) {
    let lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    scene.add(SceneNode::Light(Light::Spot(lamp)));
}

/// A small deterministic generator.
fn hash(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9e37_79b9) ^ 0x85eb_ca6b;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d);
    x ^= x >> 12;
    (x >> 8) as f32 / 16_777_216.0
}

fn drive(renderer: &mut Renderer) -> Setup {
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-64.0, -1.0, -160.0], bounds_max: [64.0, 31.0, 32.0], ..Default::default() });
    let mut scene = Scene::new();
    let mut ground = Renderable::new(PlaneGeometry::new(200.0, 260.0), lit([0.3, 0.32, 0.25], 0.9)).with_gi(GiSurface::new([0.3, 0.32, 0.25])).with_rt(RtSurface::new([0.3, 0.32, 0.25]));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    ground.object.set_position(0.0, 0.0, -60.0);
    scene.add(SceneNode::Renderable(ground));
    // trunks and crowns along a road: blocks of 2-8 m either side
    for k in 0..900u32 {
        let side = if k % 2 == 0 { -1.0 } else { 1.0 };
        let x = side * (4.0 + hash(k * 3) * 50.0);
        let z = 25.0 - hash(k * 3 + 1) * 180.0;
        let h = 2.0 + hash(k * 3 + 2) * 6.0;
        let c = [0.25 + 0.2 * hash(k), 0.35, 0.2];
        block(&mut scene, [0.6 + h * 0.15, h, 0.6 + h * 0.15], [x, h * 0.5, z], c);
    }
    let sun = kansei_core::lights::DirectionalLight::new(Vec3::new(-0.4, -1.0, -0.3), Vec3::new(1.0, 0.95, 0.9), 5.0);
    scene.add(SceneNode::Light(Light::Directional(sun)));
    Setup {
        scene,
        grid: SceneRtGridOptions { grid: RtGridOptions { dims: [128, 64, 128], cell: 0.5, below: 0.25, ..Default::default() }, ..Default::default() },
        // 12 m/s along the road (a fifth of a metre a frame at 60 Hz)
        camera: Box::new(|f| {
            let z = 20.0 - 0.2 * (f % 600) as f32;
            ([0.0, 1.6, z], [0.0, 1.4, z - 10.0])
        }),
        beads: Vec::new(),
        ball: None,
        glass: false,
        gi: true,
    }
}

fn particles(renderer: &mut Renderer) -> Setup {
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-3.0, -0.2, -3.0], bounds_max: [3.0, 3.8, 3.0], ..Default::default() });
    let mut scene = Scene::new();
    block(&mut scene, [6.0, 0.2, 6.0], [0.0, -0.1, 0.0], [0.05; 3]);
    block(&mut scene, [6.0, 4.0, 0.2], [0.0, 2.0, -3.0], [0.6; 3]);
    block(&mut scene, [0.2, 4.0, 6.0], [-3.0, 2.0, 0.0], [0.6, 0.3, 0.2]);
    let mut beads = Vec::new();
    for k in 0..400u32 {
        let home = [(hash(k * 5) - 0.5) * 4.0, 0.4 + hash(k * 5 + 1) * 2.4, (hash(k * 5 + 2) - 0.5) * 4.0];
        let mut bead = glass_ball(0.07 + 0.05 * hash(k * 5 + 3), 2);
        bead.dynamic = true;
        bead.object.set_position(home[0], home[1], home[2]);
        beads.push((scene.add(SceneNode::Renderable(bead)), home));
    }
    lamp(&mut scene);
    Setup {
        scene,
        grid: SceneRtGridOptions { grid: RtGridOptions { dims: [96, 64, 96], cell: 6.0 / 96.0, fixed_origin: Some(glam::Vec3::new(-3.0, -0.2, -3.0)), ..Default::default() }, rebuild_every_frame: true, ..Default::default() },
        camera: Box::new(|_| ([0.0, 1.8, 4.5], [0.0, 1.2, 0.0])),
        beads,
        ball: None,
        glass: true,
        gi: false,
    }
}

/// What a block of frames measured: (label, ms) for each pass, the frames' wall ms.
#[derive(Default)]
struct Totals {
    passes: Vec<(&'static str, f64, usize)>,
    wall: Vec<f64>,
    triangles: u32,
    accel: u32,
}

fn effects(renderer: &Renderer, setup: &Setup) -> Vec<Box<dyn PostProcessingEffect>> {
    let volume = renderer.voxel_gi().unwrap().volume();
    let grid = renderer.rt_grid().unwrap().handle();
    let mut out: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if setup.gi {
        let mut gi = RtDiffuseGiEffect::with_volume(volume, grid.clone(), RtDiffuseGiOptions { max_distance: 20.0, ..Default::default() });
        gi.update_lights(setup.scene.lights());
        gi.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        out.push(Box::new(gi));
    }
    let mut reflections = RtReflectionsEffect::with_volume(volume, grid, RtReflectionsOptions { max_distance: 20.0, ..Default::default() });
    reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
    reflections.set_glass(setup.glass.then_some(RtGlass::default()));
    out.push(Box::new(reflections));
    out
}

fn run(kind: Kind, args: &Args, adapter: &wgpu::Adapter) {
    let (w, h) = args.size;
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(adapter));
    let mut setup = match kind {
        Kind::Room => room(&mut renderer, args.dense),
        Kind::RoomRebuilt => {
            let mut s = room(&mut renderer, args.dense);
            s.grid.rebuild_every_frame = true;
            s
        }
        Kind::Drive => drive(&mut renderer),
        Kind::Particles => particles(&mut renderer),
        Kind::Rolling => {
            let mut s = room(&mut renderer, args.dense);
            let ball = s.ball.unwrap();
            let r = s.scene.get_renderable_mut(ball).unwrap();
            r.dynamic = true;
            let p = r.object.position;
            s.beads.push((ball, [p.x, p.y, p.z]));
            s
        }
    };
    let name = match kind {
        Kind::Room => format!("room (dense mesh of {} triangles)", 20 * 4u64.pow(args.dense)),
        Kind::RoomRebuilt => format!("room, rebuilt every frame (dense mesh of {} triangles)", 20 * 4u64.pow(args.dense)),
        Kind::Drive => "drive (grid round a moving eye)".into(),
        Kind::Particles => "particles (400 glass beads, rebuilt every frame)".into(),
        Kind::Rolling => format!("room, its glass ball moving (dense mesh of {} triangles)", 20 * 4u64.pow(args.dense)),
    };
    let gbuffer = GBuffer::new(renderer.device(), w, h, 1);
    let target = |label| {
        renderer
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let ping = [target("Bench/A"), target("Bench/B")];
    let mut camera = Camera::new(50.0, 0.1, 300.0, w as f32 / h as f32);
    renderer.set_profiling(true);
    // each backend with and without the moving layer, as asked
    let modes: Vec<(RtBackend, bool)> = args.backends.iter().flat_map(|&b| args.layers.iter().map(move |&l| (b, l))).collect();
    let mut totals: Vec<Totals> = modes.iter().map(|_| Totals::default()).collect();
    let warmup = 12;
    for round in 0..args.rounds {
        for (b, &(backend, layer)) in modes.iter().enumerate() {
            renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { backend, ..setup.grid.grid }, ..setup.grid });
            renderer.rt_grid_mut().unwrap().set_moving_layer(layer);
            let mut fx = effects(&renderer, &setup);
            for k in 0..warmup + args.frames {
                let measured = k >= warmup;
                if k == warmup {
                    let _ = renderer.take_profile();
                }
                let t0 = Instant::now();
                // every block replays the same frames
                let (eye, at) = (setup.camera)(k);
                camera.set_position(eye[0], eye[1], eye[2]);
                camera.look_at(&Vec3::new(at[0], at[1], at[2]));
                camera.update_projection_matrix();
                for (i, (idx, home)) in setup.beads.iter().enumerate() {
                    if let Some(r) = setup.scene.get_renderable_mut(*idx) {
                        let t = k as f32 * 0.03 + i as f32;
                        r.object.set_position(home[0] + 0.3 * t.sin(), home[1] + 0.2 * (1.3 * t).cos(), home[2] + 0.3 * (0.7 * t).cos());
                    }
                }
                renderer.render_scene_offscreen(&mut setup.scene, &mut camera, &gbuffer);
                let mut encoder = renderer.device().create_command_encoder(&Default::default());
                let mut input = &gbuffer.color_view;
                for (e, effect) in fx.iter_mut().enumerate() {
                    effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, input, &gbuffer.depth_view, &ping[e % 2], &camera, w, h);
                    input = &ping[e % 2];
                }
                renderer.queue().submit(Some(encoder.finish()));
                renderer.end_profiled_frame();
                let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
                if measured {
                    totals[b].wall.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                if std::env::var("RTB_FRAMES").is_ok() {
                    eprintln!("    frame {k}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let profile = renderer.take_profile();
            if std::env::var("RTB_REPORT").is_ok() {
                eprintln!("{}", profile.report());
            }
            let t = &mut totals[b];
            let stats = renderer.rt_grid().unwrap().stats().grid;
            let moving = if stats.moving_cells > 0 { stats.moving_triangles } else { 0 };
            (t.triangles, t.accel) = (stats.triangles + moving, stats.accel_triangles);
            for p in &profile.gpu {
                match t.passes.iter_mut().find(|(l, _, _)| *l == p.label) {
                    Some(e) => {
                        e.1 += p.busy_ms;
                        e.2 += 1;
                    }
                    None => t.passes.push((p.label, p.busy_ms, 1)),
                }
            }
            eprintln!("  block {round} {backend:?} (moving layer {layer}): wall {:.2} ms", median(&totals[b].wall[totals[b].wall.len() - args.frames..]));
        }
    }
    let info = renderer.adapter_info();
    println!("\n## {name}\n{} ({:?}), {w}x{h}, {} rounds of {} frames a backend", info.name, info.backend, args.rounds, args.frames);
    let header: Vec<String> = modes.iter().map(|(b, l)| if args.layers.len() > 1 { format!("{b:?}, {}", if *l { "moving layer" } else { "one layer" }) } else { format!("{b:?}") }).collect();
    println!("| | {} |", header.join(" | "));
    println!("|---|{}", "---|".repeat(modes.len()));
    println!("| grid triangles | {} |", totals.iter().map(|t| t.triangles.to_string()).collect::<Vec<_>>().join(" | "));
    println!("| frame, wall (median) | {} |", totals.iter().map(|t| format!("{:.2} ms", median(&t.wall))).collect::<Vec<_>>().join(" | "));
    println!("| BLAS triangles | {} |", totals.iter().map(|t| if t.accel > 0 { t.accel.to_string() } else { "—".into() }).collect::<Vec<_>>().join(" | "));
    // every pass the profiler timed, a frame (Apple GPUs overlap passes: more than the span)
    println!("| GPU, all passes | {} |", totals.iter().map(|t| format!("{:.2} ms", t.passes.iter().map(|(_, ms, n)| ms / *n as f64).sum::<f64>())).collect::<Vec<_>>().join(" | "));
    for label in PASSES {
        if totals.iter().all(|t| t.passes.iter().all(|(l, _, _)| *l != label)) {
            continue;
        }
        let cells: Vec<String> = totals
            .iter()
            .map(|t| t.passes.iter().find(|(l, _, _)| *l == label).map_or("—".into(), |(_, ms, n)| format!("{:.3} ms", ms / *n as f64)))
            .collect();
        println!("| {label} | {} |", cells.join(" | "));
    }
    // (no teardown: on Vulkan this run's device didn't finish dropping, whichever backend)
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
