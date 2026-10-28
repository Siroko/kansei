//! Glass caustics (`rt::RtCausticsEffect`) on `RtShadowsEffect`'s direct light, headless, at a
//! marble's scale: a 3 cm glass marble with a red glass core on a table under two small spot
//! lights (2 cm emitters), seen from 35 cm, as a marble game shows it. Each mode renders in turn,
//! blocks of frames alternating, and the profiler's pass times and the frame's wall time (the CPU
//! waiting for each frame's GPU work) are averaged per mode:
//!
//! - `shadows`: `RtShadowsEffect` alone (the glass casts no shadow: its rays pass through it);
//! - `still`: with the caustics, the marble at rest (its tile accumulated, no photons sent);
//! - `rolling`: with the caustics, the marble rolling across the table (its tile cleared and fed
//!   every frame, the grid rebuilt every frame for it in every mode that rolls).
//!
//! Through the grid's cells (`software`) and hardware ray queries (`hardware`, where the device
//! has them). The grid: `--grid fine` (the default), 1 cm cells over the table alone; `coarse`,
//! 2.5 cm cells over 2.4 m round it (the marble's triangles a cell or two, every ray through it
//! testing them all).
//!
//! ```text
//! cargo run --release -p kansei-native --example rt_caustics -- [--backends software,hardware] [--grid fine|coarse] [--size WxH] [--rounds N] [--frames N]
//! ```

use std::time::Instant;

use glam::Vec3 as GVec3;
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, MaterialOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtBackend, RtCausticsEffect, RtCausticsOptions, RtGridOptions, RtShadowsEffect, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

/// The passes reported, in frame order.
const PASSES: [&str; 12] = [
    "Renderer/GBufferPass",
    "Rt/Gather",
    "Rt/GridCount",
    "Rt/GridScan",
    "Rt/GridFill",
    "Rt/BlasInput",
    "RtShadows/Trace",
    "RtShadows/Temporal",
    "RtShadows/Composite",
    "RtCaustics/Photons",
    "RtCaustics/Resolve",
    "RtCaustics/Composite",
];

const RADIUS: f32 = 0.015;
const TABLE: f32 = 0.75;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Shadows,
    Still,
    Rolling,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Shadows => "RT shadows alone",
            Mode::Still => "+ caustics, still",
            Mode::Rolling => "+ caustics, rolling",
        }
    }
}

struct Args {
    backends: Vec<RtBackend>,
    size: (u32, u32),
    rounds: usize,
    frames: usize,
    fine: bool,
}

fn args() -> Args {
    let mut a = Args { backends: vec![RtBackend::Software, RtBackend::Hardware], size: (1920, 1080), rounds: 3, frames: 48, fine: true };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--backends" => {
                a.backends = it
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter_map(|m| match m {
                        "software" => Some(RtBackend::Software),
                        "hardware" => Some(RtBackend::Hardware),
                        _ => None,
                    })
                    .collect()
            }
            "--grid" => a.fine = it.next().as_deref() != Some("coarse"),
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

const VERTEX: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
"#;

/// A surface `RtShadowsEffect` lights.
fn lit(color: [f32; 3]) -> Material {
    let [r, g, b] = color;
    let code = format!(
        "{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return kansei_gbuffer_out_rt_lit(vec3<f32>(0.0), vec3<f32>(0.0), normalize(in.normal), vec3<f32>({r:?}, {g:?}, {b:?}), 0.7, 0.04, false); }}\n"
    );
    Material::new("Lit", &code, Vec::new(), MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
}

fn glass(tint: [f32; 3]) -> Material {
    let [r, g, b] = tint;
    let code = format!("{GBUFFER_OUT_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return kansei_gbuffer_out_glass(normalize(in.normal), vec3<f32>({r:?}, {g:?}, {b:?}), 1.5, 0.0); }}\n");
    Material::new("Glass", &code, Vec::new(), MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
}

/// The table, the floor, a few props, the marble (and its core) and the lamps; the marble's and
/// its core's scene indices.
fn scene() -> (Scene, usize, usize) {
    let mut scene = Scene::new();
    let mut add = |scene: &mut Scene, geometry, color: [f32; 3], at: [f32; 3]| {
        let mut r = Renderable::new(geometry, lit(color)).with_rt(RtSurface::new(color));
        r.object.set_position(at[0], at[1], at[2]);
        scene.add(SceneNode::Renderable(r));
    };
    add(&mut scene, BoxGeometry::new(4.0, 0.1, 4.0), [0.5; 3], [0.0, -0.05, 0.0]);
    add(&mut scene, BoxGeometry::new(1.2, 0.04, 0.7), [0.62, 0.45, 0.3], [0.0, TABLE - 0.02, 0.0]);
    add(&mut scene, BoxGeometry::new(0.06, 0.04, 0.06), [0.8; 3], [0.12, TABLE + 0.02, -0.06]);
    add(&mut scene, IcosphereGeometry::new(0.02, 3), [0.2, 0.3, 0.7], [-0.1, TABLE + 0.02, -0.08]);
    // (1280 triangles with smooth normals: a marble's mesh needs no more)
    let mut marble = Renderable::new(IcosphereGeometry::new(RADIUS, 3), glass([0.97; 3])).with_rt(RtSurface::glass([0.97; 3]));
    marble.object.set_position(0.0, TABLE + RADIUS, 0.0);
    let marble = scene.add(SceneNode::Renderable(marble));
    // the core: a red glass spindle (a sphere stretched along y)
    let mut core = Renderable::new(IcosphereGeometry::new(1.0, 2), glass([0.9, 0.08, 0.05])).with_rt(RtSurface::glass([0.9, 0.08, 0.05]));
    core.object.set_position(0.0, TABLE + RADIUS, 0.0);
    core.object.scale = Vec3::new(0.3 * RADIUS, 0.8 * RADIUS, 0.3 * RADIUS);
    let core = scene.add(SceneNode::Renderable(core));
    for (at, color) in [([0.12, TABLE + 0.6, 0.05], [1.0, 0.92, 0.8]), ([-0.2, TABLE + 0.5, -0.1], [0.8, 0.9, 1.0])] {
        let mut lamp = SpotLight::new(Vec3::new(at[0], at[1], at[2]), Vec3::new(-at[0], -(at[1] - TABLE), -at[2]), Vec3::new(color[0], color[1], color[2]), 40.0, 4.0, 25f32.to_radians(), 35f32.to_radians());
        lamp.source_radius = 0.01;
        lamp.volumetric_scale = 0.0;
        scene.add(SceneNode::Light(Light::Spot(lamp)));
    }
    (scene, marble, core)
}

#[derive(Default)]
struct Totals {
    passes: Vec<(&'static str, f64, usize)>,
    wall: Vec<f64>,
    gpu: Vec<f64>,
    photons: u32,
    tile: Option<([u32; 2], f32)>,
}

fn main() {
    env_logger::init();
    let args = args();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, driver {} {}); hardware ray queries: {}", info.name, info.backend, info.driver, info.driver_info, adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY));
    let (w, h) = args.size;
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    let (mut scene, marble, core) = scene();
    let gbuffer = GBuffer::new(renderer.device(), w, h, 1);
    let texture = |label| {
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
    let (lit_view, output) = (texture("Bench/Lit"), texture("Bench/Out"));
    let mut camera = Camera::new(38.0, 0.01, 20.0, w as f32 / h as f32);
    camera.set_position(0.0, TABLE + 0.18, 0.3);
    camera.look_at(&Vec3::new(0.0, TABLE, 0.0));
    camera.update_projection_matrix();
    // 1 cm cells over the table (1.28 x 0.64 x 0.96 m), or 2.5 cm over 2.4 m round it
    let grid = if args.fine {
        SceneRtGridOptions { grid: RtGridOptions { dims: [128, 64, 96], cell: 0.01, fixed_origin: Some(glam::Vec3::new(-0.64, TABLE - 0.12, -0.48)), ..Default::default() }, ..Default::default() }
    } else {
        SceneRtGridOptions { grid: RtGridOptions { dims: [96, 64, 96], cell: 0.025, fixed_origin: Some(glam::Vec3::new(-1.2, -0.1, -1.2)), ..Default::default() }, ..Default::default() }
    };
    renderer.set_profiling(true);
    let modes = [Mode::Shadows, Mode::Still, Mode::Rolling];
    let mut totals: Vec<Vec<Totals>> = args.backends.iter().map(|_| modes.iter().map(|_| Totals::default()).collect()).collect();
    let mut traced = vec![None; args.backends.len()];
    let mut time = 0.0f32;
    for round in 0..args.rounds {
        for (b, &backend) in args.backends.iter().enumerate() {
            renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { backend, ..grid.grid }, ..grid });
            let handle = renderer.rt_grid().unwrap().handle();
            traced[b] = Some(handle.backend());
            for (m, &mode) in modes.iter().enumerate() {
                let mut shadows = RtShadowsEffect::new(handle.clone());
                shadows.update_lights(scene.lights());
                let mut caustics = (mode != Mode::Shadows).then(|| {
                    let mut c = RtCausticsEffect::new(renderer.device(), handle.clone(), RtCausticsOptions::default());
                    c.tint_distance = 0.01;
                    c.update_lights(scene.lights());
                    c
                });
                let rolling = mode == Mode::Rolling;
                for i in [marble, core] {
                    let r = scene.get_renderable_mut(i).unwrap();
                    r.dynamic = rolling;
                    r.object.position.x = 0.0;
                }
                // still: past the tile's accumulation before timing
                let warmup = if mode == Mode::Still { 96 } else { 16 };
                for k in 0..warmup + args.frames {
                    if k == warmup {
                        let _ = renderer.take_profile();
                    }
                    if rolling {
                        time += 1.0 / 60.0;
                        let x = 0.08 * (time * 1.5).sin();
                        for i in [marble, core] {
                            let r = scene.get_renderable_mut(i).unwrap();
                            r.object.position.x = x;
                            r.object.rotation.z = -x / RADIUS;
                        }
                    }
                    let x = scene.get_renderable(marble).unwrap().object.position.x;
                    let t0 = Instant::now();
                    renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
                    let mut encoder = renderer.device().create_command_encoder(&Default::default());
                    let out = if caustics.is_some() { &lit_view } else { &output };
                    shadows.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, out, &camera, w, h);
                    if let Some(c) = &mut caustics {
                        c.set_casters(&[(GVec3::new(x - RADIUS, TABLE, -RADIUS), GVec3::new(x + RADIUS, TABLE + 2.0 * RADIUS, RADIUS))]);
                        c.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &lit_view, &gbuffer.depth_view, &output, &camera, w, h);
                    }
                    renderer.queue().submit(Some(encoder.finish()));
                    renderer.end_profiled_frame();
                    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
                    if k >= warmup {
                        totals[b][m].wall.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                let profile = renderer.take_profile();
                let t = &mut totals[b][m];
                t.gpu.push(profile.gpu_ms);
                if let Some(c) = &caustics {
                    t.photons = c.photons_sent().1;
                    t.tile = c.tile(0).map(|(_, texel, dims, _)| (dims, texel));
                }
                for p in &profile.gpu {
                    match t.passes.iter_mut().find(|(l, _, _)| *l == p.label) {
                        Some(e) => {
                            e.1 += p.busy_ms;
                            e.2 += 1;
                        }
                        None => t.passes.push((p.label, p.busy_ms, 1)),
                    }
                }
                eprintln!("  block {round} {backend:?} {}: wall {:.2} ms", mode.name(), median(&t.wall[t.wall.len() - args.frames..]));
            }
        }
    }
    println!(
        "\n## A 3 cm glass marble on a table under two spot lights\n{} ({:?}), {w}x{h}, {} rounds of {} frames a mode, RT shadows at half resolution, {}",
        info.name,
        info.backend,
        args.rounds,
        args.frames,
        if args.fine { "1 cm cells over the table" } else { "2.5 cm cells over 2.4 m" }
    );
    for (b, backend) in args.backends.iter().enumerate() {
        let name = match (backend, traced[b]) {
            (RtBackend::Hardware, Some(RtBackend::Software)) => "hardware asked (none: traced in software)".to_string(),
            (_, Some(t)) => format!("{t:?}"),
            _ => format!("{backend:?}"),
        };
        println!("\n### {name}\n");
        let t = &totals[b];
        println!("| | {} |", modes.iter().map(|m| m.name()).collect::<Vec<_>>().join(" | "));
        println!("|---|{}", "---|".repeat(modes.len()));
        println!("| frame, wall (median) | {} |", t.iter().map(|t| format!("{:.2} ms", median(&t.wall))).collect::<Vec<_>>().join(" | "));
        println!("| GPU passes, sum (median) | {} |", t.iter().map(|t| format!("{:.3} ms", median(&t.gpu))).collect::<Vec<_>>().join(" | "));
        let sum = |t: &Totals, prefix: &str| -> f64 { t.passes.iter().filter(|(l, _, _)| l.starts_with(prefix)).map(|(_, ms, n)| ms / *n as f64).sum() };
        println!("| RtCaustics/*, sum | {} |", t.iter().map(|t| format!("{:.3} ms", sum(t, "RtCaustics/"))).collect::<Vec<_>>().join(" | "));
        println!("| photons a frame, tile | {} |", t.iter().map(|t| t.tile.map_or("—".into(), |(d, texel)| format!("{}, {}x{} of {:.1} mm", t.photons, d[0], d[1], texel * 1e3))).collect::<Vec<_>>().join(" | "));
        for label in PASSES {
            if t.iter().all(|t| t.passes.iter().all(|(l, _, _)| *l != label)) {
                continue;
            }
            let cells: Vec<String> = t.iter().map(|t| t.passes.iter().find(|(l, _, _)| *l == label).map_or("—".into(), |(_, ms, n)| format!("{:.3} ms", ms / *n as f64))).collect();
            println!("| {label} | {} |", cells.join(" | "));
        }
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
