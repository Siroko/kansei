//! Ray-traced shadows (`rt::RtShadowsEffect`) against shadow maps, headless, in gi-box's room
//! (its walls, blocks and rug, a chrome ball on the short block, the lamp under the ceiling with
//! its 15 cm disk) with a dense mesh on the tall block (`--dense <subdivisions>`: an icosphere of
//! 20 × 4^n triangles, 5 by default, about the light dragon's 19k):
//!
//! - `maps`: the lamp's 2048² shadow map, read by the materials with PCSS (gi-box's
//!   `shadows=maps`);
//! - `rt`: no shadow map; the materials leave the lamp to `RtShadowsEffect`, its rays through the
//!   room's grid (96³ cells of 4.8 cm) with the software backend, and with hardware ray queries
//!   where the device has them (gi-box's `shadows=rt`).
//!
//! Direct light only (no GI, no reflections), so the frames differ by the shadows alone. Each mode
//! renders in turn, blocks of frames alternating, and the profiler's pass times and the frame's
//! wall time (the CPU waiting for each frame's GPU work) are averaged per mode.
//!
//! ```text
//! cargo run --release -p kansei-native --example rt_shadows -- [--modes maps,software,hardware] [--dense N] [--size WxH] [--rounds N] [--frames N] [--full]
//! ```

use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtBackend, RtGridOptions, RtShadowsEffect, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

/// The passes reported, in frame order.
const PASSES: [&str; 13] = [
    "Renderer/SpotShadowPass",
    "Renderer/GBufferPass",
    "Renderer/GBufferOpaquePass",
    "Rt/Gather",
    "Rt/GridCount",
    "Rt/GridScan",
    "Rt/GridFill",
    "Rt/BlasInput",
    "RtShadows/Trace",
    "RtShadows/Temporal",
    "RtShadows/Atrous1",
    "RtShadows/Atrous2",
    "RtShadows/Composite",
];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Maps,
    Rt(RtBackend),
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Maps => "shadow map (PCSS)",
            Mode::Rt(RtBackend::Hardware) => "RT, hardware",
            Mode::Rt(_) => "RT, software grid",
        }
    }
}

struct Args {
    modes: Vec<Mode>,
    dense: u32,
    size: (u32, u32),
    rounds: usize,
    frames: usize,
    full: bool,
}

fn args() -> Args {
    let mut a = Args { modes: vec![Mode::Maps, Mode::Rt(RtBackend::Software), Mode::Rt(RtBackend::Hardware)], dense: 5, size: (1920, 1080), rounds: 4, frames: 32, full: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--modes" => {
                a.modes = it
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .filter_map(|m| match m {
                        "maps" => Some(Mode::Maps),
                        "software" => Some(Mode::Rt(RtBackend::Software)),
                        "hardware" => Some(Mode::Rt(RtBackend::Hardware)),
                        _ => None,
                    })
                    .collect()
            }
            "--dense" => a.dense = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.dense),
            "--rounds" => a.rounds = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.rounds),
            "--frames" => a.frames = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.frames),
            "--full" => a.full = true,
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

/// gi-box's surface: the lamp's light shadowed by the map (`rt` false), or left to
/// `RtShadowsEffect` (its GBuffer pixels marked by `kansei_gbuffer_out_rt_lit`).
fn material(color: [f32; 3], rt: bool) -> Material {
    let light = if rt {
        format!(
            "{RT_SHADOWS_GBUFFER_WGSL}\nfn out(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, pixel: vec2<f32>) -> KanseiGBufferOut {{\n    return kansei_gbuffer_out_rt_lit(vec3<f32>(0.0), vec3<f32>(0.0), n, base, 1.0, 0.04, false);\n}}\n"
        )
    } else {
        format!(
            "{SPOT_LIGHTS_WGSL}\nfn out(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, pixel: vec2<f32>) -> KanseiGBufferOut {{\n    return kansei_gbuffer_out(kansei_spot_lights_radiance(world, n, v, base, 1.0, 0.0, pixel), vec3<f32>(0.0), n, base);\n}}\n"
        )
    };
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
{light}
struct Surface {{ albedo: vec4<f32> }};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut {{ @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> }};
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {{
    let world = world_matrix * position;
    var o: VOut;
    o.clip = projection_matrix * view_matrix * world;
    o.world = world.xyz;
    o.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return o;
}}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {{
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    return out(in.world, n, normalize(eye - in.world), surface.albedo.rgb, in.clip.xy);
}}
"#
    );
    let mut m = Material::new("Surface", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "Surface", &[color[0], color[1], color[2], 1.0]);
    m
}

/// gi-box's room, its materials for `rt` or the maps.
fn room(dense: u32, rt: bool) -> Scene {
    let mut scene = Scene::new();
    let (white, red, green) = ([0.73; 3], [0.63, 0.065, 0.05], [0.14, 0.45, 0.09]);
    let mut add = |scene: &mut Scene, geometry, color: [f32; 3], at: [f32; 3], yaw: f32| {
        let mut r = Renderable::new(geometry, material(color, rt)).with_rt(RtSurface::new(color));
        r.object.set_position(at[0], at[1], at[2]);
        r.object.rotation.y = yaw;
        scene.add(SceneNode::Renderable(r));
    };
    add(&mut scene, BoxGeometry::new(4.4, 0.2, 4.2), white, [0.0, -0.1, -2.1], 0.0);
    add(&mut scene, BoxGeometry::new(4.4, 0.2, 4.2), white, [0.0, 4.1, -2.1], 0.0);
    add(&mut scene, BoxGeometry::new(4.4, 4.4, 0.2), white, [0.0, 2.0, -4.1], 0.0);
    add(&mut scene, BoxGeometry::new(0.2, 4.4, 4.2), red, [-2.1, 2.0, -2.1], 0.0);
    add(&mut scene, BoxGeometry::new(0.2, 4.4, 4.2), green, [2.1, 2.0, -2.1], 0.0);
    add(&mut scene, BoxGeometry::new(1.2, 2.4, 1.2), white, [-0.75, 1.2, -2.6], 0.33);
    add(&mut scene, BoxGeometry::new(1.2, 1.2, 1.2), white, [0.8, 0.6, -1.5], -0.3);
    add(&mut scene, BoxGeometry::new(2.4, 0.02, 0.8), [0.45, 0.3, 0.42], [0.0, 0.01, -0.55], 0.0);
    add(&mut scene, IcosphereGeometry::new(0.4, 4), [0.6; 3], [0.8, 1.6, -1.5], 0.0);
    // where the dragon stands: a dense mesh
    add(&mut scene, IcosphereGeometry::new(0.45, dense), [0.75, 0.6, 0.35], [-0.75, 2.85, -2.6], 0.0);
    let mut lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    lamp.cast_shadow = !rt;
    lamp.source_radius = 0.15;
    lamp.volumetric_scale = 0.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));
    scene
}

#[derive(Default)]
struct Totals {
    passes: Vec<(&'static str, f64, usize)>,
    wall: Vec<f64>,
    gpu: Vec<f64>,
    backend: Option<RtBackend>,
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
    renderer.enable_spot_shadows(2048, 1);
    let mut maps_scene = room(args.dense, false);
    let mut rt_scene = room(args.dense, true);
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
    let mut camera = Camera::new(38.0, 0.1, 100.0, w as f32 / h as f32);
    camera.set_position(0.0, 2.0, 7.4);
    camera.look_at(&Vec3::new(0.0, 1.9, -2.0));
    camera.update_projection_matrix();
    let grid = SceneRtGridOptions { grid: RtGridOptions { dims: [96, 96, 96], cell: 4.6 / 96.0, fixed_origin: Some(glam::Vec3::new(-2.3, -0.3, -4.3)), ..Default::default() }, ..Default::default() };
    renderer.set_profiling(true);
    let mut totals: Vec<Totals> = args.modes.iter().map(|_| Totals::default()).collect();
    let warmup = 16;
    for round in 0..args.rounds {
        for (m, &mode) in args.modes.iter().enumerate() {
            let mut effect = match mode {
                Mode::Maps => {
                    renderer.disable_rt_grid();
                    None
                }
                Mode::Rt(backend) => {
                    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { backend, ..grid.grid }, ..grid });
                    let mut e = RtShadowsEffect::new(renderer.rt_grid().unwrap().handle());
                    e.half_resolution = !args.full;
                    e.update_lights(rt_scene.lights());
                    totals[m].backend = Some(renderer.rt_grid().unwrap().handle().backend());
                    Some(e)
                }
            };
            let scene = if effect.is_some() { &mut rt_scene } else { &mut maps_scene };
            for k in 0..warmup + args.frames {
                if k == warmup {
                    let _ = renderer.take_profile();
                }
                let t0 = Instant::now();
                renderer.render_scene_offscreen(scene, &mut camera, &gbuffer);
                if let Some(e) = &mut effect {
                    let mut encoder = renderer.device().create_command_encoder(&Default::default());
                    e.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output, &camera, w, h);
                    renderer.queue().submit(Some(encoder.finish()));
                }
                renderer.end_profiled_frame();
                let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
                if k >= warmup {
                    totals[m].wall.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let profile = renderer.take_profile();
            if std::env::var("RTS_REPORT").is_ok() {
                eprintln!("{}", profile.report());
            }
            let t = &mut totals[m];
            t.gpu.push(profile.gpu_ms);
            for p in &profile.gpu {
                match t.passes.iter_mut().find(|(l, _, _)| *l == p.label) {
                    Some(e) => {
                        e.1 += p.busy_ms;
                        e.2 += 1;
                    }
                    None => t.passes.push((p.label, p.busy_ms, 1)),
                }
            }
            eprintln!("  block {round} {}: wall {:.2} ms", mode.name(), median(&t.wall[t.wall.len() - args.frames..]));
        }
    }
    println!(
        "\n## gi-box's room, direct light only (dense mesh of {} triangles)\n{} ({:?}), {w}x{h}, {} rounds of {} frames a mode, RT traced at {} resolution",
        20 * 4u64.pow(args.dense),
        info.name,
        info.backend,
        args.rounds,
        args.frames,
        if args.full { "full" } else { "half" }
    );
    let header: Vec<String> = args.modes.iter().zip(&totals).map(|(m, t)| match (m, t.backend) {
        (Mode::Rt(RtBackend::Hardware), Some(RtBackend::Software)) => "RT, hardware (none: traced in software)".to_string(),
        _ => m.name().to_string(),
    }).collect();
    println!("| | {} |", header.join(" | "));
    println!("|---|{}", "---|".repeat(args.modes.len()));
    println!("| frame, wall (median) | {} |", totals.iter().map(|t| format!("{:.2} ms", median(&t.wall))).collect::<Vec<_>>().join(" | "));
    println!("| GPU passes, sum (median) | {} |", totals.iter().map(|t| format!("{:.3} ms", median(&t.gpu))).collect::<Vec<_>>().join(" | "));
    let shadow_sum = |t: &Totals| -> f64 {
        t.passes.iter().filter(|(l, _, _)| l.starts_with("RtShadows/") || *l == "Renderer/SpotShadowPass").map(|(_, ms, n)| ms / *n as f64).sum()
    };
    println!("| shadow passes (map, or RtShadows/*) | {} |", totals.iter().map(|t| format!("{:.3} ms", shadow_sum(t))).collect::<Vec<_>>().join(" | "));
    for label in PASSES {
        if totals.iter().all(|t| t.passes.iter().all(|(l, _, _)| *l != label)) {
            continue;
        }
        let cells: Vec<String> = totals.iter().map(|t| t.passes.iter().find(|(l, _, _)| *l == label).map_or("—".into(), |(_, ms, n)| format!("{:.3} ms", ms / *n as f64))).collect();
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
