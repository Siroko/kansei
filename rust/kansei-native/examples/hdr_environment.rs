//! What an HDR environment (`environment::HdrEnvironment`) costs, headless: decoding the `.hdr`,
//! uploading it, preparing it on the GPU (the radiance cubemap and its mips, the GGX-prefiltered
//! levels, the SH; done at load and again on each change of its settings), and per frame, gi-box's
//! room (hybrid diffuse GI, the chrome and glass balls' reflections and glass) lit by the
//! environment against the same room under its SH alone with no background, blocks of frames
//! alternating so other work on the GPU weighs on both alike.
//!
//! ```text
//! cargo run --release -p kansei-native --example hdr_environment -- [path.hdr] [--size WxH] [--rounds N] [--frames N] [--face N]
//! ```
//! The default image is gi-box's `www/assets/studio_small_09_1k.hdr`.

use std::time::Instant;

use kansei_core::cameras::Camera;
use kansei_core::environment::{EnvironmentLighting, HdrEnvironment, HdrEnvironmentOptions, HdrEnvironmentSettings, HdrImage};
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry};
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::EnvironmentBackgroundEffect;
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGlass, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions, RtSurface, SceneRtGridOptions};

const PASSES: [&str; 8] = ["Environment/Cube", "Environment/Prefilter", "Environment/SH", "Environment/Background", "RtGi/Trace", "Rt/Trace", "Rt/Glass", "Rt/Resolve"];

struct Args {
    path: String,
    size: (u32, u32),
    rounds: usize,
    frames: usize,
    face: u32,
}

fn args() -> Args {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../kansei-wasm/examples/gi-box/www/assets/studio_small_09_1k.hdr");
    let mut a = Args { path: default.into(), size: (1920, 1080), rounds: 4, frames: 24, face: 0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--rounds" => a.rounds = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.rounds),
            "--frames" => a.frames = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.frames),
            "--face" => a.face = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.face),
            "--size" => {
                if let Some((w, h)) = it.next().and_then(|v| v.split_once('x').map(|(w, h)| (w.parse().unwrap_or(1920), h.parse().unwrap_or(1080)))) {
                    a.size = (w, h);
                }
            }
            path => a.path = path.into(),
        }
    }
    a
}

fn block(scene: &mut Scene, size: [f32; 3], at: [f32; 3], color: [f32; 3]) {
    let material = Material::standard_lit("Lit", &StandardLitOptions { base_color: color, roughness: 0.6, ..Default::default() });
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material).with_gi(GiSurface::new(color)).with_rt(RtSurface::new(color));
    r.object.set_position(at[0], at[1], at[2]);
    scene.add(SceneNode::Renderable(r));
}

/// gi-box's room: 4 m each way, open toward the camera, a chrome ball on the short block and a
/// glass one on the floor, under one spot light.
fn room(renderer: &mut Renderer) -> Scene {
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-2.3, -0.3, -4.3], bounds_max: [2.3, 4.3, 0.3], ..Default::default() });
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [96, 96, 96], cell: 4.6 / 96.0, fixed_origin: Some(glam::Vec3::new(-2.3, -0.3, -4.3)), ..Default::default() }, ..Default::default() });
    let mut scene = Scene::new();
    let (white, red, green) = ([0.73; 3], [0.63, 0.065, 0.05], [0.14, 0.45, 0.09]);
    block(&mut scene, [4.4, 0.2, 4.2], [0.0, -0.1, -2.0], white);
    block(&mut scene, [4.4, 0.2, 4.2], [0.0, 4.1, -2.0], white);
    block(&mut scene, [4.4, 4.0, 0.2], [0.0, 2.0, -4.1], white);
    block(&mut scene, [0.2, 4.0, 4.2], [-2.1, 2.0, -2.0], red);
    block(&mut scene, [0.2, 4.0, 4.2], [2.1, 2.0, -2.0], green);
    block(&mut scene, [1.2, 2.4, 1.2], [-0.8, 1.2, -2.6], white);
    block(&mut scene, [1.2, 1.2, 1.2], [0.8, 0.6, -1.5], white);
    let chrome_material = Material::standard_lit("Chrome", &StandardLitOptions::mirror([0.9; 3], 0.05));
    let mut chrome = Renderable::new(IcosphereGeometry::new(0.3, 4), chrome_material).with_gi(GiSurface::new([0.5; 3])).with_rt(RtSurface::new([0.9; 3]).with_smooth_normals().with_specular(0.9));
    chrome.object.set_position(0.8, 1.5, -1.5);
    scene.add(SceneNode::Renderable(chrome));
    let tint = [0.92, 0.96, 1.0];
    let mut glass = Renderable::new(IcosphereGeometry::new(0.3, 4), Material::standard_lit("Glass", &StandardLitOptions::glass(tint, 1.5, 0.0))).with_rt(RtSurface::glass(tint));
    glass.cast_shadow = false;
    glass.object.set_position(-0.1, 0.33, -0.45);
    scene.add(SceneNode::Renderable(glass));
    let lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    scene.add(SceneNode::Light(Light::Spot(lamp)));
    scene
}

/// The room's effects: lit by the environment (its background, its cubemap for the misses), or by
/// its SH alone with no background.
fn effects(renderer: &Renderer, scene: &Scene, lighting: &EnvironmentLighting, with_map: bool) -> Vec<Box<dyn PostProcessingEffect>> {
        let volume = renderer.voxel_gi().unwrap().volume();
    let grid = renderer.rt_grid().unwrap().handle();
    let mut gi = RtDiffuseGiEffect::with_volume(volume, grid.clone(), RtDiffuseGiOptions { max_distance: 20.0, ..Default::default() });
    gi.update_lights(scene.lights());
    gi.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    let mut reflections = RtReflectionsEffect::with_volume(volume, grid, RtReflectionsOptions { max_distance: 20.0, ..Default::default() });
    reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
    reflections.set_glass(Some(RtGlass::default()));
    let mut out: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if with_map {
        out.push(Box::new(EnvironmentBackgroundEffect::new(lighting)));
        gi.set_environment(lighting);
        reflections.set_environment(lighting);
    } else {
        gi.set_sky_lighting(Some(&lighting.sky_lighting));
        reflections.set_sky_lighting(Some(&lighting.sky_lighting));
    }
    out.push(Box::new(gi));
    out.push(Box::new(reflections));
    out
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// The passes' mean busy ms, by label.
#[derive(Default)]
struct Totals {
    passes: Vec<(&'static str, f64, usize)>,
    wall: Vec<f64>,
}

impl Totals {
    fn add(&mut self, profile: &kansei_core::profiling::FrameProfile) {
        for p in &profile.gpu {
            match self.passes.iter_mut().find(|(l, _, _)| *l == p.label) {
                Some(e) => {
                    e.1 += p.busy_ms;
                    e.2 += 1;
                }
                None => self.passes.push((p.label, p.busy_ms, 1)),
            }
        }
    }

    fn pass(&self, label: &str) -> String {
        self.passes.iter().find(|(l, _, _)| *l == label).map_or("—".into(), |(_, ms, n)| format!("{:.3} ms", ms / *n as f64))
    }
}

fn main() {
    env_logger::init();
    let args = args();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?}, driver {} {})", info.name, info.backend, info.driver, info.driver_info);
    let (w, h) = args.size;
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    let (device, queue) = (renderer.device().clone(), renderer.queue().clone());

    // load: the file's bytes decoded, then the upload and the pipelines
    let bytes = std::fs::read(&args.path).unwrap_or_else(|e| panic!("{}: {e}", args.path));
    let decodes: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            let _ = HdrImage::decode(&bytes).expect("decode");
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    let image = HdrImage::decode(&bytes).unwrap();
    let t = Instant::now();
    let mut env = HdrEnvironment::new(&device, &queue, &image, HdrEnvironmentOptions { face_size: args.face, ..Default::default() });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let upload = t.elapsed().as_secs_f64() * 1e3;
    // the first prepare (the pipelines' first use), on its own
    let t = Instant::now();
    env.prepare(&device, &queue);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let first = t.elapsed().as_secs_f64() * 1e3;
    // prepared again for each turn: the profiler's pass times and the wall time to the GPU's end
    renderer.set_profiling(true);
    let mut prepare = Totals::default();
    for k in 0..args.frames.max(8) {
        env.set_settings(HdrEnvironmentSettings { rotation: 0.1 * (k + 1) as f32, ..Default::default() });
        let t = Instant::now();
        env.prepare(&device, &queue);
        renderer.end_profiled_frame();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        prepare.wall.push(t.elapsed().as_secs_f64() * 1e3);
    }
    // (readbacks arrive late: a few empty frames to let them in)
    for _ in 0..4 {
        renderer.end_profiled_frame();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
    prepare.add(&renderer.take_profile());
    env.set_settings(HdrEnvironmentSettings::default());
    env.prepare(&device, &queue);
    let (fw, fh) = (image.width, image.height);
    println!("\n## Loading and preparing {} ({fw} x {fh}, {:.1} MB file)", args.path.rsplit('/').next().unwrap_or(&args.path), bytes.len() as f64 / 1e6);
    println!("| | |\n|---|---|");
    println!("| decode (CPU, median of 5) | {:.1} ms |", median(&decodes));
    println!("| upload + pipelines | {upload:.1} ms |");
    println!("| first prepare, wall | {first:.1} ms |");
    println!("| prepare again (a turn), wall median | {:.2} ms |", median(&prepare.wall));
    for label in &PASSES[..3] {
        println!("| {label} | {} |", prepare.pass(label));
    }
    println!("| cube face | {} texels, {} mips; {:.1} MiB on the GPU |", env.face_size(), env.environment_texture().mip_level_count(), env.memory_bytes() as f64 / (1 << 20) as f64);

    // per frame: the room lit by the environment (background, cubemap misses) against its SH alone
    let mut scene = room(&mut renderer);
    let gbuffer = GBuffer::new(&device, w, h, 1);
    let target = |label| {
        device
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
    camera.set_position(0.0, 2.0, 7.0);
    camera.look_at(&Vec3::new(0.0, 1.6, -2.0));
    camera.update_projection_matrix();
    let lighting = env.lighting();
    renderer.set_environment(Some(&lighting));
    let modes = [("SH only, no background", false), ("environment: background + cubemap misses", true)];
    let mut totals: Vec<Totals> = modes.iter().map(|_| Totals::default()).collect();
    let warmup = 12;
    for round in 0..args.rounds {
        for (m, &(_, with_map)) in modes.iter().enumerate() {
            let mut fx = effects(&renderer, &scene, &lighting, with_map);
            for k in 0..warmup + args.frames {
                if k == warmup {
                    let _ = renderer.take_profile();
                }
                let t0 = Instant::now();
                renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
                let mut encoder = device.create_command_encoder(&Default::default());
                let mut input = &gbuffer.color_view;
                for (e, effect) in fx.iter_mut().enumerate() {
                    effect.render(&device, &queue, &mut encoder, &gbuffer, input, &gbuffer.depth_view, &ping[e % 2], &camera, w, h);
                    input = &ping[e % 2];
                }
                queue.submit(Some(encoder.finish()));
                renderer.end_profiled_frame();
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                if k >= warmup {
                    totals[m].wall.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            totals[m].add(&renderer.take_profile());
            eprintln!("  block {round} {}: wall {:.2} ms", modes[m].0, median(&totals[m].wall[totals[m].wall.len() - args.frames..]));
        }
    }
    println!("\n## Per frame: gi-box's room, {w}x{h}, {} rounds of {} frames a mode", args.rounds, args.frames);
    println!("| | {} |", modes.iter().map(|m| m.0).collect::<Vec<_>>().join(" | "));
    println!("|---|---|---|");
    println!("| frame, wall (median) | {} |", totals.iter().map(|t| format!("{:.2} ms", median(&t.wall))).collect::<Vec<_>>().join(" | "));
    for label in &PASSES[3..] {
        println!("| {label} | {} |", totals.iter().map(|t| t.pass(label)).collect::<Vec<_>>().join(" | "));
    }
    // (no teardown: on Vulkan a device may not finish dropping)
    std::mem::forget(renderer);
    std::process::exit(0);
}
