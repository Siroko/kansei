//! GPU memory over scene reloads, on a real GPU (a game loading level after level on one
//! renderer): each lap builds a new scene of new meshes (ray traced, voxelized with the distance
//! field, casting spot shadows, one of them moving), renders a few frames and drops it. What the
//! GPU allocator holds after each lap (`Device::generate_allocator_report`, where the backend has
//! one: Vulkan, DX12) must stop growing once the renderer's own buffers have grown to the scenes;
//! the labels that grew are printed. Skipped (passes) without an adapter or an allocator report.
//! `cargo test --release -p kansei-core --test scene_reload_memory_gpu -- --ignored --nocapture`.

use std::collections::BTreeMap;

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry};
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtGridOptions, RtSurface, SceneRtGridOptions};

/// The Rust heap's live bytes (this test's allocator counts them).
struct Counting;
static LIVE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as i64, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        LIVE.fetch_sub(layout.size() as i64, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size as i64 - layout.size() as i64, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

const W: u32 = 128;
const H: u32 = 96;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    eprintln!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// Lap `lap`'s scene: 60 blocks and 8 dense meshes, new meshes of about the same sizes each lap, one moving.
fn scene(lap: u32) -> (Scene, usize) {
    let lit = |c: [f32; 3]| Material::standard_lit("Lit", &StandardLitOptions { base_color: c, roughness: 0.6, ..Default::default() });
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), lit([0.6; 3])).with_gi(GiSurface::new([0.6; 3])).with_rt(RtSurface::new([0.6; 3]));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    for k in 0..60 {
        let s = 0.1 + 0.002 * (k as f32) + 0.0001 * (lap % 7) as f32;
        let mut b = Renderable::new(BoxGeometry::new(s, s * 1.5, s), lit([0.7, 0.5, 0.3])).with_gi(GiSurface::new([0.7, 0.5, 0.3])).with_rt(RtSurface::new([0.7, 0.5, 0.3]));
        b.object.set_position(-2.5 + 0.6 * (k % 10) as f32, s * 0.75, -2.5 + 0.8 * (k / 10) as f32);
        scene.add(SceneNode::Renderable(b));
    }
    // dense meshes (a level's models): 8 of 20k triangles
    for k in 0..8 {
        let mut d = Renderable::new(IcosphereGeometry::new(0.25 + 0.0001 * (lap % 7) as f32, 5), lit([0.5; 3])).with_gi(GiSurface::new([0.5; 3])).with_rt(RtSurface::new([0.5; 3]));
        d.object.set_position(-2.0 + 0.55 * k as f32, 1.2, 1.5);
        scene.add(SceneNode::Renderable(d));
    }
    let mut mover = Renderable::new(BoxGeometry::new(0.2, 0.2, 0.2), lit([0.9; 3])).with_gi(GiSurface::new([0.9; 3])).with_rt(RtSurface::new([0.9; 3]));
    mover.dynamic = true;
    let mover = scene.add(SceneNode::Renderable(mover));
    for at in [[-1.5, 3.0, 0.0], [1.5, 3.0, 0.0]] {
        let mut lamp = SpotLight::new(Vec3::new(at[0], at[1], at[2]), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 10.0, 20.0, 50f32.to_radians(), 60f32.to_radians());
        lamp.cast_shadow = true;
        scene.add(SceneNode::Light(Light::Spot(lamp)));
    }
    (scene, mover)
}

/// The process's private bytes (Windows; 0 elsewhere).
fn private_bytes() -> u64 {
    if !cfg!(windows) {
        return 0;
    }
    let out = std::process::Command::new("powershell").args(["-NoProfile", "-Command", &format!("(Get-Process -Id {}).PrivateMemorySize64", std::process::id())]).output();
    out.ok().and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok()).unwrap_or(0)
}

/// The allocator's bytes by label (allocations), and in all.
fn report(renderer: &Renderer) -> Option<(u64, BTreeMap<String, u64>)> {
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
    let r = renderer.device().generate_allocator_report()?;
    let mut by = BTreeMap::new();
    for a in &r.allocations {
        *by.entry(a.name.clone()).or_insert(0) += a.size;
    }
    Some((r.total_allocated_bytes, by))
}

#[test]
#[ignore]
fn reloading_scenes_holds_no_more_memory() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipped") };
    // (`RELOAD_FEATURES`: which of spot, rt, gi the renderer has; all by default)
    let features = std::env::var("RELOAD_FEATURES").unwrap_or_else(|_| "spot,rt,gi".into());
    if features.contains("spot") {
        renderer.enable_spot_shadows(512, 2);
    }
    if features.contains("rt") {
        renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 6.0 / 64.0, fixed_origin: Some(glam::Vec3::new(-3.0, -0.3, -3.0)), ..Default::default() }, ..Default::default() });
    }
    if features.contains("gi") {
        renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-3.0, -0.3, -3.0], bounds_max: [3.0, 2.7, 3.0], ..Default::default() });
        renderer.voxel_gi_mut().unwrap().enable_sdf();
    }
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(0.0, 3.0, 5.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    let mut laps = Vec::new();
    // (`RELOAD_LAPS` laps, 16 by default)
    let laps_wanted: u32 = std::env::var("RELOAD_LAPS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    for lap in 0..laps_wanted {
        {
            let (mut scene, mover) = scene(lap);
            for f in 0..6 {
                scene.get_renderable_mut(mover).unwrap().object.set_position(0.1 * f as f32, 0.5, 0.0);
                renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            }
        }
        let Some(r) = report(&renderer) else { return eprintln!("no allocator report on this backend: skipped") };
        let h = renderer.device().get_internal_counters().hal;
        eprintln!(
            "lap {lap}: allocator {:.1} MiB, process private {:.1} MiB, Rust heap {:.1} MiB; live buffers {} ({:.1} MiB), textures {} ({:.1} MiB), views {}, bind groups {}, layouts {}, render pipelines {}, compute pipelines {}, shader modules {}, samplers {}, query sets {}",
            r.0 as f64 / (1 << 20) as f64,
            private_bytes() as f64 / (1 << 20) as f64,
            LIVE.load(std::sync::atomic::Ordering::Relaxed) as f64 / (1 << 20) as f64,
            h.buffers.read(),
            h.buffer_memory.read() as f64 / (1 << 20) as f64,
            h.textures.read(),
            h.texture_memory.read() as f64 / (1 << 20) as f64,
            h.texture_views.read(),
            h.bind_groups.read(),
            h.bind_group_layouts.read(),
            h.render_pipelines.read(),
            h.compute_pipelines.read(),
            h.shader_modules.read(),
            h.samplers.read(),
            h.query_sets.read()
        );
        laps.push(r);
    }
    let (first, last) = (&laps[4], &laps[laps.len() - 1]);
    let mut grew: Vec<(String, i64)> = last.1.iter().map(|(k, v)| (k.clone(), *v as i64 - *first.1.get(k).unwrap_or(&0) as i64)).filter(|(_, d)| *d != 0).collect();
    grew.sort_by_key(|(_, d)| -d.abs());
    for (k, d) in grew.iter().take(20) {
        eprintln!("  {k}: {:+.2} MiB", *d as f64 / (1 << 20) as f64);
    }
    let growth = last.0 as i64 - first.0 as i64;
    assert!(growth <= 4 << 20, "{:.1} MiB more after laps 2 to {}", growth as f64 / (1 << 20) as f64, laps.len() - 1);
}
