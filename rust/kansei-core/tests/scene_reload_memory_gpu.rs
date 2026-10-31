//! GPU memory over scene reloads, on a real GPU (a game loading level after level on one
//! renderer): each lap builds a new scene of new meshes (ray traced, voxelized with the distance
//! field, casting spot shadows, one of them moving), renders a few frames and drops it. What the
//! GPU allocator holds after each lap (`Device::generate_allocator_report`, where the backend has
//! one: Vulkan, DX12) must stop growing once the renderer's own buffers have grown to the scenes;
//! the labels that grew are printed. Skipped (passes) without an adapter or an allocator report.
//! `cargo test --release -p kansei-core --test scene_reload_memory_gpu -- --ignored --nocapture`.

use std::collections::BTreeMap;

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtGridOptions, RtSurface, SceneRtGridOptions};

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

/// Lap `lap`'s scene: 60 blocks of sizes of its own (new meshes), one moving.
fn scene(lap: u32) -> (Scene, usize) {
    let lit = |c: [f32; 3]| Material::standard_lit("Lit", &StandardLitOptions { base_color: c, roughness: 0.6, ..Default::default() });
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), lit([0.6; 3])).with_gi(GiSurface::new([0.6; 3])).with_rt(RtSurface::new([0.6; 3]));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    for k in 0..60 {
        let s = 0.1 + 0.002 * (k as f32) + 0.01 * lap as f32;
        let mut b = Renderable::new(BoxGeometry::new(s, s * 1.5, s), lit([0.7, 0.5, 0.3])).with_gi(GiSurface::new([0.7, 0.5, 0.3])).with_rt(RtSurface::new([0.7, 0.5, 0.3]));
        b.object.set_position(-2.5 + 0.6 * (k % 10) as f32, s * 0.75, -2.5 + 0.8 * (k / 10) as f32);
        scene.add(SceneNode::Renderable(b));
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
    renderer.enable_spot_shadows(512, 2);
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 6.0 / 64.0, fixed_origin: Some(glam::Vec3::new(-3.0, -0.3, -3.0)), ..Default::default() }, ..Default::default() });
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-3.0, -0.3, -3.0], bounds_max: [3.0, 2.7, 3.0], ..Default::default() });
    renderer.voxel_gi_mut().unwrap().enable_sdf();
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(0.0, 3.0, 5.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    let mut laps = Vec::new();
    for lap in 0..8 {
        {
            let (mut scene, mover) = scene(lap);
            for f in 0..6 {
                scene.get_renderable_mut(mover).unwrap().object.set_position(0.1 * f as f32, 0.5, 0.0);
                renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            }
        }
        let Some(r) = report(&renderer) else { return eprintln!("no allocator report on this backend: skipped") };
        eprintln!("lap {lap}: {:.1} MiB", r.0 as f64 / (1 << 20) as f64);
        laps.push(r);
    }
    let (first, last) = (&laps[2], &laps[laps.len() - 1]);
    let mut grew: Vec<(String, i64)> = last.1.iter().map(|(k, v)| (k.clone(), *v as i64 - *first.1.get(k).unwrap_or(&0) as i64)).filter(|(_, d)| *d != 0).collect();
    grew.sort_by_key(|(_, d)| -d.abs());
    for (k, d) in grew.iter().take(20) {
        eprintln!("  {k}: {:+.2} MiB", *d as f64 / (1 << 20) as f64);
    }
    let growth = last.0 as i64 - first.0 as i64;
    assert!(growth <= 4 << 20, "{:.1} MiB more after laps 2 to {}", growth as f64 / (1 << 20) as f64, laps.len() - 1);
}
