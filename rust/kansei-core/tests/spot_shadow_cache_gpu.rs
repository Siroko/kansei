//! The spot shadow maps' cache of still casters on a real GPU: two shadowed spot lights over a
//! floor and a row of blocks, one block `dynamic` and moving, one that moves without being
//! `dynamic` (for a while, then rests), a light that moves and comes back, a caster hidden and
//! shown. Every frame's atlas is the same, texel for texel, with the cache as without it, and once
//! everything has rested the layers draw only what moves. Skipped (passes) when no adapter is
//! available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig, SpotShadowCacheStats, SPOT_CACHE_STILL_FRAMES};

const W: u32 = 96;
const H: u32 = 64;
const RES: u32 = 256;
const FRAMES: usize = 60;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn lit() -> Material {
    Material::standard_lit("Lit", &StandardLitOptions { base_color: [0.7; 3], roughness: 0.6, ..Default::default() })
}

fn lamp(at: [f32; 3]) -> Light {
    let mut lamp = SpotLight::new(Vec3::new(at[0], at[1], at[2]), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 10.0, 20.0, 50f32.to_radians(), 60f32.to_radians());
    lamp.cast_shadow = true;
    Light::Spot(lamp)
}

/// The atlas's depth words, every layer.
fn read_atlas(renderer: &Renderer) -> Vec<u32> {
    let atlas = renderer.spot_shadow_atlas().unwrap();
    let layers = atlas.layers;
    let row = RES * 4;
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * RES * layers) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &atlas.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::DepthOnly },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(RES) } },
        wgpu::Extent3d { width: RES, height: RES, depth_or_array_layers: layers },
    );
    renderer.queue().submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
    let words = bytemuck::cast_slice(&buffer.slice(..).get_mapped_range().unwrap()).to_vec();
    words
}

/// Each frame's atlas and the cache's stats, the cache on or off.
fn frames(cache: bool) -> Option<Vec<(Vec<u32>, SpotShadowCacheStats)>> {
    let mut renderer = renderer()?;
    renderer.enable_spot_shadows(RES, 2);
    renderer.set_spot_shadow_cache(cache);
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), lit());
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let mut blocks = Vec::new();
    for k in 0..12 {
        let mut b = Renderable::new(BoxGeometry::new(0.3, 0.4 + 0.05 * k as f32, 0.3), lit());
        b.object.set_position(-2.2 + 0.4 * k as f32, 0.3, -0.5 + 0.1 * (k % 3) as f32);
        blocks.push(scene.add(SceneNode::Renderable(b)));
    }
    let mut mover = Renderable::new(BoxGeometry::new(0.25, 0.25, 0.25), lit());
    mover.dynamic = true;
    let mover = scene.add(SceneNode::Renderable(mover));
    let lamps = [scene.add(SceneNode::Light(lamp([-1.0, 3.0, 0.0]))), scene.add(SceneNode::Light(lamp([1.2, 2.5, 0.3])))];
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(0.0, 3.0, 5.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    let mut out = Vec::new();
    for f in 0..FRAMES {
        let t = f as f32 / 10.0;
        scene.get_renderable_mut(mover).unwrap().object.set_position(0.8 * t.sin(), 0.5, 0.6 * t.cos());
        // a block that moves, not dynamic, for frames 10-24, then rests
        if (10..25).contains(&f) {
            scene.get_renderable_mut(blocks[3]).unwrap().object.set_position(-1.0 + 0.03 * (f - 10) as f32, 0.3, 0.4);
        }
        // a block hidden for frames 30-34
        scene.get_renderable_mut(blocks[8]).unwrap().visible = !(30..35).contains(&f);
        // the second light moves at frame 40 and comes back at 45
        let at = if (40..45).contains(&f) { [1.4, 2.5, 0.3] } else { [1.2, 2.5, 0.3] };
        if let Some(Light::Spot(l)) = scene.get_light_mut(lamps[1]) {
            l.position = Vec3::new(at[0], at[1], at[2]);
        }
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        out.push((read_atlas(&renderer), renderer.spot_shadow_cache_stats()));
    }
    Some(out)
}

#[test]
fn still_casters_from_the_cache_give_the_same_shadow_maps() {
    let Some(cached) = frames(true) else { return eprintln!("no GPU adapter: skipping") };
    let live = frames(false).unwrap();
    for (f, ((a, stats), (b, _))) in cached.iter().zip(&live).enumerate() {
        let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "frame {f}: {differ} texels differ ({stats:?})");
    }
    let shadowed = live[FRAMES - 1].0.iter().filter(|&&d| f32::from_bits(d) < 1.0).count();
    assert!(shadowed > 1000, "the lights see the casters ({shadowed} texels)");
    for (f, (_, s)) in cached.iter().enumerate() {
        eprintln!("frame {f}: {s:?}");
    }
    // at rest (past the moving block, the hidden one and the moving light, and the frames a
    // change takes to be cached): only the dynamic block is drawn, from both layers' caches
    let last = cached[FRAMES - 1].1;
    assert_eq!(last.layers_recached, 0, "{last:?}");
    assert!(last.live_draws <= 2 && last.kept >= 20, "{last:?}");
    let off = live[FRAMES - 1].1;
    assert!(off.live_draws >= 20 && off.kept == 0, "{off:?}");
    assert!(SPOT_CACHE_STILL_FRAMES < 15);
}

/// The cache's cost and gain on this machine, for the PR's numbers (`cargo test --release -p
/// kansei-core --test spot_shadow_cache_gpu -- --ignored --nocapture`): 400 still blocks and a
/// moving one under four shadowed spot lights at 2048², the cache on and off in alternating
/// blocks of frames: the spot passes' GPU time (the cache's copy included) and the CPU time of
/// recording a frame.
#[test]
#[ignore]
fn cache_timings() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_spot_shadows(2048, 4);
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(20.0, 0.2, 20.0), lit());
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    for k in 0..400 {
        let mut b = Renderable::new(BoxGeometry::new(0.3, 0.3 + 0.01 * (k % 17) as f32, 0.3), lit());
        b.object.set_position(-9.5 + 0.95 * (k % 20) as f32, 0.3, -9.5 + 0.95 * (k / 20) as f32);
        scene.add(SceneNode::Renderable(b));
    }
    let mut mover = Renderable::new(BoxGeometry::new(0.25, 0.25, 0.25), lit());
    mover.dynamic = true;
    let mover = scene.add(SceneNode::Renderable(mover));
    for at in [[-4.0, 6.0, -4.0], [4.0, 6.0, -4.0], [-4.0, 6.0, 4.0], [4.0, 6.0, 4.0]] {
        scene.add(SceneNode::Light(lamp(at)));
    }
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.1, 80.0, W as f32 / H as f32);
    camera.set_position(0.0, 12.0, 14.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    renderer.set_profiling(true);
    let (mut gpu, mut cpu, mut n) = ([0.0f64; 2], [0.0f64; 2], [0usize; 2]);
    let mut frame = 0;
    for round in 0..6 {
        for (m, on) in [true, false].into_iter().enumerate() {
            renderer.set_spot_shadow_cache(on);
            for k in 0..40 {
                frame += 1;
                let t = frame as f32 / 20.0;
                scene.get_renderable_mut(mover).unwrap().object.set_position(3.0 * t.sin(), 0.5, 3.0 * t.cos());
                if k == 20 {
                    let _ = renderer.take_profile();
                }
                let t0 = std::time::Instant::now();
                renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
                let recorded = t0.elapsed().as_secs_f64() * 1e3;
                renderer.end_profiled_frame();
                let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
                if k >= 20 {
                    cpu[m] += recorded;
                    n[m] += 1;
                }
            }
            let profile = renderer.take_profile();
            let spot: f64 = profile.gpu.iter().filter(|p| p.label.starts_with("Renderer/SpotShadow")).map(|p| p.busy_ms).sum();
            gpu[m] += spot;
            eprintln!("round {round}, cache {on}: spot passes {spot:.3} ms, {:?}", renderer.spot_shadow_cache_stats());
        }
    }
    eprintln!("cache on: spot passes {:.3} ms GPU, frame recorded in {:.3} ms CPU", gpu[0] / 6.0, cpu[0] / n[0] as f64);
    eprintln!("cache off: spot passes {:.3} ms GPU, frame recorded in {:.3} ms CPU", gpu[1] / 6.0, cpu[1] / n[1] as f64);
}
