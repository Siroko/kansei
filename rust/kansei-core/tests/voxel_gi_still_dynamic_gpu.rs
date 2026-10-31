//! Voxel GI's distance field with `SceneVoxelGi::skip_still_dynamic` on a real GPU: a dynamic
//! block over a floor moves for a while, then rests, then moves again. The frames it rests skip
//! its voxelization and flood, and every frame's field and lit volume are the same, voxel for
//! voxel, as with the skip off. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

const W: u32 = 64;
const H: u32 = 48;
const FRAMES: usize = 30;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// A texture's bytes, every layer of mip 0.
fn read_texture(renderer: &Renderer, texture: &wgpu::Texture, texel_bytes: u32) -> Vec<u8> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let size = texture.size();
    let row = (size.width * texel_bytes).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * size.height * size.depth_or_array_layers) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.height) } },
        size,
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let bytes = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    bytes
}

/// Each frame's distance field and lit volume, and whether it voxelized the block again.
fn frames(skip: bool) -> Option<Vec<(Vec<u8>, Vec<u8>, bool)>> {
    let mut renderer = renderer()?;
    renderer.enable_spot_shadows(256, 1);
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    let gi = renderer.voxel_gi_mut().unwrap();
    gi.enable_sdf();
    gi.skip_still_dynamic = skip;
    let lit = |c: [f32; 3]| Material::standard_lit("Lit", &StandardLitOptions { base_color: c, roughness: 0.6, ..Default::default() });
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(1.8, 0.2, 1.8), lit([0.6; 3])).with_gi(GiSurface::new([0.6; 3]));
    floor.object.set_position(0.0, -0.7, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let mut block = Renderable::new(BoxGeometry::new(0.3, 0.3, 0.3), lit([0.8, 0.3, 0.2])).with_gi(GiSurface::new([0.8, 0.3, 0.2]));
    block.dynamic = true;
    let block = scene.add(SceneNode::Renderable(block));
    let mut lamp = SpotLight::new(Vec3::new(0.0, 0.9, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 10.0, 20.0, 60f32.to_radians(), 70f32.to_radians());
    lamp.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Spot(lamp)));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.1, 20.0, W as f32 / H as f32);
    camera.set_position(0.0, 1.0, 2.5);
    camera.look_at(&Vec3::new(0.0, -0.3, 0.0));
    camera.update_projection_matrix();
    let mut out = Vec::new();
    for f in 0..FRAMES {
        // moves for frames 0-7 and 20-24, rests between
        let t = if f < 8 { f } else if f < 20 { 7 } else if f < 25 { f - 12 } else { 12 } as f32;
        scene.get_renderable_mut(block).unwrap().object.set_position(-0.4 + 0.05 * t, -0.3, 0.02 * t);
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        let gi = renderer.voxel_gi().unwrap();
        out.push((read_texture(&renderer, gi.sdf().unwrap().texture(), 4), read_texture(&renderer, gi.volume().texture(), 8), gi.dynamic_redone()));
    }
    Some(out)
}

#[test]
fn a_still_dynamic_block_keeps_the_same_field_without_flooding_it() {
    let Some(skipped) = frames(true) else { return eprintln!("no GPU adapter: skipping") };
    let every = frames(false).unwrap();
    for (f, ((sdf, volume, _), (sdf_every, volume_every, redone))) in skipped.iter().zip(&every).enumerate() {
        assert!(redone, "without the skip every frame voxelizes the block");
        assert!(sdf == sdf_every, "frame {f}: the distance field differs");
        assert!(volume == volume_every, "frame {f}: the lit volume differs");
    }
    let redone: Vec<bool> = skipped.iter().map(|s| s.2).collect();
    eprintln!("voxelized again: {redone:?}");
    // the frames it rests after the first (whose key is the frame before's) skip it
    assert!(redone[..8].iter().all(|&r| r) && redone[20..25].iter().all(|&r| r));
    assert!(redone[9..20].iter().all(|&r| !r) && redone[26..].iter().all(|&r| !r));
}
