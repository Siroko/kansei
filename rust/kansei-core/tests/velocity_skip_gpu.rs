//! The renderer's velocity pass runs only for a post-processing chain that reads the GBuffer's
//! velocity (`PostProcessingEffect::reads_velocity`): drawn under a TAA chain, skipped under a
//! tonemapper alone. A headless renderer (no surface: the frame is drawn into the GBuffer and
//! skipped at the surface). Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{TemporalAAEffect, ToneMapEffect};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};

const SIZE: u32 = 64;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let mut renderer = Renderer::new(RendererConfig { width: SIZE, height: SIZE, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The volume's velocity texture, (x, y) per texel.
fn read_velocity(renderer: &Renderer, volume: &PostProcessingVolume) -> Vec<[f32; 2]> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let texture = &volume.gbuffer().unwrap().velocity_texture;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (SIZE * SIZE * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SIZE * 4), rows_per_image: Some(SIZE) } },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, u16>(&data).chunks(2).map(|v| [f16_to_f32(v[0]), f16_to_f32(v[1])]).collect()
}

#[test]
fn the_velocity_pass_runs_only_for_a_chain_that_reads_it() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    let mut scene = Scene::new();
    let material = Material::standard_lit("Box", &StandardLitOptions { base_color: [0.6, 0.4, 0.3], outputs_velocity: true, ..Default::default() });
    scene.add(SceneNode::Renderable(Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), material)));
    let mut camera = Camera::new(45.0, 0.1, 100.0, 1.0);
    camera.set_position(0.0, 0.0, 3.0);
    camera.look_at(&kansei_core::math::Vec3::new(0.0, 0.0, 0.0));

    // TAA reads the velocity: the pass clears the texture to NO_VELOCITY and draws the box's
    // (still) motion vectors over it
    let effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(TemporalAAEffect::new(Default::default())), Box::new(ToneMapEffect::new(Default::default()))];
    let mut volume = PostProcessingVolume::new(&renderer, effects);
    renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    let velocity = read_velocity(&renderer, &volume);
    assert_eq!(velocity[0], [GBuffer::NO_VELOCITY; 2], "the sky's texel");
    let centre = velocity[(SIZE / 2 * SIZE + SIZE / 2) as usize];
    assert!(centre.iter().all(|v| v.is_finite() && v.abs() < 1e-3), "the box's texel: {centre:?}");

    // a tonemapper alone reads none: the pass never runs, so the texture is as created (zeros)
    let mut volume = PostProcessingVolume::new(&renderer, vec![Box::new(ToneMapEffect::new(Default::default()))]);
    renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    let velocity = read_velocity(&renderer, &volume);
    assert!(velocity.iter().all(|v| *v == [0.0, 0.0]), "the velocity pass ran for a chain that doesn't read it");
}
