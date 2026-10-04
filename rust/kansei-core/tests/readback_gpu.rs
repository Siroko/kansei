//! `Renderer::read_buffer_async` on a real GPU: a buffer comes back whole, the future outlives
//! the borrow of the renderer, and reads see the writes submitted before them.
//! Skipped (passes) when no adapter is available.

use kansei_core::renderers::{Renderer, RendererConfig};
use wgpu::util::DeviceExt;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: 64, height: 64, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

#[test]
fn a_buffer_reads_back_whole_and_after_later_writes() {
    let Some(renderer) = renderer() else { return };
    let values: Vec<[f32; 4]> = (0..4096).map(|i| [i as f32, -(i as f32), 0.5, 1.0]).collect();
    let buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Readback/Test"),
        contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    let first = renderer.read_buffer_async::<[f32; 4]>(&buffer);
    renderer.queue().write_buffer(&buffer, 16, bytemuck::cast_slice(&[[7.0f32; 4]]));
    let second = renderer.read_buffer_async::<u32>(&buffer);
    drop(renderer);

    let first = pollster::block_on(first).expect("mapped");
    assert_eq!(first, values);
    let second = pollster::block_on(second).expect("mapped");
    assert_eq!(second.len(), values.len() * 4);
    assert_eq!(&second[..4], bytemuck::cast_slice::<f32, u32>(&values[0]));
    assert_eq!(&second[4..8], &[7.0f32.to_bits(); 4]);
}
