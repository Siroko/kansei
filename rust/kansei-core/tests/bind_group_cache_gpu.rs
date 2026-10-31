//! `cached_bind_group` on a real GPU: the same layout and resources give back the bind group made
//! before, another view (a ping-pong partner, a target made anew) a bind group of its own, and the
//! post-processing chain's passes stop making bind groups once their textures have come round.
//! Skipped (passes) when no adapter is available.

use kansei_core::renderers::{cached_bind_group, cached_bind_group_stats};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    Some(pollster::block_on(adapter.request_device(&Default::default())).unwrap())
}

#[test]
fn the_same_resources_give_back_the_same_bind_group() {
    let Some((device, _queue)) = device() else { return eprintln!("no GPU adapter: skipped") };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let texture = || {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let (ping, pong) = (texture(), texture());
    let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
    let group = |view: &wgpu::TextureView| {
        cached_bind_group(
            &device,
            &wgpu::BindGroupDescriptor {
                label: Some("test"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }, wgpu::BindGroupEntry { binding: 1, resource: uniform.as_entire_binding() }],
            },
        )
    };
    let (made, _) = cached_bind_group_stats();
    let (a, b) = (group(&ping), group(&pong));
    assert_ne!(a, b, "another view, another bind group");
    for _ in 0..10 {
        assert_eq!(group(&ping), a);
        assert_eq!(group(&pong), b);
    }
    assert_eq!(cached_bind_group_stats().0 - made, 2, "two made, the rest from the cache");
    // a target made anew: its view is another, so is its bind group
    let resized = texture();
    assert!(group(&resized) != a && group(&resized) != b);
    assert_eq!(cached_bind_group_stats().0 - made, 3);
}
