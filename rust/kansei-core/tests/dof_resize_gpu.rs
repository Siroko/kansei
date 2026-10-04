//! DepthOfFieldEffect follows a resize: rendered once, then at a larger size, a flat colour comes
//! out flat over the whole new image (its textures were kept at the first size, leaving the new
//! area black). Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{DepthOfFieldEffect, DepthOfFieldOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

const COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

fn f16_to_f32(h: u16) -> f32 {
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => frac * 2f32.powi(-24),
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn f32_to_f16(v: f32) -> u16 {
    // positive normal values only
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((exp as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// The effect's output (rgb per pixel) for a flat `COLOR` input with every pixel at the far plane.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, effect: &mut DepthOfFieldEffect, w: u32, h: u32) -> Vec<[f32; 3]> {
    let gbuffer = GBuffer::new(device, w, h, 1);
    effect.resize(w, h, &gbuffer);
    let camera = Camera::new(40.0, 0.1, 100.0, w as f32 / h as f32);
    effect.initialize(device, &gbuffer, &camera);
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let texture = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor { label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
    };
    let input = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    let texel: Vec<u16> = COLOR.iter().map(|&c| f32_to_f16(c)).collect();
    let pixels: Vec<u16> = (0..w * h).flat_map(|_| texel.iter().copied()).collect();
    queue.write_texture(input.as_image_copy(), bytemuck::cast_slice(&pixels), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: None }, size);
    let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
    let output = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
    let (input_view, depth_view, output_view) = (input.create_view(&Default::default()), depth.create_view(&Default::default()), output.create_view(&Default::default()));

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    effect.render(device, queue, &mut encoder, &gbuffer, &input_view, &depth_view, &output_view, &camera, w, h);
    let row = (w * 8).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        size,
    );
    queue.submit(std::iter::once(encoder.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = readback.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let o = (y * row + x * 8) as usize;
            let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
            out.push([c(0), c(1), c(2)]);
        }
    }
    out
}

#[test]
fn a_flat_image_stays_flat_after_a_resize() {
    let Some((device, queue)) = gpu() else { return };
    let mut effect = DepthOfFieldEffect::new(DepthOfFieldOptions::default());
    render(&device, &queue, &mut effect, 64, 48);
    let (w, h) = (160, 96);
    let out = render(&device, &queue, &mut effect, w, h);
    for (i, px) in out.iter().enumerate() {
        for c in 0..3 {
            assert!((px[c] - COLOR[c]).abs() < 0.02, "pixel ({}, {}) is {px:?}, not {COLOR:?}", i as u32 % w, i as u32 / w);
        }
    }
}
