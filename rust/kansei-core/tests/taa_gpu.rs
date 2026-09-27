//! Runs TemporalAAEffect on a real GPU over a still, nearly horizontal edge, point-sampled with
//! the renderer's Halton jitter, at 1:1 and as a temporal upscaler (from 0.67 and 0.5), and checks that the
//! upscaled edge converges to the right place in every output column (no staircase of rendered
//! pixels) and is about as sharp as the 1:1 one. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{TemporalAAEffect, TemporalAAOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

/// Output size.
const W: u32 = 128;
const H: u32 = 64;
/// The edge: y = EDGE_Y0 + EDGE_SLOPE x in output pixels, 1.0 below it and 0.0 above.
const EDGE_Y0: f32 = 24.3;
const EDGE_SLOPE: f32 = 0.13;
const FRAMES: u32 = 256;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
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

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut result, mut f) = (0.0, 1.0);
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

/// The scene at a point in output pixels: bright below the edge.
fn edge(x: f32, y: f32) -> f32 {
    if y > EDGE_Y0 + EDGE_SLOPE * x { 1.0 } else { 0.0 }
}

/// A dark wire LINE_WIDTH output pixels thick across a bright sky, along the edge's slope: 0.8
/// of a rendered pixel from 0.5. (Much thinner than a rendered pixel, it is missed by most
/// frames' samples and the neighbourhood clip erases it, as it does at 1:1 below 0.5 px.)
const LINE_WIDTH: f32 = 1.6;
fn wire(x: f32, y: f32) -> f32 {
    let d = y - (EDGE_Y0 + EDGE_SLOPE * x);
    if (0.0..LINE_WIDTH).contains(&d) { 0.0 } else { 1.0 }
}

fn texture(device: &wgpu::Device, (w, h): (u32, u32), format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// Resolve FRAMES jittered frames rendered at `input` into a W x H output, as the renderer does
/// (8 jitter phases per output pixel); the grey level of each output pixel.
fn resolve(device: &wgpu::Device, queue: &wgpu::Queue, scene: fn(f32, f32) -> f32, input: (u32, u32)) -> Vec<f32> {
    let gbuffer = GBuffer::new(device, input.0, input.1, 1);
    let color = texture(device, input, wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    let output = texture(device, (W, H), wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
    let (color_view, output_view) = (color.create_view(&Default::default()), output.create_view(&Default::default()));
    let mut effect = TemporalAAEffect::new(TemporalAAOptions::default());
    let mut camera = Camera::new(40.0, 0.1, 100.0, W as f32 / H as f32);
    let scale = input.0 as f32 / W as f32;
    let phases = (8.0 / (scale * scale)).round() as u32;
    for frame in 0..FRAMES {
        let i = frame % phases + 1;
        let (jx, jy) = (halton(i, 2) - 0.5, halton(i, 3) - 0.5);
        camera.jitter = [2.0 * jx / input.0 as f32, 2.0 * jy / input.1 as f32];
        // the resolve's convention: the sample of rendered pixel k sits at k + 0.5 - jitter_px,
        // with jitter_px = (jx, -jy) (+y down)
        let mut rgba = Vec::with_capacity((input.0 * input.1 * 4) as usize);
        for y in 0..input.1 {
            for x in 0..input.0 {
                let sx = (x as f32 + 0.5 - jx) / scale;
                let sy = (y as f32 + 0.5 + jy) / scale;
                let v = scene(sx, sy);
                rgba.extend_from_slice(&[v, v, v, 1.0]);
            }
        }
        queue.write_texture(
            color.as_image_copy(),
            bytemuck::cast_slice(&rgba),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(input.0 * 16), rows_per_image: None },
            color.size(),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        effect.render(device, queue, &mut encoder, &gbuffer, &color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
        queue.submit(std::iter::once(encoder.finish()));
    }

    let row = (W * 8).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
        output.size(),
    );
    queue.submit(std::iter::once(encoder.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = readback.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let o = (y * row + x * 8) as usize;
            out.push(f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])));
        }
    }
    out
}

/// Coverage of each output pixel by the bright side as the resolve averaged it: it blends in
/// x / (1 + x), where the scene's 0 and 1 are 0 and 0.5.
fn coverage(grey: &[f32]) -> Vec<f32> {
    grey.iter().map(|&x| 2.0 * x / (1.0 + x)).collect()
}

/// RMS distance (output pixels) between the edge found in each column (the column's dark
/// pixels, which any symmetric filter keeps) and the true edge, away from the image's sides;
/// and the mean width of the transition (sum of c (1 - c) per column).
fn edge_error_and_width(cov: &[f32]) -> (f32, f32) {
    let columns = 8..W - 8;
    let n = columns.len() as f32;
    let (mut err2, mut width) = (0.0, 0.0);
    for x in columns {
        let column = (0..H).map(|y| cov[(y * W + x) as usize]);
        let dark: f32 = column.clone().map(|c| 1.0 - c).sum();
        let truth = EDGE_Y0 + EDGE_SLOPE * (x as f32 + 0.5);
        err2 += (dark - truth).powi(2);
        width += column.map(|c| c * (1.0 - c)).sum::<f32>();
    }
    ((err2 / n).sqrt(), width / n)
}

#[test]
fn upscaled_edge_converges_without_a_staircase() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (full_err, full_width) = edge_error_and_width(&coverage(&resolve(&device, &queue, edge, (W, H))));
    let (half_err, half_width) = edge_error_and_width(&coverage(&resolve(&device, &queue, edge, (W / 2, H / 2))));
    let (two_thirds_err, two_thirds_width) = edge_error_and_width(&coverage(&resolve(&device, &queue, edge, (86, 43))));
    // one unjittered frame at half resolution, stretched: the staircase the upscaler removes
    let stretched: Vec<f32> = (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            edge(((x / 2) * 2) as f32 + 1.0, ((y / 2) * 2) as f32 + 1.0)
        })
        .collect();
    let (stair_err, _) = edge_error_and_width(&stretched);
    eprintln!(
        "edge RMS error: 1:1 {full_err:.3} px, from 0.67 {two_thirds_err:.3} px, from 0.5 {half_err:.3} px, stretched {stair_err:.3} px; \
         width 1:1 {full_width:.3}, from 0.67 {two_thirds_width:.3}, from 0.5 {half_width:.3}"
    );
    assert!(full_err < 0.1, "1:1 edge off by {full_err} px");
    assert!(stair_err > 0.4, "the stretched frame should show a staircase ({stair_err} px)");
    assert!(half_err < 0.1 && half_err < stair_err / 3.0, "upscaled edge off by {half_err} px (staircase {stair_err} px)");
    assert!(half_width < full_width * 1.5, "upscaled edge {half_width} wide, 1:1 {full_width}");
    assert!(two_thirds_err < 0.1, "edge upscaled from 0.67 off by {two_thirds_err} px");
    assert!(two_thirds_width < full_width * 1.5, "edge upscaled from 0.67 {two_thirds_width} wide, 1:1 {full_width}");
}

/// Mean darkness per column (the wire's width as the resolve sees it), away from the sides.
fn wire_width(cov: &[f32]) -> f32 {
    let columns = 8..W - 8;
    let n = columns.len() as f32;
    columns.map(|x| (0..H).map(|y| 1.0 - cov[(y * W + x) as usize]).sum::<f32>()).sum::<f32>() / n
}

#[test]
fn a_wire_under_a_rendered_pixel_thick_keeps_its_weight() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let widths: Vec<f32> = [(W, H), (86, 43), (W / 2, H / 2)].into_iter().map(|input| wire_width(&coverage(&resolve(&device, &queue, wire, input)))).collect();
    eprintln!("wire {LINE_WIDTH} px wide resolves to: 1:1 {:.3}, from 0.67 {:.3}, from 0.5 {:.3}", widths[0], widths[1], widths[2]);
    for w in widths {
        assert!((w - LINE_WIDTH).abs() < 0.1 * LINE_WIDTH, "{w}");
    }
}
