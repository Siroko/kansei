//! Builds DepthPyramid on a real GPU from a noisy depth buffer of odd and even sizes, with each
//! reduction, and checks every texel of every mip against the documented coverage: texel (x, y)
//! of mip L reduces exactly the depth pixels [x, x + 1) * 2^(L + 1), clipped to the buffer, and
//! the top mip is one texel.
//! Skipped (passes) when no adapter is available.

use kansei_core::culling::{DepthPyramid, DepthReduction};

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// A full-screen triangle writing a hash of the pixel as its depth.
const NOISE_DEPTH: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.5, 1.0);
}
@fragment
fn fs(@builtin(position) p : vec4f) -> @builtin(frag_depth) f32 {
    var x = u32(p.x) * 73856093u ^ u32(p.y) * 19349663u;
    x ^= x >> 13u;
    x *= 0x5bd1e995u;
    x ^= x >> 15u;
    return f32(x & 0xffffu) / 65536.0;
}
"#;

/// Read back a (mip of a) 4- or 8-byte-per-texel texture as rows of f32 texels.
fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, level: u32, (w, h): (u32, u32), channels: u32, aspect: wgpu::TextureAspect) -> Vec<Vec<f32>> {
    let row = (w * 4 * channels).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    (0..h as usize)
        .map(|y| bytemuck::cast_slice::<u8, f32>(&bytes[y * row as usize..][..(w * 4 * channels) as usize]).to_vec())
        .collect()
}

#[test]
fn every_texel_reduces_exactly_the_pixels_it_covers() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(NOISE_DEPTH.into()) });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });

    for (w, h) in [(37, 23), (64, 32), (1, 5)] {
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        let pixels = read_texture(&device, &queue, &depth, 0, (w, h), 1, wgpu::TextureAspect::DepthOnly);
        assert!(pixels.iter().flatten().any(|&d| d != pixels[0][0]), "the noise varies");

        for reduction in [DepthReduction::Max, DepthReduction::Min, DepthReduction::MinMax] {
            let mut pyramid = DepthPyramid::new(&device, 4, 4, reduction);
            pyramid.resize(&device, w, h);
            let mut encoder = device.create_command_encoder(&Default::default());
            pyramid.build(&device, &mut encoder, &depth_view);
            queue.submit(Some(encoder.finish()));
            let channels = if reduction == DepthReduction::MinMax { 2 } else { 1 };
            for level in 0..pyramid.mip_count() {
                let (mw, mh) = pyramid.mip_size(level);
                let texels = read_texture(&device, &queue, pyramid.texture(), level, (mw, mh), channels, wgpu::TextureAspect::All);
                let shift = level + 1;
                for ty in 0..mh {
                    for tx in 0..mw {
                        let covered: Vec<f32> = ((ty << shift)..((ty + 1) << shift).min(h))
                            .flat_map(|y| ((tx << shift)..((tx + 1) << shift).min(w)).map(move |x| (x, y)))
                            .map(|(x, y)| pixels[y as usize][x as usize])
                            .collect();
                        if covered.is_empty() {
                            continue; // wholly past the buffer (mip 0 is a power of two)
                        }
                        let lo = covered.iter().copied().fold(f32::INFINITY, f32::min);
                        let hi = covered.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let texel = &texels[ty as usize][(tx * channels) as usize..][..channels as usize];
                        let expected: &[f32] = match reduction {
                            DepthReduction::Max => &[hi],
                            DepthReduction::Min => &[lo],
                            DepthReduction::MinMax => &[lo, hi],
                        };
                        assert_eq!(texel, expected, "{reduction:?}, {w} x {h}, mip {level}, texel ({tx}, {ty})");
                    }
                }
            }
            assert_eq!(pyramid.mip_size(pyramid.mip_count() - 1), (1, 1));
        }
    }
}
