//! KTX2 textures on a real GPU: every target the device supports is uploaded with its whole mip
//! chain through `Texture::from_levels`, read back texel by texel with `textureLoad`, and compared
//! with the official `basisu -unpack` CPU decode of the same target (for the uncompressed ones,
//! its RGBA32 output). That checks the upload (block order, row pitch, partial blocks, every mip)
//! rather than the codec's loss. Skipped (passes) when no adapter is available; targets the
//! adapter lacks are skipped and listed.

use kansei_core::buffers::{Bindable, Texture};
use kansei_core::loaders::ktx2::{self, CompressionSupport, GpuTarget};
use kansei_core::materials::{BindGroupBuilder, Binding, BindingResource};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let features = adapter.features() & CompressionSupport::FEATURES;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() }, None)).ok()
}

mod ktx2_fixtures;
use ktx2_fixtures::{fixture, golden, target};

/// Every texel of every level of `texture`, level 0 first, as the shader reads it.
/// For an array texture, each level holds every layer in turn.
fn read_texels(device: &wgpu::Device, queue: &wgpu::Queue, texture: &mut Texture) -> Vec<[f32; 4]> {
    texture.ensure_ready(device, queue);
    let size = texture.size();
    let levels = texture.gpu_texture().unwrap().mip_level_count();
    let layers = size.depth_or_array_layers;
    let array = layers > 1;
    let texels: u64 = (0..levels).map(|l| ((size.width >> l).max(1) * (size.height >> l).max(1) * layers) as u64).sum();
    let bytes = texels * 16;
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let texture_binding = if array { Binding::texture_2d_array(0, wgpu::ShaderStages::COMPUTE) } else { Binding::texture_2d(0, wgpu::ShaderStages::COMPUTE) };
    let bindings = [texture_binding, Binding::storage(1, wgpu::ShaderStages::COMPUTE, false)];
    let layout = BindGroupBuilder::create_layout(device, "Ktx2Read", &bindings);
    let group = BindGroupBuilder::create_bind_group(
        device,
        "Ktx2Read",
        &layout,
        &[(0, texture.binding_resource().unwrap()), (1, BindingResource::Buffer { buffer: &out, offset: 0, size: None })],
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            format!(
                r#"
            @group(0) @binding(0) var tex : {ty};
            @group(0) @binding(1) var<storage, read_write> out : array<vec4f>;
            @compute @workgroup_size(1) fn main() {{
                var i = 0u;
                for (var level = 0u; level < textureNumLevels(tex); level++) {{
                    let size = textureDimensions(tex, level);
                    for (var layer = 0u; layer < {layers}u; layer++) {{
                        for (var y = 0u; y < size.y; y++) {{
                            for (var x = 0u; x < size.x; x++) {{
                                out[i] = {load};
                                i++;
                            }}
                        }}
                    }}
                }}
            }}
            "#,
                ty = if array { "texture_2d_array<f32>" } else { "texture_2d<f32>" },
                load = if array {
                    "textureLoad(tex, vec2i(i32(x), i32(y)), i32(layer), i32(level))"
                } else {
                    "textureLoad(tex, vec2i(i32(x), i32(y)), i32(level))"
                },
            )
            .into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let values: Vec<f32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
    values.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

/// The reference channels each read channel is compared with; `None` skips it. The official
/// decodes of one- and two-channel targets hold them in R (and G); the uncompressed RG8 cut keeps
/// the Basis layout's G, which the RGBA32 reference has in alpha.
fn channel_map(target: GpuTarget, opaque: bool) -> [Option<usize>; 4] {
    use GpuTarget::*;
    let a = if opaque { None } else { Some(3) };
    match target {
        Bc4 | EacR11 | R8 => [Some(0), None, None, None],
        Bc5 | EacRg11 => [Some(0), Some(1), None, None],
        Rg8 => [Some(0), Some(3), None, None],
        Bc1 | Etc2Rgb => [Some(0), Some(1), Some(2), None],
        _ => [Some(0), Some(1), Some(2), a],
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

#[test]
fn every_supported_target_uploads_its_mips_and_reads_back() {
    let Some((device, queue)) = device() else { return eprintln!("no GPU adapter: skipping") };
    let support = CompressionSupport::of_device(&device);
    eprintln!("{support:?}");
    let mut skipped = vec![];
    for name in ["etc1s_rgb", "etc1s_rgba", "uastc_rgba", "uastc_normal", "etc1s_array", "uastc_array"] {
        let bytes = fixture(&format!("{name}.ktx2"));
        let info = ktx2::inspect(&bytes).unwrap();
        let (w, h) = (info.header.width, info.header.height);
        let goldens = golden(name);
        let rgba32 = goldens["cli/RGBA32"].concat();
        let mut cases: Vec<(GpuTarget, Vec<u8>)> = goldens
            .iter()
            .filter_map(|(k, v)| k.strip_prefix("decoded/").map(|t| (target(t), v.concat())))
            .collect();
        cases.extend([GpuTarget::Rgba8, GpuTarget::R8, GpuTarget::Rg8].map(|t| (t, rgba32.clone())));
        for (target, reference) in cases {
            if !support.supports(target) {
                skipped.push(target.name());
                continue;
            }
            let levels = ktx2::transcode_levels(&bytes, target).unwrap();
            let mut texture = match info.header.layers {
                0 => Texture::from_levels(name, target.format(false), w, h, levels),
                layers => Texture::from_array_levels(name, target.format(false), w, h, layers, levels),
            };
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let texels = read_texels(&device, &queue, &mut texture);
            assert!(pollster::block_on(device.pop_error_scope()).is_none(), "{name} {}: validation error", target.name());
            assert_eq!(texels.len() * 4, reference.len());
            let map = channel_map(target, !info.has_alpha);
            let mut worst = 0.0f32;
            for (i, texel) in texels.iter().enumerate() {
                for (c, source) in map.iter().enumerate() {
                    if let Some(s) = source {
                        worst = worst.max((texel[c] - reference[i * 4 + s] as f32 / 255.0).abs() * 255.0);
                    }
                }
            }
            eprintln!("{name:>12} {:>9}: {} texels in {} mips, largest difference {worst:.2}/255", target.name(), texels.len(), texture.gpu_texture().unwrap().mip_level_count());
            // decoders may round interpolated BC1/ETC/EAC colours differently; a misplaced block
            // or mip is off by far more. Where this machine's basisu wrote other BC7 p-bits than
            // ours (see generate.py) its decode can differ by one p-bit step of a 6-bit endpoint.
            let basis_name = goldens.keys().find_map(|k| k.strip_prefix("oracle/").filter(|n| ktx2_fixtures::target(n) == target));
            let other_blocks = basis_name.is_some_and(|n| goldens.get(&format!("cli/{n}")) != goldens.get(&format!("oracle/{n}")));
            let tolerance = if other_blocks { 255.0 / 63.0 + 0.01 } else { 3.0 };
            assert!(worst <= tolerance, "{name} {}: largest difference {worst}/255", target.name());
        }
    }
    skipped.sort();
    skipped.dedup();
    if !skipped.is_empty() {
        eprintln!("skipped, not supported by this adapter: {skipped:?}");
    }
}

#[test]
fn srgb_formats_decode_to_linear_light() {
    let Some((device, queue)) = device() else { return eprintln!("no GPU adapter: skipping") };
    let bytes = fixture("uastc_rgba.ktx2");
    let reference = ktx2::transcode_levels(&bytes, GpuTarget::Rgba8).unwrap();
    let t = ktx2::transcode("Srgb", &bytes, &ktx2::Ktx2Options::color(), CompressionSupport::NONE).unwrap();
    assert!(t.format.is_srgb());
    let mut texture = t.into_texture();
    let texels = read_texels(&device, &queue, &mut texture);
    for (i, texel) in texels.iter().take(20 * 12).enumerate() {
        for c in 0..3 {
            let want = srgb_to_linear(reference[0][i * 4 + c] as f32 / 255.0);
            assert!((texel[c] - want).abs() < 2e-3, "texel {i} channel {c}: {} vs {want}", texel[c]);
        }
        // alpha stays linear
        assert!((texel[3] - reference[0][i * 4 + 3] as f32 / 255.0).abs() < 1e-3);
    }
}

#[test]
fn hdr_targets_upload_and_read_back() {
    let Some((device, queue)) = device() else { return eprintln!("no GPU adapter: skipping") };
    let support = CompressionSupport::of_device(&device);
    let bytes = fixture("uastc_hdr.ktx2");
    let half = ktx2::transcode("Hdr", &bytes, &Default::default(), CompressionSupport::NONE).unwrap();
    assert_eq!(half.target, GpuTarget::Rgba16Float);
    let mut half_texture = half.into_texture();
    let reference = read_texels(&device, &queue, &mut half_texture);
    for target in [GpuTarget::Bc6h, GpuTarget::AstcHdr4x4] {
        let supported = match target {
            GpuTarget::Bc6h => support.bc,
            _ => support.astc_hdr,
        };
        if !supported {
            eprintln!("{}: not supported by this adapter, skipped", target.name());
            continue;
        }
        let levels = ktx2::transcode_levels(&bytes, target).unwrap();
        let mut texture = Texture::from_levels("Hdr", target.format(false), 20, 12, levels);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let texels = read_texels(&device, &queue, &mut texture);
        assert!(pollster::block_on(device.pop_error_scope()).is_none());
        let errors: Vec<f32> = texels
            .iter()
            .zip(&reference)
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs() / b[c].abs().max(0.05)))
            .collect();
        let worst = errors.iter().copied().fold(0.0f32, f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("uastc_hdr {}: relative error mean {mean:.3}, largest {worst:.3}", target.name());
        // ASTC HDR holds UASTC HDR exactly; BC6H is a lossy transcode (the fixture is noisy)
        let (mean_limit, limit) = if target == GpuTarget::AstcHdr4x4 { (1e-3, 1e-2) } else { (0.1, 0.6) };
        assert!(mean < mean_limit && worst < limit, "{}: mean {mean}, largest {worst}", target.name());
    }
}
