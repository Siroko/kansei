//! Prepares HDR environments on a real GPU and reads back what the consumers see: the SH and the
//! cubemap of a constant environment (its radiance everywhere, its irradiance pi times it), and
//! the turn about the vertical. Skipped (passes) when no adapter is available.

use kansei_core::environment::{HdrEnvironment, HdrEnvironmentOptions, HdrEnvironmentSettings, HdrImage};

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

fn f16_to_f32(h: u16) -> f32 {
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    let v = match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    };
    if h & 0x8000 != 0 { -v } else { v }
}

fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = staging.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&data).to_vec()
}

/// Every texel of mip `mip` (6 faces, rgb), face by face.
fn read_mip(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, mip: u32) -> (u32, Vec<[f32; 3]>) {
    let size = (texture.width() >> mip).max(1);
    let row = (size * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * size * 6) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size) } },
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
    );
    queue.submit(std::iter::once(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer.slice(..).get_mapped_range().unwrap();
    let mut texels = Vec::new();
    for y in 0..size * 6 {
        for x in 0..size {
            let o = (y * row + x * 8) as usize;
            let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
            texels.push([c(0), c(1), c(2)]);
        }
    }
    (size, texels)
}

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1e-3)
}

#[test]
fn a_constant_environment_is_its_radiance_everywhere_and_pi_times_it_on_a_surface() {
    let Some((device, queue)) = gpu() else { return };
    let c = [2.0, 1.0, 0.5];
    let image = HdrImage::from_pixels(128, 64, vec![c; 128 * 64]);
    let mut env = HdrEnvironment::new(&device, &queue, &image, HdrEnvironmentOptions::default());
    // a quarter of the width, as a power of two
    assert_eq!(env.face_size(), 32);
    assert!(env.prepare(&device, &queue));
    // nothing changed: nothing to do
    assert!(!env.prepare(&device, &queue));

    let sky = read_buffer(&device, &queue, &env.lighting().sky_lighting);
    let sh = |i: usize, ch: usize| sky[i * 4 + ch];
    for ch in 0..3 {
        // skyRadiance = 0.282095 sh0 (the other bands nothing)
        assert!(close(0.282095 * sh(0, ch), c[ch], 0.01), "band 0 {ch}: {}", 0.282095 * sh(0, ch));
        for i in 1..9 {
            assert!(sh(i, ch).abs() < 0.01 * c[ch], "band {i} {ch}: {}", sh(i, ch));
        }
        // clearSkyUp: the irradiance on an upward surface, pi c; distantSkyLight: the mean
        assert!(close(sky[13 * 4 + ch], std::f32::consts::PI * c[ch], 0.01), "irradiance {ch}: {}", sky[13 * 4 + ch]);
        assert!(close(sky[14 * 4 + ch], c[ch], 0.01));
    }
    // no sun in front of the scene's own lights
    assert_eq!(sky[9 * 4 + 3], 1.0);
    // every mip of the environment holds c, the mirror and the rough ones alike
    let environment = env.environment_texture();
    for mip in 0..environment.mip_level_count() {
        let (_, texels) = read_mip(&device, &queue, environment, mip);
        for t in &texels {
            for ch in 0..3 {
                assert!(close(t[ch], c[ch], 0.01), "mip {mip}: {t:?}");
            }
        }
    }
}

#[test]
fn the_intensity_and_the_tint_scale_it_and_the_rotation_turns_minus_z_toward_plus_x() {
    let Some((device, queue)) = gpu() else { return };
    // a dim image with a bright band down its middle column: the light along -z
    let (w, h) = (256u32, 128u32);
    let pixels = (0..w * h).map(|i| if (i % w).abs_diff(w / 2) < 4 { [100.0; 3] } else { [0.1; 3] }).collect();
    let image = HdrImage::from_pixels(w, h, pixels);
    let mut env = HdrEnvironment::new(&device, &queue, &image, HdrEnvironmentOptions::default());
    let centre = |env: &HdrEnvironment, face: usize| {
        let (size, texels) = read_mip(&device, &queue, env.radiance_texture(), 0);
        texels[face * (size * size) as usize + (size / 2 * size + size / 2) as usize]
    };
    // faces: +X, -X, +Y, -Y, +Z, -Z
    env.prepare(&device, &queue);
    let (bright, dim) = (centre(&env, 5), centre(&env, 0));
    assert!(bright[0] > 50.0 && dim[0] < 1.0, "-z {bright:?}, +x {dim:?}");
    env.set_settings(HdrEnvironmentSettings { rotation: std::f32::consts::FRAC_PI_2, intensity: 2.0, tint: [1.0, 0.5, 0.25] });
    assert!(env.prepare(&device, &queue));
    let turned = centre(&env, 0);
    assert!(close(turned[0], 2.0 * bright[0], 0.02) && close(turned[1], bright[1], 0.02) && close(turned[2], 0.5 * bright[2], 0.02), "+x {turned:?}");
    assert!(centre(&env, 5)[0] < 1.0);
}
