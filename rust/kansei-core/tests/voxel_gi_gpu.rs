//! Voxel GI on a real GPU: the particle splat conserves mass and emission, the 3D mips keep the
//! volume's mean, a cone through an empty volume sees the whole sky, a cone into an opaque slab
//! sees nothing past it, and a glowing particle lights its neighbour.
//! Skipped (passes) when no adapter is available.

use kansei_core::gi::{
    gradient_sky_lighting, GiBox, ParticleConeSettings, ParticleConeShading, ParticleEmission, ParticleGi, ParticleGiOptions,
    ParticleSplatSettings, ParticleVoxelizer, VoxelGiQuality, VoxelVolume, PARTICLE_LIGHTING_STRIDE,
};
use wgpu::util::DeviceExt;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

fn storage(device: &wgpu::Device, data: &[[f32; 4]]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
    })
}

fn sky(device: &wgpu::Device, up: [f32; 3], down: [f32; 3]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&gradient_sky_lighting(up, down)),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn read_buffer<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, encoder: wgpu::CommandEncoder, buffer: &wgpu::Buffer) -> Vec<T> {
    let mut encoder = encoder;
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    data
}

fn f16_bits(v: f32) -> u16 {
    // normal numbers and zero only, which is all these tests write
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((bits >> 16) & 0x8000) as u16 | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

fn f16_value(h: u16) -> f32 {
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Every texel of one mip of an rgba16float 3D texture, as rgba f32.
fn read_mip(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, level: u32) -> Vec<[f32; 4]> {
    let size = texture.size();
    let [w, h, d] = [size.width, size.height, size.depth_or_array_layers].map(|s| (s >> level).max(1));
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    let mut texels = Vec::new();
    for z in 0..d {
        for y in 0..h {
            let start = ((z * h + y) * row) as usize;
            let halves: &[u16] = bytemuck::cast_slice(&bytes[start..start + (w * 8) as usize]);
            texels.extend(halves.chunks(4).map(|c| [0, 1, 2, 3].map(|i| f16_value(c[i]))));
        }
    }
    texels
}

/// The same PCG hash as particle_emission.wgsl, to know which particles glow.
fn glows(index: u32, e: &ParticleEmission) -> bool {
    let state = (index ^ e.seed).wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    let hash = (word >> 22) ^ word;
    ((hash >> 8) as f32 / 16777216.0) < e.share
}

fn random_points(count: usize, lo: f32, hi: f32) -> Vec<[f32; 4]> {
    let mut rng = 0x2545_f491_u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        lo + (rng >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
    };
    (0..count).map(|_| [next(), next(), next(), 1.0]).collect()
}

#[test]
fn the_splat_conserves_every_particles_density_and_emission() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    // 16 voxels of 0.25 m; particles anywhere inside, some across voxel boundaries
    let volume = VoxelVolume::new(&device, [0.0; 3], [4.0; 3], 16, 1.0);
    let count = 1000;
    let positions = storage(&device, &random_points(count, 0.2, 3.8));
    let sky = sky(&device, [1.0; 3], [1.0; 3]);
    let voxelizer = ParticleVoxelizer::new(&device, &volume, &positions, None, &sky);
    let emission = ParticleEmission { color: [2.0, 1.0, 0.5], share: 0.25, ..Default::default() };
    voxelizer.set_emission(&queue, emission);
    let glowing = (0..count as u32).filter(|&i| glows(i, &emission)).count() as f64;
    assert!(glowing > 150.0 && glowing < 350.0, "{glowing} of {count} glow");

    // trilinear (radius under a voxel), then a footprint 2.4 voxels wide
    for radius in [0.0, 0.6] {
        let settings = ParticleSplatSettings { particle_radius_m: radius, ..Default::default() };
        let mut encoder = device.create_command_encoder(&Default::default());
        voxelizer.encode_splat(&queue, &mut encoder, count as u32, &settings);
        let accum: Vec<u32> = read_buffer(&device, &queue, encoder, voxelizer.accumulators());
        let sum = |c: usize| accum.chunks(4).map(|v| v[c] as f64).sum::<f64>();
        let density = sum(3) / 4096.0;
        assert!((density - count as f64).abs() < count as f64 * 2e-3, "radius {radius}: density {density} for {count} particles");
        // emission is stored times density, 1 per particle
        for (c, color) in emission.color.iter().enumerate() {
            let expected = glowing * *color as f64;
            let got = sum(c) / 1024.0;
            assert!((got - expected).abs() < expected * 5e-3, "radius {radius}: channel {c} {got}, expected {expected}");
        }
        // resolve clears the accumulators for the next frame
        let mut encoder = device.create_command_encoder(&Default::default());
        voxelizer.encode_resolve(&mut encoder);
        let accum: Vec<u32> = read_buffer(&device, &queue, encoder, voxelizer.accumulators());
        assert!(accum.iter().all(|&v| v == 0));
    }
}

#[test]
fn every_mip_keeps_the_volumes_mean() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    // 32 x 16 x 8 voxels, 6 mips
    let volume = VoxelVolume::new(&device, [0.0; 3], [4.0, 2.0, 1.0], 32, 1.0);
    assert_eq!(volume.dims(), [32, 16, 8]);
    assert_eq!(volume.mip_count(), 6);
    let voxels = 32 * 16 * 8;
    // values k / 64, exact in f16
    let mut rng = 7u32;
    let data: Vec<u16> = (0..voxels * 4)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            f16_bits((rng >> 26) as f32 / 64.0)
        })
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: volume.texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(32 * 8), rows_per_image: Some(16) },
        wgpu::Extent3d { width: 32, height: 16, depth_or_array_layers: 8 },
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    volume.build_mips(&mut encoder);
    queue.submit(Some(encoder.finish()));

    let mean = |texels: &[[f32; 4]]| -> [f64; 4] { std::array::from_fn(|c| texels.iter().map(|t| t[c] as f64).sum::<f64>() / texels.len() as f64) };
    let base = mean(&read_mip(&device, &queue, volume.texture(), 0));
    for level in 1..volume.mip_count() {
        let texels = read_mip(&device, &queue, volume.texture(), level);
        let m = mean(&texels);
        for c in 0..4 {
            assert!((m[c] - base[c]).abs() < 2e-3, "mip {level} channel {c}: mean {} vs {}", m[c], base[c]);
        }
    }
    assert_eq!(read_mip(&device, &queue, volume.texture(), 5).len(), 1, "the last mip is one voxel");
}

fn shade(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &VoxelVolume,
    voxelizer: &ParticleVoxelizer,
    shading: &mut ParticleConeShading,
    particles: u32,
    to_sun: [f32; 3],
) -> Vec<[f32; 4]> {
    let mut encoder = device.create_command_encoder(&Default::default());
    // no particle density: the volume holds the boxes alone
    let splat = ParticleSplatSettings { density_per_particle: 0.0, to_sun, sun_illuminance: [0.0; 3], box_sky_scale: 0.0, ..Default::default() };
    voxelizer.encode(queue, &mut encoder, particles, &splat);
    volume.build_mips(&mut encoder);
    shading.reset_history();
    let cones = ParticleConeSettings { to_sun, jitter_voxels: 0.0, max_steps: VoxelGiQuality::High.cone_steps(), ..Default::default() };
    shading.encode(queue, &mut encoder, particles, &cones);
    read_buffer(device, queue, encoder, shading.lighting_buffer())
}

#[test]
fn a_cone_through_an_empty_volume_sees_the_whole_sky() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let volume = VoxelVolume::new(&device, [0.0; 3], [4.0; 3], 32, 1.0);
    let positions = storage(&device, &[[2.0, 2.0, 2.0, 1.0], [0.3, 3.5, 1.0, 1.0]]);
    let sky_color = [0.3, 0.6, 0.9];
    let sky = sky(&device, sky_color, sky_color);
    let voxelizer = ParticleVoxelizer::new(&device, &volume, &positions, None, &sky);
    voxelizer.set_emission(&queue, ParticleEmission::default());
    let mut shading = ParticleConeShading::new(&device, &volume, &positions, None, voxelizer.emission_buffer(), &sky, 2);
    let lighting = shade(&device, &queue, &volume, &voxelizer, &mut shading, 2, [0.3, 1.0, 0.2]);
    assert_eq!(lighting.len() as u64, 2 * PARTICLE_LIGHTING_STRIDE / 16);
    for p in 0..2 {
        let [r, g, b, sun] = lighting[2 * p];
        for (got, want) in [r, g, b].into_iter().zip(sky_color) {
            assert!((got - want).abs() < 1e-4, "particle {p}: incoming {:?}, sky {sky_color:?}", lighting[2 * p]);
        }
        assert_eq!(sun, 1.0, "particle {p} sees the sun");
        assert_eq!(lighting[2 * p + 1], [0.0; 4], "particle {p} does not glow");
    }
}

#[test]
fn a_cone_into_an_opaque_slab_sees_nothing_past_it() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    // a black slab filling x > 2.5 of a 4 m volume of 0.125 m voxels; the particle 4 voxels
    // short of it
    let volume = VoxelVolume::new(&device, [0.0; 3], [4.0; 3], 32, 1.0);
    let positions = storage(&device, &[[2.0, 2.0, 2.0, 1.0]]);
    let sky = sky(&device, [1.0; 3], [1.0; 3]);
    let mut voxelizer = ParticleVoxelizer::new(&device, &volume, &positions, None, &sky);
    voxelizer.set_emission(&queue, ParticleEmission::default());
    voxelizer.set_boxes(&queue, &[GiBox::new([2.5, -1.0, -1.0], [5.0, 5.0, 5.0], [0.0; 3], [-1.0, 0.0, 0.0])]);
    let mut shading = ParticleConeShading::new(&device, &volume, &positions, None, voxelizer.emission_buffer(), &sky, 1);

    let toward = shade(&device, &queue, &volume, &voxelizer, &mut shading, 1, [1.0, 0.0, 0.0]);
    assert!(toward[0][3] < 0.02, "sun behind the slab: visibility {}", toward[0][3]);
    // the +x cone gathers nothing (the slab is black and hides the sky), the -x one the whole
    // sky, and the side cones some of each
    let incoming = toward[0][0];
    assert!(incoming > 0.5 && incoming < 5.0 / 6.0 + 1e-3, "incoming {incoming}");

    let away = shade(&device, &queue, &volume, &voxelizer, &mut shading, 1, [-1.0, 0.0, 0.0]);
    assert!(away[0][3] > 0.99, "sun away from the slab: visibility {}", away[0][3]);
}

#[test]
fn a_glowing_particle_lights_its_neighbour() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    // particle 0 glows (share 1 would light both: pick by index with the hash), particle 1 sits
    // 3 voxels away along +x in the dark
    let options = ParticleGiOptions { quality: VoxelGiQuality::Low, bounds_min: [0.0; 3], bounds_max: [4.0; 3], capacity: 2, ..Default::default() };
    let positions = storage(&device, &[[2.0, 2.0, 2.0, 1.0], [2.1875, 2.0, 2.0, 1.0]]);
    let mut gi = ParticleGi::new(&device, options, &positions, None);
    assert_eq!(gi.quality(), VoxelGiQuality::Low);
    gi.set_sky_gradient(&queue, [0.0; 3], [0.0; 3]);
    let mut emission = ParticleEmission { color: [8.0, 4.0, 2.0], share: 0.5, ..Default::default() };
    // a seed for which particle 0 glows and particle 1 does not
    emission.seed = (0..u32::MAX).find(|&s| glows(0, &ParticleEmission { seed: s, ..emission }) && !glows(1, &ParticleEmission { seed: s, ..emission })).unwrap();
    gi.settings.emission = emission;
    gi.settings.splat.density_per_particle = 4.0;
    gi.settings.cones.jitter_voxels = 0.0;
    gi.settings.cones.start_voxels = 1.0;
    let mut encoder = device.create_command_encoder(&Default::default());
    gi.encode(&queue, &mut encoder, 2);
    let lighting: Vec<[f32; 4]> = read_buffer(&device, &queue, encoder, gi.lighting_buffer());
    assert_eq!(lighting[1], [8.0, 4.0, 2.0, 0.0], "particle 0 glows");
    assert_eq!(lighting[3], [0.0; 4], "particle 1 does not");
    // the sky is black and nothing else glows: all particle 1 receives is particle 0's colour
    let [r, g, b, _] = lighting[2];
    assert!(r > 1e-3 && (r - 2.0 * g).abs() < r * 1e-2 && (g - 2.0 * b).abs() < g * 1e-2, "particle 1 receives particle 0's light: {:?}", lighting[2]);
}
