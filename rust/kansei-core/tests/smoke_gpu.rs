//! The smoke simulation (`simulations::smoke`) on a real GPU: fills, the Fluid and Curl levels,
//! the mouse. Skipped (passes) when no adapter is available.

use kansei_core::math::Vec3;
use kansei_core::simulations::smoke::*;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter: skipping");
        return None;
    };
    Some(pollster::block_on(adapter.request_device(&Default::default())).unwrap())
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => if frac == 0.0 { f32::INFINITY } else { f32::NAN },
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Every cell of an rgba16float 3D texture, x fastest, then y, then z.
fn read_rgba16(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, dims: [u32; 3]) -> Vec<[f32; 4]> {
    let [w, h, d] = dims;
    let row = (w * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
    );
    queue.submit(std::iter::once(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h * d) as usize);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let o = ((z * h + y) * row + x * 8) as usize;
                let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
                out.push([c(0), c(1), c(2), c(3)]);
            }
        }
    }
    out
}

/// A volume's (density, temperature) per cell.
fn read_scalars(device: &wgpu::Device, queue: &wgpu::Queue, sim: &SmokeSimulation, id: SmokeVolumeId) -> (Vec<[f32; 2]>, [u32; 3]) {
    let grid = sim.grid_textures(id).expect("a grid");
    let dims = grid.dims;
    (read_rgba16(device, queue, grid.scalars, dims).into_iter().map(|c| [c[0], c[1]]).collect(), dims)
}

#[test]
fn a_sphere_fill_fills_its_cells() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut volume = SmokeVolume::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 1.0, 1.0), SmokeMotion::fluid());
    volume.resolution = 32;
    volume.fill = SmokeFill::Sphere { center: Vec3::new(0.5, 0.5, 0.5), radius: 0.25, density: 2.0, temperature: 1.0 };
    let id = sim.add(volume);
    let (cells, dims) = read_scalars(&device, &queue, &sim, id);
    assert_eq!(dims, [32, 32, 32]);
    let at = |x: u32, y: u32, z: u32| cells[((z * 32 + y) * 32 + x) as usize];
    let centre = at(16, 16, 16);
    assert!((centre[0] - 2.0).abs() < 0.01 && (centre[1] - 1.0).abs() < 0.01, "centre {centre:?}");
    assert_eq!(at(2, 2, 2), [0.0, 0.0]);
    let waves = sim.add(SmokeVolume::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), SmokeMotion::waves()));
    assert!(sim.grid_textures(waves).is_none());
}
