//! Builds the atmosphere LUTs on a real GPU and checks the values that matter for the look: the
//! transmittance LUT against the CPU model, and the sky's colour and brightness from noon to dusk.
//! Skipped (passes) when no adapter is available.

use kansei_core::atmosphere::{direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions};
use kansei_core::cameras::Camera;
use kansei_core::math::Vec3;

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

struct Lut {
    width: u32,
    texels: Vec<[f32; 4]>,
}

impl Lut {
    fn at(&self, x: u32, y: u32) -> glam::Vec3 {
        let t = self.texels[(y * self.width + x) as usize];
        glam::Vec3::new(t[0], t[1], t[2])
    }
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Lut {
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        texture.size(),
    );
    queue.submit(std::iter::once(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer.slice(..).get_mapped_range();
    let mut texels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let o = (y * row + x * 8) as usize;
            let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
            texels.push([c(0), c(1), c(2), c(3)]);
        }
    }
    Lut { width: w, texels }
}

/// The sky-view LUT with the sun at `elevation` degrees, bearing 180 (azimuth u = 0.25).
fn sky_view(device: &wgpu::Device, queue: &wgpu::Queue, sky: &mut SkyAtmosphere, elevation: f32) -> Lut {
    sky.sun.direction = direction_from_elevation_bearing(elevation, 180.0);
    let mut camera = Camera::new(60.0, 0.1, 1000.0, 1.0);
    camera.set_position(0.0, 2.0, 0.0);
    camera.look_at(&Vec3::new(0.0, 2.0, -10.0));
    sky.update(device, queue, &mut camera);
    read(device, queue, &sky.lut_textures()[2])
}

#[test]
fn atmosphere_luts_match_the_cpu_model_and_the_sky_from_noon_to_dusk() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let options = SkyAtmosphereOptions::default();
    let mut sky = SkyAtmosphere::new(&device, options);

    // noon-ish
    let noon = sky_view(&device, &queue, &mut sky, 60.0);

    // transmittance texel (0, 0) is the ground looking straight up, in both parameterisations
    let transmittance = read(&device, &queue, &sky.lut_textures()[0]);
    let gpu_zenith = transmittance.at(0, 0);
    let cpu_zenith = sky.params.transmittance_to_space(0.0, 1.0);
    assert!(((gpu_zenith - cpu_zenith) / cpu_zenith).abs().max_element() < 5e-3, "{gpu_zenith} vs {cpu_zenith}");

    // multiple scattering is a fraction of single scattering, never negative or runaway
    let ms = read(&device, &queue, &sky.lut_textures()[1]);
    assert!(ms.texels.iter().all(|t| t[..3].iter().all(|c| c.is_finite() && *c >= 0.0 && *c < 1.0)));

    let (w, h) = (options.sky_view_size.0, options.sky_view_size.1);
    let (toward_sun, away) = (w / 4, 3 * w / 4);
    let zenith = noon.at(0, 0);
    // a clear day: the zenith is blue, and brighter than the dusk sky by far
    assert!(zenith.z > zenith.y && zenith.y > zenith.x, "noon zenith {zenith}");
    assert!(zenith.z > 1e-3 && zenith.z < 1.0, "noon zenith {zenith}");
    assert!(noon.texels.iter().all(|t| t.iter().all(|c| c.is_finite() && *c >= 0.0)));

    // sunset: the horizon toward the sun is brighter and redder than the one opposite it
    let sunset = sky_view(&device, &queue, &mut sky, 1.0);
    let horizon = h / 2 - 1;
    let (hs, ha) = (sunset.at(toward_sun, horizon), sunset.at(away, horizon));
    assert!(hs.x > ha.x, "sunset horizon toward the sun {hs}, away {ha}");
    assert!(hs.x / hs.z > ha.x / ha.z, "sunset horizon toward the sun {hs}, away {ha}");

    // dusk, the Midsommar intro's sun 2.5 degrees below the horizon: the planet shadows the low
    // air, but the sky is still lit from above it, blue at the zenith and far dimmer than noon
    let dusk = sky_view(&device, &queue, &mut sky, -2.5);
    let dz = dusk.at(0, 0);
    assert!(dz.max_element() > 0.0, "dusk zenith {dz}");
    assert!(dz.z > dz.x, "dusk zenith {dz}");
    assert!(dz.z < zenith.z * 0.05, "dusk zenith {dz} vs noon {zenith}");
    let dusk_glow = dusk.at(toward_sun, horizon);
    assert!(dusk_glow.max_element() > dusk.at(away, horizon).max_element(), "dusk horizon {dusk_glow}");
    eprintln!("noon zenith {zenith}; sunset horizon toward/away {hs} / {ha}; dusk zenith {dz}, horizon toward sun {dusk_glow}");
}
