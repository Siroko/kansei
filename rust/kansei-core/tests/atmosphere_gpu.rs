//! Builds the atmosphere LUTs on a real GPU and checks the values that matter for the look: the
//! transmittance LUT against the CPU model, and the sky's colour and brightness from noon to dusk.
//! Skipped (passes) when no adapter is available.

use kansei_core::atmosphere::{direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions, SkyCaptureFog, SkyLowerHemisphere};
use kansei_core::postprocessing::effects::HeightFogLayer;
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
    height: u32,
    texels: Vec<[f32; 4]>,
}

impl Lut {
    fn at(&self, x: u32, y: u32) -> glam::Vec3 {
        self.at3(x, y, 0)
    }

    fn at3(&self, x: u32, y: u32, z: u32) -> glam::Vec3 {
        let t = self.texels[((z * self.height + y) * self.width + x) as usize];
        glam::Vec3::new(t[0], t[1], t[2])
    }
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Lut {
    read_mip(device, queue, texture, 0)
}

fn read_mip(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, mip: u32) -> Lut {
    let (w, h, d) = ((texture.width() >> mip).max(1), (texture.height() >> mip).max(1), texture.depth_or_array_layers());
    let row = (w * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * h * d) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
    );
    queue.submit(std::iter::once(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer.slice(..).get_mapped_range();
    let mut texels = Vec::with_capacity((w * h * d) as usize);
    for y in 0..h * d {
        for x in 0..w {
            let o = (y * row + x * 8) as usize;
            let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
            texels.push([c(0), c(1), c(2), c(3)]);
        }
    }
    Lut { width: w, height: h, texels }
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

#[test]
fn aerial_perspective_matches_the_optical_depth_along_its_rays() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let options = SkyAtmosphereOptions::default();
    let mut sky = SkyAtmosphere::new(&device, options);
    let _ = sky_view(&device, &queue, &mut sky, 20.0);
    let scattering = read(&device, &queue, &sky.aerial_perspective_textures()[0]);
    let transmittance = read(&device, &queue, &sky.aerial_perspective_textures()[1]);
    let (w, h, d) = options.aerial_perspective_size;
    let (x, y) = (w / 2, h / 2);

    // the ray through that column: the camera looks along -Z with a 60 degree vertical fov
    let ndc = glam::Vec2::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0);
    let t = 30f32.to_radians().tan();
    let rd = glam::Vec3::new(ndc.x * t, ndc.y * t, -1.0).normalize();
    // from the camera, clamped 5 m above the ground as the LUTs are
    let bottom = sky.params.bottom_radius_km;
    let ro = glam::Vec3::new(0.0, bottom + 0.005, 0.0);
    let distance = options.aerial_perspective_distance_km;
    let steps = 4000;
    let mut depth = glam::Vec3::ZERO;
    for i in 0..steps {
        let p = ro + rd * ((i as f32 + 0.5) / steps as f32 * distance);
        depth += sky.params.extinction_at(p.length() - bottom) * (distance / steps as f32);
    }
    let expected = (-depth).exp();
    let last = transmittance.at3(x, y, d - 1);
    assert!(((last - expected) / expected).abs().max_element() < 0.02, "{last} vs {expected}");

    // front to back, the air only adds light and only removes transmittance
    for z in 1..d {
        let (s0, s1) = (scattering.at3(x, y, z - 1), scattering.at3(x, y, z));
        let (t0, t1) = (transmittance.at3(x, y, z - 1), transmittance.at3(x, y, z));
        assert!(s1.cmpge(s0 * 0.999).all() && t1.cmple(t0 * 1.001).all(), "slice {z}: {s0} -> {s1}, {t0} -> {t1}");
    }
    // distant haze is blue by day
    let far = scattering.at3(x, y, d - 1);
    assert!(far.z > far.x, "{far}");
    eprintln!("aerial perspective at {distance} km: scattering {far}, transmittance {last} (cpu {expected})");
}

fn read_floats(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let floats = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
    floats
}

/// Irradiance from order-2 SH (Ramamoorthi and Hanrahan), as `skyIrradiance` in WGSL.
fn sh_irradiance(sh: &[f32], n: glam::Vec3) -> glam::Vec3 {
    let c = |i: usize| glam::Vec3::new(sh[i * 4], sh[i * 4 + 1], sh[i * 4 + 2]);
    let (x, y, z) = (n.x, n.y, n.z);
    c(0) * 0.282095 * std::f32::consts::PI
        + (c(1) * y + c(2) * z + c(3) * x) * 0.488603 * (2.0 * std::f32::consts::PI / 3.0)
        + (c(4) * (1.092548 * x * y) + c(5) * (1.092548 * y * z) + c(6) * (0.315392 * (3.0 * z * z - 1.0))
            + c(7) * (1.092548 * x * z) + c(8) * (0.546274 * (x * x - y * y)))
            * (std::f32::consts::PI / 4.0)
}

#[test]
fn sky_lighting_sh_matches_the_sky_it_projects() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let options = SkyAtmosphereOptions::default();
    let mut sky = SkyAtmosphere::new(&device, options);
    sky.sun.illuminance = Vec3::new(1.0, 1.0, 1.0);

    for elevation in [35.0f32, 3.0, -2.5] {
        // no ground bounce: the SH is the sky alone
        sky.sky_light_ground_albedo = Some(Vec3::ZERO);
        let lut = sky_view(&device, &queue, &mut sky, elevation);
        let sh = read_floats(&device, &queue, &sky.bindings().sky_lighting);

        // integrate cos-weighted radiance over the upper hemisphere straight from the LUT texels
        let bottom = sky.params.bottom_radius_km;
        let r = bottom + 0.005;
        let theta_h = (-((r - bottom) * (r + bottom)).sqrt() / r).acos();
        let zenith = |v: f32| {
            if v < 0.5 {
                let c = 1.0 - 2.0 * v;
                theta_h * (1.0 - c * c)
            } else {
                let c = 2.0 * v - 1.0;
                theta_h + (std::f32::consts::PI - theta_h) * c * c
            }
        };
        let (w, h) = options.sky_view_size;
        // and over the lower hemisphere onto a downward-facing surface (the air below the horizon),
        // and project the texels onto SH as the GPU pass should
        let (mut e_up, mut e_air_down) = (glam::Vec3::ZERO, glam::Vec3::ZERO);
        let mut sh_cpu = [glam::Vec3::ZERO; 9];
        for j in 0..h {
            let (t0, t1) = (zenith(j as f32 / h as f32), zenith((j + 1) as f32 / h as f32));
            let t = zenith((j as f32 + 0.5) / h as f32);
            for i in 0..w {
                let d_omega = t.sin() * (t1 - t0) * (std::f32::consts::TAU / w as f32);
                let e = lut.at(i, j) * t.cos() * d_omega;
                if t < std::f32::consts::FRAC_PI_2 {
                    e_up += e;
                } else {
                    e_air_down -= e;
                }
                let phi = (i as f32 + 0.5) / w as f32 * std::f32::consts::TAU;
                let d = glam::Vec3::new(t.sin() * phi.cos(), t.cos(), t.sin() * phi.sin());
                let basis = [
                    0.282095,
                    0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
                    1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
                    1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y),
                ];
                for k in 0..9 {
                    sh_cpu[k] += lut.at(i, j) * basis[k] * d_omega;
                }
            }
        }
        let scale = sh_cpu[0].max_element();
        for k in 0..9 {
            let gpu = glam::Vec3::new(sh[k * 4], sh[k * 4 + 1], sh[k * 4 + 2]);
            assert!((gpu - sh_cpu[k]).abs().max_element() < 0.03 * scale, "sun {elevation}: SH[{k}] GPU {gpu} vs CPU {}", sh_cpu[k]);
        }
        // order-2 SH fits the clamped cosine to about 10% for a sky peaked at the horizon
        let from_sh = sh_irradiance(&sh, glam::Vec3::Y);
        assert!(((from_sh - e_up) / e_up).abs().max_element() < 0.12, "sun {elevation}: SH {from_sh} vs LUT {e_up}");
        assert!(sh_irradiance(&sh, -glam::Vec3::Y).max_element() < 0.2 * e_up.max_element(), "sun {elevation}: lower hemisphere without ground");

        // the sun at the camera agrees with the CPU helper
        let sun = glam::Vec3::new(sh[36], sh[37], sh[38]);
        let cpu = sky.sun_illuminance_at(Vec3::new(0.0, 5.0, 0.0)).to_glam();
        assert!((sun - cpu).abs().max_element() <= 0.01 * cpu.max_element().max(1e-4), "sun {elevation}: GPU {sun} vs CPU {cpu}");

        // with a ground, the downward-facing irradiance is the ground's bounce: albedo x (sky + sun)
        sky.sky_light_ground_albedo = Some(Vec3::new(0.3, 0.3, 0.3));
        let _ = sky_view(&device, &queue, &mut sky, elevation);
        let sh = read_floats(&device, &queue, &sky.bindings().sky_lighting);
        let ground_e = e_up + cpu * sky.sun.direction.to_glam().y.max(0.0);
        let down = sh_irradiance(&sh, -glam::Vec3::Y);
        let expected = 0.3 * ground_e + e_air_down;
        assert!(((down - expected) / expected).abs().max_element() < 0.2, "sun {elevation}: down {down} vs {expected}");
        eprintln!("sun {elevation}: E_up SH {from_sh} LUT {e_up}; sun {sun}; E_down {down}");
    }
}

/// With a capture fog the sky lighting sees the sky through it (Unreal's real-time capture): an
/// opaque fog is its colour all round, one capped at half opacity is half the sky and half the fog
/// from every side (the projection is linear), and the environment cubemap agrees. By default the
/// lighting sees the lit ground below the horizon instead, as Unreal's Lumen does, while the
/// environment keeps the fog there. A lower hemisphere colour replaces what is below the horizon.
#[test]
fn sky_lighting_captures_the_height_fog() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let options = SkyAtmosphereOptions::default();
    let mut sky = SkyAtmosphere::new(&device, options);
    sky.sun.illuminance = Vec3::new(1.0, 1.0, 1.0);
    // no ground bounce, so the clear SH is the sky alone
    sky.sky_light_ground_albedo = Some(Vec3::ZERO);
    let irradiance = |sky: &mut SkyAtmosphere| {
        let _ = sky_view(&device, &queue, sky, 20.0);
        let sh = read_floats(&device, &queue, &sky.bindings().sky_lighting);
        [glam::Vec3::Y, -glam::Vec3::Y, glam::Vec3::X].map(|n| sh_irradiance(&sh, n))
    };
    let clear = irradiance(&mut sky);
    let c = Vec3::new(0.02, 0.03, 0.05);
    let pi_c = c.to_glam() * std::f32::consts::PI;
    let dense = HeightFogLayer { density: 1.0, height_falloff: 1e-4, height: 0.0 };
    let fog = |max_opacity: f32| SkyCaptureFog { layers: [dense, HeightFogLayer::default()], inscattering: c, max_opacity, capture_height_m: 6.0, sky_ambient_scale: 0.0 };

    // the capture itself: the fog below the horizon too
    sky.lighting_sees_ground = false;
    sky.capture_fog = Some(fog(1.0));
    let opaque = irradiance(&mut sky);
    for e in opaque {
        assert!(((e - pi_c) / pi_c).abs().max_element() < 0.01, "an opaque fog gives {e}, not pi C {pi_c}");
    }
    // the environment is the fog's colour, overhead and below
    let env = read(&device, &queue, sky.environment_texture());
    let centre = options.environment_size / 2;
    for layer in [2, 3] {
        let l = env.at3(centre, centre, layer);
        assert!(((l - c.to_glam()) / c.to_glam()).abs().max_element() < 0.01, "environment {l} under an opaque fog of {c:?}");
    }

    sky.capture_fog = Some(fog(0.5));
    let half = irradiance(&mut sky);
    for (e, e_clear) in half.iter().zip(clear) {
        let want = 0.5 * e_clear + 0.5 * pi_c;
        assert!(((*e - want) / want).abs().max_element() < 0.01, "half the fog gives {e}, not {want}");
    }

    // by default, the lit ground below the horizon (a Lambertian of albedo 0.3 under the fog's pi
    // C and the sun, which the capture's fog does not dim), the fog above it; the environment
    // still sees the fog below
    sky.lighting_sees_ground = true;
    sky.sky_light_ground_albedo = Some(Vec3::new(0.3, 0.3, 0.3));
    sky.capture_fog = Some(fog(1.0));
    let [up, down, _] = irradiance(&mut sky);
    let sh = read_floats(&device, &queue, &sky.bindings().sky_lighting);
    let sun = glam::Vec3::new(sh[36], sh[37], sh[38]) * sky.sun.direction.to_glam().y.max(0.0);
    let ground = 0.3 * (pi_c + sun);
    assert!(((up - pi_c) / pi_c).abs().max_element() < 0.1, "up {up} under the fog vs {pi_c}");
    assert!(((down - ground) / ground).abs().max_element() < 0.1, "down {down} from the ground vs {ground}");
    let env = read(&device, &queue, sky.environment_texture());
    let l = env.at3(centre, centre, 3);
    assert!(((l - c.to_glam()) / c.to_glam()).abs().max_element() < 0.01, "environment below {l} under the fog {c:?}");
    eprintln!("the ground under the fog: up {up} down {down}");
    sky.sky_light_ground_albedo = Some(Vec3::ZERO);

    // Unreal's lower-hemisphere colour, without fog: the ground is gone, the colour is below
    sky.capture_fog = None;
    let low = Vec3::new(0.2, 0.3, 0.4);
    sky.lower_hemisphere = SkyLowerHemisphere::Color(low);
    let [up, down, _] = irradiance(&mut sky);
    let want = low.to_glam() * std::f32::consts::PI;
    assert!(((down - want) / want).abs().max_element() < 0.1, "down {down} with the lower hemisphere {want}");
    assert!(((up - clear[0]) / clear[0]).abs().max_element() < 0.1, "up {up} vs {}", clear[0]);
    eprintln!("clear {clear:?}; opaque fog {opaque:?} (pi C {pi_c}); half {half:?}; lower colour: up {up} down {down}");
}

/// The sky's distant light (Unreal's: the mean radiance all round from 6 km above the ground) is
/// the sphere mean of the sky-view LUT seen from 6 km up, at noon and at twilight, whatever the
/// camera's height.
#[test]
fn distant_sky_light_is_the_sky_s_mean_radiance_from_6_km() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let options = SkyAtmosphereOptions::default();
    let mut sky = SkyAtmosphere::new(&device, options);
    sky.sun.illuminance = Vec3::new(1.0, 1.0, 1.0);
    let bottom = sky.params.bottom_radius_km;
    for elevation in [40.0f32, -2.5] {
        sky.sun.direction = direction_from_elevation_bearing(elevation, 320.0);
        // the distant light, from a camera on the ground
        let mut camera = Camera::new(60.0, 0.1, 1000.0, 1.0);
        camera.set_position(0.0, 2.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 2.0, -10.0));
        sky.update(&device, &queue, &mut camera);
        let sh = read_floats(&device, &queue, &sky.bindings().sky_lighting);
        let d = glam::Vec3::new(sh[56], sh[57], sh[58]);
        // the sky-view LUT from 6 km up, averaged over the sphere with its texels' solid angles
        camera.set_position(0.0, 6000.0, 0.0);
        sky.update(&device, &queue, &mut camera);
        let lut = read(&device, &queue, &sky.lut_textures()[2]);
        let r = bottom + 6.0;
        let theta_h = (-((r - bottom) * (r + bottom)).sqrt() / r).acos();
        let zenith = |v: f32| {
            if v < 0.5 {
                let c = 1.0 - 2.0 * v;
                theta_h * (1.0 - c * c)
            } else {
                let c = 2.0 * v - 1.0;
                theta_h + (std::f32::consts::PI - theta_h) * c * c
            }
        };
        let (w, h) = options.sky_view_size;
        let mut sum = glam::Vec3::ZERO;
        for j in 0..h {
            let (t0, t1) = (zenith(j as f32 / h as f32), zenith((j + 1) as f32 / h as f32));
            let t = zenith((j as f32 + 0.5) / h as f32);
            for i in 0..w {
                sum += lut.at(i, j) * (t.sin() * (t1 - t0) * (std::f32::consts::TAU / w as f32));
            }
        }
        let mean = sum / (4.0 * std::f32::consts::PI);
        eprintln!("sun {elevation}: distant sky light {d}, sky-view mean from 6 km {mean}");
        assert!(((d - mean) / mean).abs().max_element() < 0.1, "sun {elevation}: {d} vs {mean}");
    }
}

#[test]
fn environment_cubemap_faces_follow_the_webgpu_layout_and_the_sky() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let options = SkyAtmosphereOptions::default();
    let mut sky = SkyAtmosphere::new(&device, options);
    // a low sun in the east (+X)
    sky.sun.direction = direction_from_elevation_bearing(4.0, 90.0);
    let mut camera = Camera::new(60.0, 0.1, 1000.0, 1.0);
    camera.set_position(0.0, 2.0, 0.0);
    camera.look_at(&Vec3::new(0.0, 2.0, -10.0));
    sky.update(&device, &queue, &mut camera);
    let env = read(&device, &queue, sky.environment_texture());
    let lut = read(&device, &queue, &sky.lut_textures()[2]);
    let c = options.environment_size / 2;
    // layers +X, -X, +Y, -Y, +Z, -Z; the face centres look along the axes (t runs down the side
    // faces, so two rows above the centre look 2 degrees above the horizon)
    let face = |layer: u32| env.at3(c, if layer == 2 || layer == 3 { c } else { c - 2 }, layer);
    let (east, west, zenith, nadir) = (face(0), face(1), face(2), face(3));
    assert!(east.x > 3.0 * west.x, "east {east}, west {west}");
    assert!(zenith.z > zenith.x, "zenith {zenith}");
    // the zenith face centre is the sky-view LUT's top row
    let lut_zenith = lut.at(0, 0);
    assert!(((zenith - lut_zenith) / lut_zenith).abs().max_element() < 0.03, "{zenith} vs {lut_zenith}");
    // below the horizon: the ground's bounce, far dimmer than the sky toward the sun
    assert!(nadir.max_element() > 0.0 && nadir.max_element() < east.max_element());
    // +Z (south) and -Z (north) are symmetric about the east-west sun plane
    let (south, north) = (face(4), face(5));
    assert!(((south - north) / north).abs().max_element() < 0.05, "south {south}, north {north}");
    // the prefiltered mips are filled, and the roughest spreads the sunset glow round the cube
    let mips = sky.environment_texture().mip_level_count();
    let last = read_mip(&device, &queue, sky.environment_texture(), mips - 1);
    let (e1, w1) = (last.at3(0, 0, 0), last.at3(0, 0, 1));
    assert!(e1.is_finite() && w1.min_element() > 0.0 && e1.x > w1.x && e1.x / w1.x < east.x / west.x, "rough east {e1}, west {w1}");
    eprintln!("env: east {east} west {west} zenith {zenith} nadir {nadir}; roughest east {e1} west {w1}");
}

/// Per-frame cost of `SkyAtmosphere::update` (sky view, aerial perspective, sky lighting and
/// environment passes): wall time of submit-and-wait, minus that of an empty submit. Run by hand:
/// `cargo test -p kansei-core --release --test atmosphere_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn atmosphere_frame_cost() {
    let Some((device, queue)) = gpu() else { return };
    let mut sky = SkyAtmosphere::new(&device, SkyAtmosphereOptions::default());
    let mut camera = Camera::new(60.0, 0.1, 1000.0, 16.0 / 9.0);
    camera.set_position(0.0, 2.0, 0.0);
    camera.look_at(&Vec3::new(0.0, 2.0, -10.0));
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let time = |f: &mut dyn FnMut()| {
        let mut samples = Vec::new();
        for i in 0..220 {
            let t0 = std::time::Instant::now();
            f();
            device.poll(wgpu::Maintain::Wait);
            if i >= 20 {
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        median(samples)
    };
    let empty = time(&mut || {
        queue.submit(std::iter::once(device.create_command_encoder(&Default::default()).finish()));
    });
    // one frame, then the static LUTs are cached as in a running app
    sky.update(&device, &queue, &mut camera);
    let update = time(&mut || sky.update(&device, &queue, &mut camera));
    sky.invalidate();
    let rebuild = time(&mut || {
        sky.invalidate();
        sky.update(&device, &queue, &mut camera)
    });
    eprintln!(
        "SkyAtmosphere::update: {:.3} ms per frame ({:.3} ms with the static LUTs rebuilt), over an empty submit of {:.3} ms",
        update - empty,
        rebuild - empty,
        empty
    );
}
