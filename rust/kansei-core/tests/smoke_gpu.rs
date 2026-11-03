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

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a < 6.1e-5 {
        // subnormal (or zero)
        return sign | (a / 2f32.powi(-24)).round() as u16;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let mant = bits & 0x7f_ffff;
    // round to nearest
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    if mant & 0x1000 != 0 {
        h += 1;
    }
    sign | h as u16
}

/// Write an rgba16float 3D texture from f32 texels (x fastest).
fn write_rgba16(queue: &wgpu::Queue, texture: &wgpu::Texture, dims: [u32; 3], texels: &[[f32; 4]]) {
    let [w, h, d] = dims;
    let data: Vec<u16> = texels.iter().flat_map(|t| t.map(f32_to_f16)).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
    );
}

/// The MAC grid's divergence per cell, rms (walls: the +faces hold 0), cells `h` metres.
fn rms_mac_div(vel: &[[f32; 4]], dims: [u32; 3], h: [f32; 3]) -> f32 {
    let [w, hh, d] = dims;
    let at = |x: u32, y: u32, z: u32| vel[((z * hh + y) * w + x) as usize];
    let mut sum = 0.0f64;
    for z in 0..d {
        for y in 0..hh {
            for x in 0..w {
                let c = at(x, y, z);
                let u1 = if x + 1 < w { at(x + 1, y, z)[0] } else { 0.0 };
                let v1 = if y + 1 < hh { at(x, y + 1, z)[1] } else { 0.0 };
                let w1 = if z + 1 < d { at(x, y, z + 1)[2] } else { 0.0 };
                let div = (u1 - c[0]) / h[0] + (v1 - c[1]) / h[1] + (w1 - c[2]) / h[2];
                sum += (div * div) as f64;
            }
        }
    }
    (sum / (w * hh * d) as f64).sqrt() as f32
}

/// Total density, and its centre of mass in the box's local metres from its minimum corner.
fn mass_and_com(cells: &[[f32; 2]], dims: [u32; 3], size: Vec3) -> (f32, glam::Vec3) {
    let [w, h, d] = dims;
    let (mut mass, mut moment) = (0.0f64, glam::DVec3::ZERO);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let m = cells[((z * h + y) * w + x) as usize][0] as f64;
                let p = glam::DVec3::new((x as f64 + 0.5) / w as f64 * size.x as f64, (y as f64 + 0.5) / h as f64 * size.y as f64, (z as f64 + 0.5) / d as f64 * size.z as f64);
                mass += m;
                moment += p * m;
            }
        }
    }
    (mass as f32, (moment / mass.max(1e-12)).as_vec3())
}

fn fluid(f: impl FnOnce(&mut FluidSmokeSettings)) -> SmokeMotion {
    let mut s = FluidSmokeSettings::default();
    f(&mut s);
    SmokeMotion::Fluid(s)
}

#[test]
fn the_projection_leaves_no_divergence() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let motion = fluid(|s| {
        (s.buoyancy, s.weight, s.vorticity, s.cooling, s.pressure_iterations) = (0.0, 0.0, 0.0, 0.0, 1000);
    });
    let mut v = SmokeVolume::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 1.0, 1.0), motion);
    v.resolution = 16;
    v.look.dissipation = 0.0;
    let id = sim.add(v);
    let dims = [16u32; 3];
    // v = (p - c) exp(-|p - c|^2 / 0.04) on the faces: x on the -x face, y on -y, z on -z
    let field = |p: glam::Vec3| {
        let q = p - glam::Vec3::splat(0.5);
        q * (-q.length_squared() / 0.04).exp()
    };
    let n = 16.0;
    let mut texels = Vec::new();
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                let (fx, fy, fz) = (x as f32, y as f32, z as f32);
                texels.push([
                    field(glam::Vec3::new(fx / n, (fy + 0.5) / n, (fz + 0.5) / n)).x,
                    field(glam::Vec3::new((fx + 0.5) / n, fy / n, (fz + 0.5) / n)).y,
                    field(glam::Vec3::new((fx + 0.5) / n, (fy + 0.5) / n, fz / n)).z,
                    0.0,
                ]);
            }
        }
    }
    write_rgba16(&queue, sim.grid_textures(id).unwrap().velocity.unwrap(), dims, &texels);
    let before = rms_mac_div(&texels, dims, [1.0 / n; 3]);
    sim.step(1.0 / 60.0, &SmokeMouse::NONE);
    let grid = sim.grid_textures(id).unwrap();
    let after = rms_mac_div(&read_rgba16(&device, &queue, grid.velocity.unwrap(), dims), dims, [1.0 / n; 3]);
    eprintln!("rms divergence: before {before}, after {after}");
    assert!(after < 0.02 * before, "before {before}, after {after}");
}

/// A hot ball of smoke 0.3 m above the floor of a 1 x 1.5 x 1 m box.
fn hot_blob(dissipation: f32) -> SmokeVolume {
    let mut v = SmokeVolume::new(Vec3::new(0.5, 0.75, 0.5), Vec3::new(1.0, 1.5, 1.0), SmokeMotion::fluid());
    v.resolution = 48;
    v.look.dissipation = dissipation;
    v.fill = SmokeFill::Sphere { center: Vec3::new(0.5, 0.3, 0.5), radius: 0.15, density: 1.0, temperature: 1.0 };
    v
}

#[test]
fn a_hot_blob_rises() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let v = hot_blob(SmokeLook::default().dissipation);
    let size = v.size;
    let id = sim.add(v);
    let (cells, dims) = read_scalars(&device, &queue, &sim, id);
    let (_, com0) = mass_and_com(&cells, dims, size);
    for _ in 0..90 {
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
    }
    let (cells, _) = read_scalars(&device, &queue, &sim, id);
    let (_, com) = mass_and_com(&cells, dims, size);
    eprintln!("centre of mass {com0} -> {com}");
    assert!(com.y - com0.y > 0.15, "rose {}", com.y - com0.y);
    assert!((com.x - com0.x).abs() < 0.02 && (com.z - com0.z).abs() < 0.02, "drifted {com0} -> {com}");
}

#[test]
fn the_gas_stays_finite_and_in_its_box() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let v = hot_blob(0.0);
    let size = v.size;
    let id = sim.add(v);
    let (cells, dims) = read_scalars(&device, &queue, &sim, id);
    let (mass0, _) = mass_and_com(&cells, dims, size);
    let max0 = cells.iter().map(|c| c[0]).fold(0.0f32, f32::max);
    for _ in 0..600 {
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
    }
    let (cells, _) = read_scalars(&device, &queue, &sim, id);
    let all_finite = cells.iter().all(|c| c[0].is_finite() && c[1].is_finite());
    let min = cells.iter().map(|c| c[0]).fold(f32::MAX, f32::min);
    let max = cells.iter().map(|c| c[0]).fold(0.0f32, f32::max);
    let (mass, _) = mass_and_com(&cells, dims, size);
    eprintln!("density {min}..{max} (was ..{max0}), mass {mass0} -> {mass}");
    assert!(all_finite && min >= 0.0 && max <= max0 * 1.001, "density {min}..{max}, was ..{max0}");
    assert!((mass / mass0 - 1.0).abs() < 0.10, "mass {mass0} -> {mass}");
}

#[test]
fn the_default_plume_fades_before_the_lid() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let source = SmokeSource { position: Vec3::new(0.0, 0.0, 0.0), radius: 0.03, rate: 4.0, temperature: 1.0, velocity: Vec3::new(0.0, 0.15, 0.0) };
    let v = SmokeVolume::rising_from(source, 0.8, SmokeMotion::fluid());
    let dissipation = v.look.dissipation;
    let id = sim.add(v);
    for _ in 0..1200 {
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
    }
    let (cells, dims) = read_scalars(&device, &queue, &sim, id);
    let [w, h, d] = dims;
    let at = |x: u32, y: u32, z: u32| cells[((z * h + y) * w + x) as usize];
    // the top tenth of the rows, all of each
    let top_rows = h / 10;
    let mut top = 0.0;
    for y in h - top_rows..h {
        for z in 0..d {
            for x in 0..w {
                top += at(x, y, z)[0];
            }
        }
    }
    top /= (top_rows * w * d) as f32;
    // the source's column (its radius), its lower third
    let (cx, cz, r) = (w / 2, d / 2, (0.03 / (0.4 / w as f32)).ceil() as u32);
    let (mut column, mut n) = (0.0, 0);
    for y in 0..h / 3 {
        for z in cz - r..cz + r {
            for x in cx - r..cx + r {
                column += at(x, y, z)[0];
                n += 1;
            }
        }
    }
    column /= n as f32;
    let max_density = cells.iter().map(|c| c[0]).fold(0.0f32, f32::max);
    let max_temperature = cells.iter().map(|c| c[1]).fold(0.0f32, f32::max);
    eprintln!("top rows {top}, source column {column}, max density {max_density}, max temperature {max_temperature}");
    assert!(top < 0.1 * column, "top rows {top}, source column {column}");
    assert!(max_density <= 1.05 * 4.0 / dissipation && max_temperature <= 1.001, "max density {max_density}, max temperature {max_temperature}");
}


#[test]
fn the_curl_noise_velocity_is_divergence_free() {
    let Some((device, queue)) = gpu() else { return };
    // curlNoise and, as a control, the gradient of the potential's x (not divergence-free), on a
    // 32^3 lattice of spacing h (noise units)
    const N: u32 = 32;
    let h = 1.0f32 / 8.0;
    let code = format!(
        "{CURL_NOISE_WGSL}\n\
        @group(0) @binding(0) var<storage, read_write> out : array<vec4f>;\n\
        @compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) g : vec3u) {{\n\
            if (any(g >= vec3u({N}u))) {{ return; }}\n\
            let p = vec3f(g) * {h} + vec3f(0.3, 1.7, -2.1);\n\
            let i = (g.z * {N}u + g.y) * {N}u + g.x;\n\
            out[2u * i] = vec4f(curlNoise(p, 0.37), 0.0);\n\
            let e = 1e-3;\n\
            let gx = (curlNoisePotential(p + vec3f(e, 0.0, 0.0), 0.37).x - curlNoisePotential(p - vec3f(e, 0.0, 0.0), 0.37).x) / (2.0 * e);\n\
            let gy = (curlNoisePotential(p + vec3f(0.0, e, 0.0), 0.37).x - curlNoisePotential(p - vec3f(0.0, e, 0.0), 0.37).x) / (2.0 * e);\n\
            let gz = (curlNoisePotential(p + vec3f(0.0, 0.0, e), 0.37).x - curlNoisePotential(p - vec3f(0.0, 0.0, e), 0.37).x) / (2.0 * e);\n\
            out[2u * i + 1u] = vec4f(gx, gy, gz, 0.0);\n\
        }}"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let size = (N * N * N * 2 * 16) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }] });
    let mut e = device.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(N / 4, N / 4, N / 4);
    }
    e.copy_buffer_to_buffer(&out, 0, &read, 0, size);
    queue.submit([e.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
    let field = |k: usize| -> Vec<[f32; 3]> { (0..(N * N * N) as usize).map(|i| { let v = data[2 * i + k]; [v[0], v[1], v[2]] }).collect() };
    let (curl, gradient) = (field(0), field(1));
    let mean_abs_central_div = |v: &[[f32; 3]], h: f32| {
        let at = |x: u32, y: u32, z: u32| v[((z * N + y) * N + x) as usize];
        let (mut sum, mut n) = (0.0f64, 0);
        for z in 1..N - 1 {
            for y in 1..N - 1 {
                for x in 1..N - 1 {
                    let div = (at(x + 1, y, z)[0] - at(x - 1, y, z)[0] + at(x, y + 1, z)[1] - at(x, y - 1, z)[1] + at(x, y, z + 1)[2] - at(x, y, z - 1)[2]) / (2.0 * h);
                    sum += div.abs() as f64;
                    n += 1;
                }
            }
        }
        (sum / n as f64) as f32
    };
    let mean_len = |v: &[[f32; 3]]| v.iter().map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt() as f64).sum::<f64>() as f32 / v.len() as f32;
    let ratio = |v: &[[f32; 3]]| mean_abs_central_div(v, h) * h / mean_len(v);
    eprintln!("divergence ratio: curl {}, gradient (control) {}; mean speed {}", ratio(&curl), ratio(&gradient), mean_len(&curl));
    assert!(ratio(&curl) < 0.02, "curl {}", ratio(&curl));
    assert!(ratio(&gradient) > 0.2, "control {}", ratio(&gradient));
}

#[test]
fn curl_noise_carries_the_gas_up_and_keeps_it() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    // the plume's cells (0.83 cm): the default noise's eddies, 0.15 m, span about 18 of them
    let mut v = SmokeVolume::new(Vec3::new(0.3, 0.45, 0.3), Vec3::new(0.6, 0.9, 0.6), SmokeMotion::CurlNoise(CurlSettings { rise: 0.2, ..Default::default() }));
    v.resolution = 108;
    v.look.dissipation = 0.0;
    v.fill = SmokeFill::Sphere { center: Vec3::new(0.3, 0.3, 0.3), radius: 0.15, density: 1.0, temperature: 0.0 };
    let size = v.size;
    let id = sim.add(v);
    let (cells, dims) = read_scalars(&device, &queue, &sim, id);
    let (mass0, com0) = mass_and_com(&cells, dims, size);
    for _ in 0..60 {
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
    }
    let (cells, _) = read_scalars(&device, &queue, &sim, id);
    let (mass, com) = mass_and_com(&cells, dims, size);
    eprintln!("centre of mass {com0} -> {com}, mass {mass0} -> {mass}");
    assert!((com.y - com0.y - 0.2).abs() < 0.06, "rose {}", com.y - com0.y);
    assert!((mass / mass0 - 1.0).abs() < 0.10, "mass {mass0} -> {mass}");
}

/// Sheets along y: the number of local maxima of the x-z mean density's profile over y, and the
/// coefficient of variation along x of the middle sheet's band (summed over its rows and z).
fn sheets(density: impl Fn(u32, u32, u32) -> f32, [w, h, d]: [u32; 3]) -> (usize, f32) {
    let profile: Vec<f32> = (0..h).map(|y| (0..d).flat_map(|z| (0..w).map(move |x| (x, z))).map(|(x, z)| density(x, y, z)).sum::<f32>() / (w * d) as f32).collect();
    let max = profile.iter().cloned().fold(0.0f32, f32::max);
    let peaks: Vec<usize> = (1..h as usize - 1).filter(|&y| profile[y] > profile[y - 1] && profile[y] >= profile[y + 1] && profile[y] > 0.1 * max).collect();
    if peaks.len() < 2 {
        return (peaks.len(), f32::NAN);
    }
    // the middle peak's band: to the minima either side
    let mid = peaks[peaks.len() / 2];
    let (mut lo, mut hi) = (mid, mid);
    while lo > 0 && profile[lo - 1] < profile[lo] { lo -= 1; }
    while hi + 1 < h as usize && profile[hi + 1] < profile[hi] { hi += 1; }
    let along_x: Vec<f32> = (0..w).map(|x| (lo as u32..=hi as u32).flat_map(|y| (0..d).map(move |z| (y, z))).map(|(y, z)| density(x, y, z)).sum()).collect();
    let mean = along_x.iter().sum::<f32>() / w as f32;
    let var = along_x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
    (peaks.len(), var.sqrt() / mean)
}

/// A room of hanging smoke: three sheets, no source.
fn layers_room(motion: SmokeMotion) -> SmokeVolume {
    let mut v = SmokeVolume::new(Vec3::new(0.0, 1.5, 0.0), Vec3::new(6.0, 3.0, 5.0), motion);
    v.resolution = 64;
    v.look.dissipation = 0.0;
    v.fill = SmokeFill::Layers { count: 3, thickness: 0.25, density: 0.6 };
    v
}

#[test]
fn layers_fill_lies_in_horizontal_sheets() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let id = sim.add(layers_room(SmokeMotion::fluid()));
    let (cells, dims) = read_scalars(&device, &queue, &sim, id);
    let [w, h, _] = dims;
    let (count, cv) = sheets(|x, y, z| cells[((z * h + y) * w + x) as usize][0], dims);
    eprintln!("{count} sheets, the middle one's variation along x {cv}");
    assert_eq!(count, 3);
    assert!(cv < 0.3, "cv {cv}");
}

#[test]
fn layers_drift_on_fluid_and_do_not_dissipate() {
    let Some((device, queue)) = gpu() else { return };
    let mut sim = SmokeSimulation::new(&device, &queue);
    let motion = fluid(|s| {
        (s.ambient_draft, s.draft_scale, s.weight, s.buoyancy) = (0.05, 1.5, 0.0, 0.0);
    });
    let v = layers_room(motion);
    let size = v.size;
    let id = sim.add(v);
    let (before, dims) = read_scalars(&device, &queue, &sim, id);
    let (mass0, _) = mass_and_com(&before, dims, size);
    for _ in 0..300 {
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
    }
    let (after, _) = read_scalars(&device, &queue, &sim, id);
    let (mass, _) = mass_and_com(&after, dims, size);
    let l1: f32 = before.iter().map(|c| c[0].abs()).sum();
    let change: f32 = before.iter().zip(&after).map(|(a, b)| (a[0] - b[0]).abs()).sum();
    eprintln!("mass {mass0} -> {mass}, L1 change {} of L1", change / l1);
    assert!((mass / mass0 - 1.0).abs() < 0.05, "mass {mass0} -> {mass}");
    assert!(change > 0.05 * l1, "the layers barely moved: {}", change / l1);
}
