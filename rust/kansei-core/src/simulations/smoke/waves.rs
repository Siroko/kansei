//! The Waves level's density, on the CPU: the mirror of WGSL `smokeWavesDensity`
//! (`shaders/smoke_common.wgsl`), which the march evaluates where it samples (no state).
//!
//! Each source sends up a column: Gaussian across, its radius growing by `spread` per metre, its
//! axis swayed by two sines of height and time each way (`amplitude`, `wavelength`, `speed`;
//! no sway at the rim, full sway a wavelength up), its density diluted as it widens, pulsing
//! gently as it rises, and fading with the time the gas takes to rise at its source's speed
//! (`dissipation` per second). Nothing outside the box. A volume with no sources holds its
//! layers fill (`SmokeFill::Layers`), rolling slowly.

use super::volume::{SmokeVolume, SmokeVolumeGpu};
use crate::math::Vec3;

/// The density a source's column has at its centre, per unit of the source's `rate`.
pub(crate) const WAVES_DENSITY_PER_RATE: f32 = 0.3;

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// WGSL `smokeLayers`: `count` soft sheets over the unit cube's height, rolling and uneven.
pub(crate) fn layers(local: glam::Vec3, count: u32, thickness: f32, phase: f32) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let tau = std::f32::consts::TAU;
    let spacing = 1.0 / count as f32;
    let sigma = thickness.max(0.02) * spacing;
    let mut d = 0.0;
    for i in 0..count {
        let fi = i as f32;
        let centre = (fi + 0.5) * spacing + 0.15 * spacing * (tau * (0.6 * local.x + 0.35 * local.z) + fi * 2.1 + phase).sin();
        let t = (local.y - centre) / sigma;
        let uneven = 1.0 + 0.2 * (tau * (1.3 * local.z - 0.4 * local.x) + fi * 1.3 + 0.7 * phase).sin();
        d += (-0.5 * t * t).exp() * uneven;
    }
    d
}

/// The Waves density of `volume` at a world point, at `time` (the CPU mirror of the march's).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn waves_density(volume: &SmokeVolume, p: Vec3, time: f32) -> f32 {
    waves_density_gpu(&volume.gpu(volume.grid_dims()), glam::Vec3::new(p.x, p.y, p.z), time)
}

/// `waves_density` from the volume's GPU data, operation for operation as WGSL's.
pub(crate) fn waves_density_gpu(v: &SmokeVolumeGpu, p: glam::Vec3, time: f32) -> f32 {
    let center = glam::Vec3::from(v.center);
    let d = p - center;
    let local = glam::Vec3::new(v.cos_yaw * d.x - v.sin_yaw * d.z, d.y, v.sin_yaw * d.x + v.cos_yaw * d.z) / glam::Vec3::from(v.size) + 0.5;
    if local.cmplt(glam::Vec3::ZERO).any() || local.cmpgt(glam::Vec3::ONE).any() {
        return 0.0;
    }
    let k = std::f32::consts::TAU / v.waves_wavelength;
    if v.num_sources == 0 {
        return v.layer_density * layers(local, v.layer_count, v.layer_thickness, 0.1 * v.waves_speed * time);
    }
    let mut total = 0.0;
    for i in 0..v.num_sources as usize {
        let s = &v.sources[i];
        let sp = glam::Vec3::from(s.position);
        let h = p.y - sp.y;
        if h < -s.radius {
            continue;
        }
        let start = smoothstep01((h + s.radius) / s.radius);
        let hp = h.max(0.0);
        let grow = (hp / v.waves_wavelength).clamp(0.0, 1.0);
        let ph = k * hp - v.waves_speed * time + i as f32 * 1.7;
        let sway = glam::Vec2::new(ph.sin() + 0.5 * (1.7 * ph + 1.1).sin(), (0.8 * ph + 0.3).cos() + 0.5 * (2.1 * ph + 2.3).cos()) * (v.waves_amplitude * grow / 1.5);
        let radius = s.radius + v.waves_spread * hp;
        let q = (glam::Vec2::new(p.x - sp.x, p.z - sp.z) - sway).length() / radius;
        let dilution = (s.radius / radius) * (s.radius / radius);
        let rise = glam::Vec3::from(s.velocity).length().max(0.05);
        let fade = (-v.dissipation * hp / rise).exp();
        let pulse = 0.8 + 0.2 * (0.5 * k * hp - 1.3 * v.waves_speed * time + i as f32).sin();
        total += WAVES_DENSITY_PER_RATE * s.rate * dilution * fade * pulse * start * (-2.0 * q * q).exp();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::simulations::smoke::{SmokeMotion, SmokeSource, SmokeVolume, WavesSettings};

    fn plume() -> (SmokeVolume, Vec3) {
        let src = Vec3::new(0.3, 0.9, -0.2);
        let v = SmokeVolume::rising_from(SmokeSource { position: src, ..Default::default() }, 0.8, SmokeMotion::waves());
        (v, src)
    }

    #[test]
    fn waves_plume_rises_from_its_source_and_fades() {
        let (v, src) = plume();
        let d = |dy: f32, dx: f32| waves_density(&v, src + Vec3::new(dx, dy, 0.0), 0.0);
        assert!(d(0.05, 0.0) > d(0.5, 0.0) && d(0.5, 0.0) > 0.0, "{} {}", d(0.05, 0.0), d(0.5, 0.0));
        assert!(d(0.05, 0.15) < 0.01 * d(0.05, 0.0), "{} vs {}", d(0.05, 0.15), d(0.05, 0.0));
    }

    #[test]
    fn waves_density_is_zero_outside_the_box_and_never_negative() {
        let (v, _) = plume();
        let hash = |i: u32, k: u32| {
            let mut x = i.wrapping_mul(0x9e37_79b9) ^ k.wrapping_mul(0x85eb_ca6b);
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            (x & 0xffff) as f32 / 65535.0
        };
        let mut inside = 0;
        for i in 0..10_000u32 {
            // a box twice the volume's
            let local = Vec3::new(hash(i, 1) * 2.0 - 0.5, hash(i, 2) * 2.0 - 0.5, hash(i, 3) * 2.0 - 0.5);
            let p = v.to_world(local);
            let d = waves_density(&v, p, hash(i, 4) * 10.0);
            assert!(d >= 0.0 && d.is_finite(), "{d} at {p:?}");
            let out = [local.x, local.y, local.z].iter().any(|c| *c < 0.0 || *c > 1.0);
            if out {
                assert_eq!(d, 0.0, "density {d} outside the box at {local:?}");
            } else if d > 0.0 {
                inside += 1;
            }
        }
        assert!(inside > 10, "hardly any density inside: {inside}");
    }

    #[test]
    fn waves_sway_with_time() {
        let (v, src) = plume();
        let WavesSettings { amplitude, spread, .. } = match v.motion {
            SmokeMotion::Waves(w) => w,
            _ => unreachable!(),
        };
        let radius = v.sources[0].radius;
        // the density-weighted x of the slice 0.4 m up
        let centre_x = |t: f32| {
            let (mut sum, mut weight) = (0.0f64, 0.0f64);
            for i in 0..101 {
                for k in 0..101 {
                    let p = src + Vec3::new((i as f32 / 50.0 - 1.0) * 0.2, 0.4, (k as f32 / 50.0 - 1.0) * 0.2);
                    let d = waves_density(&v, p, t) as f64;
                    sum += d * p.x as f64;
                    weight += d;
                }
            }
            (sum / weight) as f32
        };
        let (x0, x1) = (centre_x(0.0), centre_x(1.0));
        eprintln!("the slice's centre: {x0} at t 0, {x1} at t 1 (source {})", src.x);
        assert!((x1 - x0).abs() > 0.2 * amplitude, "moved {}", (x1 - x0).abs());
        for x in [x0, x1] {
            assert!((x - src.x).abs() <= amplitude + radius + spread * 0.4, "{x} strays from {}", src.x);
        }
    }

    #[test]
    fn waves_layers_lie_in_sheets() {
        use crate::simulations::smoke::SmokeFill;
        let mut v = SmokeVolume::new(Vec3::new(0.0, 1.5, 0.0), Vec3::new(6.0, 3.0, 5.0), SmokeMotion::waves());
        v.fill = SmokeFill::Layers { count: 3, thickness: 0.25, density: 0.6 };
        let n = 48u32;
        let at = |x: u32, y: u32, z: u32| {
            let local = Vec3::new((x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, (z as f32 + 0.5) / n as f32);
            waves_density(&v, v.to_world(local), 0.0)
        };
        let profile: Vec<f32> = (0..n).map(|y| (0..n).flat_map(|z| (0..n).map(move |x| (x, z))).map(|(x, z)| at(x, y, z)).sum::<f32>() / (n * n) as f32).collect();
        let max = profile.iter().cloned().fold(0.0f32, f32::max);
        let peaks: Vec<usize> = (1..n as usize - 1).filter(|&y| profile[y] > profile[y - 1] && profile[y] >= profile[y + 1] && profile[y] > 0.1 * max).collect();
        assert_eq!(peaks.len(), 3, "profile {profile:?}");
        let mid = peaks[1];
        let (mut lo, mut hi) = (mid, mid);
        while lo > 0 && profile[lo - 1] < profile[lo] { lo -= 1; }
        while hi + 1 < n as usize && profile[hi + 1] < profile[hi] { hi += 1; }
        let along_x: Vec<f32> = (0..n).map(|x| (lo as u32..=hi as u32).flat_map(|y| (0..n).map(move |z| (y, z))).map(|(y, z)| at(x, y, z)).sum()).collect();
        let mean = along_x.iter().sum::<f32>() / n as f32;
        let cv = (along_x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32).sqrt() / mean;
        assert!(cv < 0.3, "cv {cv}");
    }

    /// The WGSL and the CPU agree.
    #[test]
    fn waves_density_matches_its_wgsl() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let (v, src) = plume();
        let gpu = v.gpu(v.grid_dims());
        let points: Vec<[f32; 4]> = (0..64)
            .map(|i| {
                let f = i as f32;
                [src.x + ((f * 0.37).sin()) * 0.08, src.y + (f / 64.0) * 0.7 - 0.02, src.z + ((f * 0.71).cos()) * 0.08, (f * 0.13) % 3.0]
            })
            .collect();
        let code = format!(
            "{}\n\
            @group(0) @binding(0) var<uniform> vol : SmokeVolumeData;\n\
            @group(0) @binding(1) var<storage, read> pts : array<vec4f>;\n\
            @group(0) @binding(2) var<storage, read_write> out : array<f32>;\n\
            @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3u) {{\n\
                if (g.x >= arrayLength(&pts)) {{ return; }}\n\
                out[g.x] = smokeWavesDensity(vol, pts[g.x].xyz, pts[g.x].w);\n\
            }}",
            crate::simulations::smoke::SMOKE_COMMON_WGSL
        );
        use wgpu::util::DeviceExt;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let vol = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&gpu), usage: wgpu::BufferUsages::UNIFORM });
        let pts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&points), usage: wgpu::BufferUsages::STORAGE });
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64 * 4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vol.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            ],
        });
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut pass = e.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        e.copy_buffer_to_buffer(&out, 0, &read, 0, 64 * 4);
        queue.submit([e.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let got: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        let mut nonzero = 0;
        for (p, g) in points.iter().zip(&got) {
            let cpu = waves_density(&v, Vec3::new(p[0], p[1], p[2]), p[3]);
            assert!((g - cpu).abs() <= 1e-3 * cpu + 1e-5, "at {p:?}: gpu {g}, cpu {cpu}");
            nonzero += (cpu > 1e-3) as u32;
        }
        assert!(nonzero > 10, "too few points in the plume: {nonzero}");
    }
}
