//! Runs SmaaEffect on a real GPU over synthetic images in the display signal: flat colour and
//! low-contrast noise come through bit-exact, straight axis-aligned edges stay sharp, jagged
//! edges (a turned square, near-horizontal and near-vertical lines) are blended between their two
//! sides, a display-linear signal gives the sRGB-encoded one's image, and a resized effect
//! renders what a fresh one does. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{FilmGrain, SmaaEffect, SmaaOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

struct Dev {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gbuffer: GBuffer,
    camera: Camera,
}

fn gpu() -> Option<Dev> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    // SMAA reads neither the GBuffer nor the camera
    let gbuffer = GBuffer::new(&device, 1, 1, 1);
    Some(Dev { device, queue, gbuffer, camera: Camera::new(60.0, 0.1, 100.0, 1.0) })
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    // added, not or-ed: rounding the mantissa up may carry into the exponent
    sign | ((((exp as u32) << 10) + ((mant + 0x1000) >> 13)).min(0x7c00) as u16)
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

/// The halves an image is uploaded as (alpha 1).
fn halves(input: &[[f32; 3]]) -> Vec<[u16; 4]> {
    input.iter().map(|c| [f32_to_f16(c[0]), f32_to_f16(c[1]), f32_to_f16(c[2]), f32_to_f16(1.0)]).collect()
}

/// `fx` on `input` (`w` x `h`, uploaded as `Rgba16Float`, alpha 1) into an `Rgba16Float` output, in
/// its own submit; the output's halves.
fn render_with(dev: &Dev, fx: &mut SmaaEffect, input: &[[f32; 3]], w: u32, h: u32) -> Vec<[u16; 4]> {
    let device = &dev.device;
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let texture = |usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
            view_formats: &[],
        })
    };
    let input_tex = texture(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    dev.queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &input_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&halves(input)),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        size,
    );
    let output_tex = texture(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
    let (input_view, output_view) = (input_tex.create_view(&Default::default()), output_tex.create_view(&Default::default()));
    let row = (w * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    fx.render(device, &dev.queue, &mut encoder, &dev.gbuffer, &input_view, &dev.gbuffer.depth_view, &output_view, &dev.camera, w, h);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        size,
    );
    dev.queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let line: &[u16] = bytemuck::cast_slice(&data[(y * row) as usize..(y * row + w * 8) as usize]);
        out.extend(line.chunks(4).map(|px| [px[0], px[1], px[2], px[3]]));
    }
    out
}

/// A fresh `SmaaEffect` with `film` on `input`.
fn run(dev: &Dev, input: &[[f32; 3]], w: u32, h: u32, film: FilmGrain) -> Vec<[u16; 4]> {
    run_with_options(dev, SmaaOptions::default(), input, w, h, film)
}

fn run_with_options(dev: &Dev, options: SmaaOptions, input: &[[f32; 3]], w: u32, h: u32, film: FilmGrain) -> Vec<[u16; 4]> {
    let mut fx = SmaaEffect::new(options);
    fx.set_film(film);
    render_with(dev, &mut fx, input, w, h)
}

fn srgb_decode(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// The largest difference of any channel between two outputs, and the pixels that differ by more
/// than 2/255 in any channel.
fn difference(a: &[[u16; 4]], b: &[[u16; 4]]) -> (f32, f32, usize) {
    let (mut max, mut sum, mut over) = (0f32, 0f64, 0);
    for (p, q) in a.iter().zip(b) {
        let d = (0..3).map(|c| (f16_to_f32(p[c]) - f16_to_f32(q[c])).abs()).fold(0f32, f32::max);
        max = max.max(d);
        sum += d as f64;
        if d > 2.0 / 255.0 {
            over += 1;
        }
    }
    (max, (sum / a.len() as f64) as f32, over)
}

const KINDS: [Kind; 10] = [Kind::Flat, Kind::Noise, Kind::StraightVertical, Kind::StraightHorizontal, Kind::RotatedSquare, Kind::NearHorizontal, Kind::NearVertical, Kind::Diagonal, Kind::Lattice, Kind::FaintBright];

#[derive(Debug, Clone, Copy)]
enum Kind {
    Flat,
    Noise,
    StraightVertical,
    StraightHorizontal,
    RotatedSquare,
    NearHorizontal,
    NearVertical,
    Diagonal,
    /// Thin dark lines on a bright ground, a shallow set and a steep one: edges in most tiles,
    /// as a fence fills a frame.
    Lattice,
    /// A turned square 0.95 on 0.86: below the threshold in the encoded signal (0.09), above it in
    /// linear light (0.18), so only luma taken on the encoded signal leaves it alone.
    FaintBright,
}

/// A grey image of `kind` in the encoded signal, each pixel point-sampled at its centre.
fn image(kind: Kind, w: u32, h: u32) -> Vec<[f32; 3]> {
    let (wf, hf) = (w as f32, h as f32);
    let value = |x: u32, y: u32| -> f32 {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        let two = |inside: bool| if inside { 0.8 } else { 0.15 };
        match kind {
            Kind::Flat => 0.42,
            Kind::Noise => {
                let mut v = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
                v ^= v >> 15;
                v = v.wrapping_mul(0x2c1b_3c6d);
                v ^= v >> 12;
                0.5 + 0.03 * ((v & 0xffff) as f32 / 32767.5 - 1.0)
            }
            Kind::StraightVertical => if x < w / 2 { 0.1 } else { 0.9 },
            Kind::StraightHorizontal => if y < h / 2 { 0.1 } else { 0.9 },
            Kind::RotatedSquare => {
                let (s, c) = 20f32.to_radians().sin_cos();
                let (dx, dy) = (px - wf * 0.5, py - hf * 0.5);
                let (u, v) = (c * dx + s * dy, -s * dx + c * dy);
                let half = 0.3 * wf.min(hf);
                two(u.abs() < half && v.abs() < half)
            }
            Kind::NearHorizontal => two(py - 0.3 * hf > px / 7.0 && py - 0.7 * hf < px / 40.0),
            Kind::NearVertical => two(px - 0.3 * wf > py / 7.0 && px - 0.7 * wf < py / 40.0),
            Kind::Diagonal => two(px > py),
            Kind::Lattice => {
                let line = (px - 3.0 * py).rem_euclid(14.0) < 1.6 || (0.35 * px + py).rem_euclid(11.0) < 1.4;
                if line { 0.1 } else { 0.85 }
            }
            Kind::FaintBright => {
                let (s, c) = 20f32.to_radians().sin_cos();
                let (dx, dy) = (px - wf * 0.5, py - hf * 0.5);
                let half = 0.3 * wf.min(hf);
                if (c * dx + s * dy).abs() < half && (-s * dx + c * dy).abs() < half { 0.95 } else { 0.86 }
            }
        }
    };
    (0..h).flat_map(|y| (0..w).map(move |x| [value(x, y); 3])).collect()
}

#[test]
fn flat_and_low_contrast_noise_come_through_unchanged() {
    // and an edge below the threshold
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let (w, h) = (250, 190);
        for kind in [Kind::Flat, Kind::Noise, Kind::FaintBright] {
            let input = image(kind, w, h);
            let out = run(&dev, &input, w, h, FilmGrain::default());
            let changed = out.iter().zip(halves(&input)).filter(|(o, i)| **o != *i).count();
            assert_eq!(changed, 0, "{kind:?}: {changed} pixels changed");
    }
}

#[test]
fn straight_axis_aligned_edges_stay_sharp() {
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let (w, h) = (250, 190);
        for kind in [Kind::StraightVertical, Kind::StraightHorizontal] {
            let input = image(kind, w, h);
            let out = run(&dev, &input, w, h, FilmGrain::default());
            let changed = out.iter().zip(halves(&input)).filter(|(o, i)| **o != *i).count();
            assert_eq!(changed, 0, "{kind:?}: {changed} pixels changed");
    }
}

#[test]
fn jagged_edges_are_blended_between_their_sides() {
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let (w, h) = (250, 190);
        for kind in [Kind::RotatedSquare, Kind::NearHorizontal, Kind::NearVertical] {
            let input = image(kind, w, h);
            let out = run(&dev, &input, w, h, FilmGrain::default());
            let mut moved = 0;
            for (o, i) in out.iter().zip(&input) {
                for c in 0..3 {
                    let v = f16_to_f32(o[c]);
                    assert!(v.is_finite() && (0.15 - 1e-3..=0.8 + 1e-3).contains(&v), "{kind:?}: {v}");
                }
                if (f16_to_f32(o[0]) - i[0]).abs() > 2.0 / 255.0 {
                    moved += 1;
                }
            }
            assert!(moved >= 50, "{kind:?}: only {moved} pixels blended");
    }
}

/// A display-linear signal (`encode_srgb: false`, sRGB surfaces) gives the encoded signal's image:
/// its edges are found on the encoded luma and its pixels blended in linear light. On partial
/// workgroups and tiny frames too.
#[test]
fn a_linear_signal_gives_the_encoded_signal_s_image() {
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let encode = |v: f32| if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    for (w, h) in [(250, 190), (7, 5)] {
        for kind in KINDS {
            let encoded = image(kind, w, h);
            let linear: Vec<[f32; 3]> = encoded.iter().map(|c| c.map(srgb_decode)).collect();
            let a = run(&dev, &encoded, w, h, FilmGrain { encode_srgb: true, ..Default::default() });
            let b = run(&dev, &linear, w, h, FilmGrain { encode_srgb: false, ..Default::default() });
            let b: Vec<[u16; 4]> = b.iter().map(|p| [0, 1, 2].map(|c| f32_to_f16(encode(f16_to_f32(p[c])))).into_iter().chain([p[3]]).collect::<Vec<_>>().try_into().unwrap()).collect();
            let (max, _, over) = difference(&a, &b);
            assert!(max <= 2.0 / 255.0, "{kind:?} {w}x{h}: max {max} ({over} pixels over 2/255)");
        }
    }
    // not vacuous: it blends
    let input = image(Kind::RotatedSquare, 250, 190);
    let (_, _, moved) = difference(&run(&dev, &input, 250, 190, FilmGrain::default()), &halves(&input));
    assert!(moved >= 50, "only {moved} pixels blended");
}

/// One effect resized back and forth renders what a fresh one does (its edges and weights follow
/// the frame's size).
#[test]
fn resizing_matches_fresh_effects() {
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = SmaaEffect::new(SmaaOptions::default());
    for (w, h) in [(64, 48), (96, 80), (96, 80), (64, 48)] {
        let input = image(Kind::RotatedSquare, w, h);
        let reused = render_with(&dev, &mut fx, &input, w, h);
        let fresh = run(&dev, &input, w, h, FilmGrain::default());
        assert!(reused == fresh, "{w}x{h}: the reused effect differs from a fresh one");
    }
}

/// GPU time at 1920 x 1080 by wall clock, SMAA's three passes and the TAA resolve alternated, on
/// a mostly flat image (RotatedSquare) and a dense one (Lattice: edges in most of the frame):
/// `cargo test -p kansei-core --release --test smaa_gpu time_smaa_1080p -- --ignored --nocapture`.
#[test]
#[ignore]
fn time_smaa_1080p() {
    use kansei_core::postprocessing::effects::{TemporalAAEffect, TemporalAAOptions};
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let (w, h) = (1920u32, 1080u32);
    let device = &dev.device;
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let texture = |usage| {
        device.create_texture(&wgpu::TextureDescriptor { label: None, size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float, usage, view_formats: &[] })
    };
    let output = texture(wgpu::TextureUsages::STORAGE_BINDING).create_view(&Default::default());
    let gbuffer = GBuffer::new(device, w, h, 1);
    for kind in [Kind::RotatedSquare, Kind::Lattice] {
        let input_tex = texture(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        dev.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &input_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halves(&image(kind, w, h))),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            size,
        );
        let input = input_tex.create_view(&Default::default());
        let mut effects: Vec<(&str, Box<dyn PostProcessingEffect>)> = vec![
            ("SMAA", Box::new(SmaaEffect::new(SmaaOptions::default()))),
            ("TAA resolve", Box::new(TemporalAAEffect::new(TemporalAAOptions::default()))),
        ];
        let mut times = vec![Vec::new(); effects.len()];
        for round in 0..45 {
            for (k, (_, fx)) in effects.iter_mut().enumerate() {
                let mut encoder = device.create_command_encoder(&Default::default());
                for _ in 0..10 {
                    fx.render(device, &dev.queue, &mut encoder, &gbuffer, &input, &gbuffer.depth_view, &output, &dev.camera, w, h);
                }
                let t = std::time::Instant::now();
                dev.queue.submit([encoder.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                if round >= 5 {
                    times[k].push(t.elapsed().as_secs_f64() * 1e3 / 10.0);
                }
            }
        }
        for ((name, _), t) in effects.iter().zip(&mut times) {
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("{kind:?} {name}: {:.3} ms (median per frame, {w}x{h})", t[t.len() / 2]);
        }
    }
}

/// What the default reach (16 px) costs against iryoku High's 32, on every image at 1920 x 1080:
/// `cargo test -p kansei-core --release --test smaa_gpu reach_cost_on_synthetic_edges -- --ignored --nocapture`.
#[test]
#[ignore]
fn reach_cost_on_synthetic_edges() {
    let Some(dev) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let (w, h) = (1920, 1080);
    let far = SmaaOptions { reach: 32, ..Default::default() };
    for kind in KINDS {
        let input = image(kind, w, h);
        let near = run(&dev, &input, w, h, FilmGrain::default());
        let far = run_with_options(&dev, far, &input, w, h, FilmGrain::default());
        let (max, mean, over) = difference(&near, &far);
        eprintln!("{kind:?}: mean {mean:.6}, max {max:.4} ({:.1}/255), {over} pixels over 2/255", max * 255.0);
    }
}
