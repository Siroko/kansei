//! Smoke rendered through the volumetric fog (`VolumetricFogEffect::set_smoke`) on a real GPU:
//! the seam with the fog, self-shadowing, the camera inside a volume, geometry in the smoke, the
//! history. Each test renders a small image (the camera at (0, 1, 0) looking down -z, a constant
//! background of 0.3) and compares it with the fog alone or with a CPU march. Skipped (passes)
//! when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::math::Vec3;
use kansei_core::postprocessing::effects::{VolumetricFogEffect, VolumetricFogOptions};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::GBuffer;
use kansei_core::simulations::smoke::*;

const BACKGROUND: f32 = 0.3;

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

/// The image's textures and camera: a constant background, a depth from `linear_depth(x, y)`
/// (None: the sky), the camera at (0, 1, 0) looking down -z.
struct Scene {
    w: u32,
    h: u32,
    input: wgpu::TextureView,
    output_tex: wgpu::Texture,
    output: wgpu::TextureView,
    depth: wgpu::TextureView,
    gbuffer: GBuffer,
    camera: Camera,
}

fn scene(device: &wgpu::Device, queue: &wgpu::Queue, (w, h): (u32, u32), linear_depth: impl Fn(u32, u32) -> Option<f32>) -> Scene {
    let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
    let input_tex = texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    let texels: Vec<f32> = (0..w * h).flat_map(|_| [BACKGROUND, BACKGROUND, BACKGROUND, 1.0]).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &input_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 16), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let output_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
    let output = output_tex.create_view(&Default::default());
    let depth_tex = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
    let depth = depth_tex.create_view(&Default::default());
    let mut camera = Camera::new(60.0, 0.1, 100.0, w as f32 / h as f32);
    camera.set_position(0.0, 1.0, 0.0);
    camera.look_at(&Vec3::new(0.0, 1.0, -1.0));
    camera.update_view_matrix();
    // the depth per pixel, drawn as frag_depth
    let (n, f) = (camera.near, camera.far);
    let ndc: Vec<f32> = (0..w * h).map(|i| linear_depth(i % w, i / w).map_or(1.0, |d| f * (d - n) / (d * (f - n)))).collect();
    let buf = { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&ndc), usage: wgpu::BufferUsages::STORAGE }) };
    let code = format!("@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
        @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
        @fragment fn fs(@builtin(position) pos : vec4f) -> @builtin(frag_depth) f32 {{ return d[u32(pos.y) * {w}u + u32(pos.x)]; }}");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState { format: GBuffer::DEPTH_FORMAT, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::Always), stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
    let mut e = device.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit([e.finish()]);
    Scene { w, h, input: input_tex.create_view(&Default::default()), output_tex, output, depth, gbuffer: GBuffer::new(device, w, h, 1), camera }
}

/// Render `frames` frames of the fog, and read the last one back.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, fog: &mut VolumetricFogEffect, s: &Scene, frames: u32) -> Vec<[f32; 4]> {
    for _ in 0..frames {
        let mut e = device.create_command_encoder(&Default::default());
        fog.render(device, queue, &mut e, &s.gbuffer, &s.input, &s.depth, &s.output, &s.camera, s.w, s.h);
        queue.submit([e.finish()]);
    }
    let (w, h) = (s.w, s.h);
    let row = (w * 8).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut e = device.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &s.output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([e.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buf.slice(..).get_mapped_range().unwrap();
    (0..w * h)
        .map(|i| {
            let o = ((i / w) * row + (i % w) * 8) as usize;
            let c = |k: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * k], data[o + 2 * k + 1]]));
            [c(0), c(1), c(2), c(3)]
        })
        .collect()
}

/// A fog of `density` per metre lit by a uniform sky of `ambient`, its grid fine along the view.
fn fog(density: f32, ambient: f32) -> VolumetricFogEffect {
    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
        grid: FroxelGridOptions { grid_w: 16, grid_h: 16, grid_d: 128, near: 0.1, far: 40.0, temporal: false, blend_factor: 1.0 },
        base_density: density,
        height_falloff: 0.0,
        fog_height: 100.0,
        ambient: Vec3::new(ambient, ambient, ambient),
        ..Default::default()
    });
    fog.smoke_render.temporal = false;
    fog
}

/// A box of smoke `density` per metre, albedo 1, extinction 1, no fade at its faces.
fn smoke_box(center: Vec3, size: Vec3, density: f32) -> SmokeVolume {
    let mut v = SmokeVolume::new(center, size, SmokeMotion::curl_noise());
    v.resolution = 32;
    v.fill = SmokeFill::Box { density, temperature: 0.0 };
    v.look = SmokeLook { density_scale: 1.0, albedo: Vec3::new(1.0, 1.0, 1.0), extinction: 1.0, dissipation: 0.0, edge_fade: 0.0 };
    v
}

/// The mean colour (r) over the `pw` x `ph` pixels centred on (cx, cy).
fn mean(img: &[[f32; 4]], w: u32, (cx, cy): (u32, u32), (pw, ph): (u32, u32)) -> f32 {
    let mut sum = 0.0;
    for y in cy - ph / 2..cy + ph / 2 {
        for x in cx - pw / 2..cx + pw / 2 {
            sum += img[(y * w + x) as usize][0];
        }
    }
    sum / (pw * ph) as f32
}

/// The image's ray through pixel (x, y).
fn ray(s: &Scene, x: u32, y: u32) -> glam::Vec3 {
    let inv = (s.camera.projection_matrix.to_glam() * s.camera.view_matrix.to_glam()).inverse();
    let ndc = glam::Vec2::new((x as f32 + 0.5) / s.w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / s.h as f32 * 2.0);
    (inv.project_point3(glam::Vec3::new(ndc.x, ndc.y, 1.0)) - glam::Vec3::new(0.0, 1.0, 0.0)).normalize()
}

/// A CPU march (4000 steps) of an ambient-lit fog (`fog` per metre to the grid's far, scattering
/// as density * ambient) and boxes of smoke ((min, max, density), albedo 1, lit the same) along
/// the pixel's ray, over the background, to `linear_depth` (None: the sky).
fn cpu(s: &Scene, x: u32, y: u32, fog: f32, ambient: f32, boxes: &[(glam::Vec3, glam::Vec3, f32)], linear_depth: Option<f32>) -> f32 {
    let rd = ray(s, x, y);
    let cos_view = rd.dot(glam::Vec3::NEG_Z);
    let t_end = linear_depth.unwrap_or(s.camera.far) / cos_view;
    let n = 4000;
    let dt = t_end / n as f32;
    let (mut transmittance, mut light) = (1.0f32, 0.0f32);
    for k in 0..n {
        let t = (k as f32 + 0.5) * dt;
        let p = glam::Vec3::new(0.0, 1.0, 0.0) + rd * t;
        let mut sigma = if t * cos_view < 40.0 { fog } else { 0.0 };
        for (lo, hi, d) in boxes {
            if p.cmpge(*lo).all() && p.cmple(*hi).all() {
                sigma += d;
            }
        }
        if sigma > 0.0 {
            let e = (-sigma * dt).exp();
            light += transmittance * ambient * (1.0 - e);
            transmittance *= e;
        }
    }
    light + transmittance * BACKGROUND
}

#[test]
fn smoke_with_no_density_leaves_the_fog_as_it_was() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| Some(12.0));
    let mut alone = fog(0.05, 1.0);
    let want = render(&device, &queue, &mut alone, &s, 1);
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut empty = smoke_box(Vec3::new(0.0, 1.0, -4.0), Vec3::new(6.0, 6.0, 4.0), 0.0);
    empty.fill = SmokeFill::None;
    sim.add(empty);
    let mut with = fog(0.05, 1.0);
    with.set_smoke(Some(&sim.handle()));
    let got = render(&device, &queue, &mut with, &s, 1);
    let max = want.iter().map(|c| c[0]).fold(0.0f32, f32::max);
    let worst = want.iter().zip(&got).map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
    assert!(worst < 1e-3 * max, "worst difference {worst} (max {max})");
}

#[test]
fn thin_smoke_scatters_as_the_fog_does() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| Some(20.0));
    let sun = Light::Directional(DirectionalLight::new(Vec3::new(-0.4, -0.5, -0.6), Vec3::new(1.0, 1.0, 1.0), 3.0));
    let mut alone = fog(0.02, 0.05);
    alone.update_lights([&sun]);
    let want = mean(&render(&device, &queue, &mut alone, &s, 1), s.w, (32, 16), (8, 4));
    let mut sim = SmokeSimulation::new(&device, &queue);
    sim.add(smoke_box(Vec3::new(0.0, 1.0, 0.0), Vec3::new(200.0, 200.0, 200.0), 0.02));
    let mut with = fog(0.0, 0.05);
    with.update_lights([&sun]);
    with.set_smoke(Some(&sim.handle()));
    let got = mean(&render(&device, &queue, &mut with, &s, 1), s.w, (32, 16), (8, 4));
    eprintln!("fog {want}, smoke {got}");
    assert!((got / want - 1.0).abs() < 0.03, "fog {want}, smoke {got}");
}

#[test]
fn dense_smoke_hides_the_fog_behind_it() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| None);
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut black = smoke_box(Vec3::new(0.0, 1.0, -6.0), Vec3::new(30.0, 30.0, 2.0), 50.0);
    black.look.albedo = Vec3::ZERO;
    sim.add(black);
    let mut with = fog(0.05, 1.0);
    with.set_smoke(Some(&sim.handle()));
    let img = render(&device, &queue, &mut with, &s, 1);
    let got = mean(&img, s.w, (32, 16), (8, 4));
    // the fog's light over the 5 m before the smoke, nothing from behind it
    let mut want = 0.0;
    for y in 14..18 {
        for x in 28..36 {
            let rd = ray(&s, x, y);
            want += 1.0 - (-0.05 * 5.0 / rd.dot(glam::Vec3::NEG_Z)).exp();
        }
    }
    want /= 32.0;
    assert!((got / want - 1.0).abs() < 0.03, "got {got}, want {want}");
}

#[test]
fn fog_in_front_veils_the_smoke() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| None);
    let mut sim = SmokeSimulation::new(&device, &queue);
    sim.add(smoke_box(Vec3::new(0.0, 1.0, -8.5), Vec3::new(30.0, 30.0, 1.0), 2.0));
    let mut with = fog(0.1, 1.0);
    with.set_smoke(Some(&sim.handle()));
    let img = render(&device, &queue, &mut with, &s, 1);
    let got = mean(&img, s.w, (32, 16), (8, 4));
    let boxes = [(glam::Vec3::new(-15.0, -14.0, -9.0), glam::Vec3::new(15.0, 16.0, -8.0), 2.0)];
    let mut want = 0.0;
    for y in 14..18 {
        for x in 28..36 {
            want += cpu(&s, x, y, 0.1, 1.0, &boxes, None);
        }
    }
    want /= 32.0;
    assert!((got / want - 1.0).abs() < 0.03, "got {got}, want {want}");
}

#[test]
fn the_smoke_is_darker_away_from_the_light() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (128, 64), |_, _| None);
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut ball = smoke_box(Vec3::new(0.0, 1.0, -3.0), Vec3::new(1.2, 1.2, 1.2), 0.0);
    ball.resolution = 64;
    ball.fill = SmokeFill::Sphere { center: Vec3::new(0.0, 1.0, -3.0), radius: 0.5, density: 20.0, temperature: 0.0 };
    sim.add(ball);
    // the sun travels -x: it lights the ball's right side (+x)
    let sun = Light::Directional(DirectionalLight::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 5.0));
    let mut with = fog(0.0, 0.0);
    with.update_lights([&sun]);
    with.set_smoke(Some(&sim.handle()));
    let img = render(&device, &queue, &mut with, &s, 1);
    // at 3 m a pixel is 2 * 3 tan(30 degrees) / 64 = 5.4 cm: the ball's thirds left and right of
    // its middle (|x| from 1/6 m to 0.45 m, short of the rim the half-resolution march blends
    // with the background), over the middle rows
    let px = 6.0 * 30f32.to_radians().tan() / 64.0;
    let third = |x0: f32, x1: f32| {
        let (mut sum, mut n) = (0.0, 0);
        for y in 30..34u32 {
            for x in 0..s.w {
                let m = (x as f32 + 0.5 - 64.0) * px;
                if m > x0 && m < x1 {
                    sum += img[(y * s.w + x) as usize][0];
                    n += 1;
                }
            }
        }
        sum / n as f32
    };
    let (left, right) = (third(-0.45, -1.0 / 6.0), third(1.0 / 6.0, 0.45));
    eprintln!("left third {left}, right third {right}");
    assert!(right > 2.0 * left, "left {left}, right {right}");
}

#[test]
fn the_camera_inside_a_volume_sees_the_smoke_around_it() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| None);
    let mut sim = SmokeSimulation::new(&device, &queue);
    sim.add(smoke_box(Vec3::new(0.0, 1.0, 0.0), Vec3::new(4.0, 4.0, 4.0), 0.5));
    let mut with = fog(0.0, 1.0);
    with.set_smoke(Some(&sim.handle()));
    let img = render(&device, &queue, &mut with, &s, 1);
    assert!(img.iter().all(|c| c.iter().all(|v| v.is_finite())), "a pixel is not finite");
    let got = mean(&img, s.w, (32, 16), (8, 4));
    let boxes = [(glam::Vec3::new(-2.0, -1.0, -2.0), glam::Vec3::new(2.0, 3.0, 2.0), 0.5)];
    let mut want = 0.0;
    for y in 14..18 {
        for x in 28..36 {
            want += cpu(&s, x, y, 0.0, 1.0, &boxes, None);
        }
    }
    want /= 32.0;
    let e1 = (-1.0f32).exp();
    eprintln!("got {got}, CPU {want}, 1 - e^-1 + 0.3 e^-1 = {}", 1.0 - e1 + 0.3 * e1);
    assert!((got / want - 1.0).abs() < 0.03, "got {got}, want {want}");
    assert!((want / (1.0 - e1 + 0.3 * e1) - 1.0).abs() < 0.03);
}

#[test]
fn smoke_behind_a_surface_is_hidden() {
    let Some((device, queue)) = gpu() else { return };
    let plane = |x: u32, _y: u32| (x < 32).then_some(1.0);
    let s = scene(&device, &queue, (64, 32), plane);
    let mut sim = SmokeSimulation::new(&device, &queue);
    sim.add(smoke_box(Vec3::new(0.0, 1.0, -2.5), Vec3::new(20.0, 20.0, 4.0), 1.0));
    let mut with = fog(0.0, 1.0);
    with.set_smoke(Some(&sim.handle()));
    let img = render(&device, &queue, &mut with, &s, 1);
    assert!(img.iter().all(|c| c.iter().all(|v| v.is_finite())), "a pixel is not finite");
    let boxes = [(glam::Vec3::new(-10.0, -9.0, -4.5), glam::Vec3::new(10.0, 11.0, -0.5), 1.0)];
    for (cx, depth) in [(16u32, Some(1.0)), (48, None)] {
        let got = mean(&img, s.w, (cx, 16), (4, 4));
        let mut want = 0.0;
        for y in 14..18 {
            for x in cx - 2..cx + 2 {
                want += cpu(&s, x, y, 0.0, 1.0, &boxes, depth);
            }
        }
        want /= 16.0;
        eprintln!("column {cx}: got {got}, want {want}");
        assert!((got / want - 1.0).abs() < 0.03, "column {cx}: got {got}, want {want}");
    }
    // the rim: the pixels next to the surface's edge keep to their own side
    for y in 12..20 {
        let (left, right) = (img[(y * s.w + 31) as usize][0], img[(y * s.w + 32) as usize][0]);
        let (want_left, want_right) = (cpu(&s, 31, y, 0.0, 1.0, &boxes, Some(1.0)), cpu(&s, 32, y, 0.0, 1.0, &boxes, None));
        assert!((left / want_left - 1.0).abs() < 0.1 && (right / want_right - 1.0).abs() < 0.1, "row {y} at the edge: {left} | {right}, want {want_left} | {want_right}");
    }
}

#[test]
fn a_moved_volume_leaves_no_ghost() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| Some(10.0));
    let background = render(&device, &queue, &mut fog(0.02, 0.5), &s, 1);
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut ball = smoke_box(Vec3::new(-1.0, 1.0, -4.0), Vec3::new(1.2, 1.2, 1.2), 0.0);
    ball.fill = SmokeFill::Sphere { center: Vec3::new(-1.0, 1.0, -4.0), radius: 0.5, density: 5.0, temperature: 0.0 };
    let id = sim.add(ball);
    let mut with = fog(0.02, 0.5);
    with.smoke_render.temporal = true;
    with.set_smoke(Some(&sim.handle()));
    let before = render(&device, &queue, &mut with, &s, 30);
    // the ball's old place: x = -1 m at 4 m is 32 - 1 / (4 tan 30 * 2 / 32) = 25 pixels across
    let old = |img: &[[f32; 4]]| mean(img, s.w, (25, 16), (4, 4));
    assert!((old(&before) / old(&background) - 1.0).abs() > 0.2, "the ball does not show: {} vs {}", old(&before), old(&background));
    // a still view's history settles on the march itself (it reprojects where it should)
    let mut still = fog(0.02, 0.5);
    still.set_smoke(Some(&sim.handle()));
    let march = render(&device, &queue, &mut still, &s, 1);
    assert!((old(&before) / old(&march) - 1.0).abs() < 0.02, "the history drifted: {} vs {}", old(&before), old(&march));
    sim.volume_mut(id).center = Vec3::new(1.0, 1.0, -4.0);
    sim.step(0.0, &SmokeMouse::NONE);
    let after = render(&device, &queue, &mut with, &s, 1);
    eprintln!("old place: background {}, with the ball {}, after the move {}", old(&background), old(&before), old(&after));
    assert!((old(&after) / old(&background) - 1.0).abs() < 0.05, "a ghost: {} vs {}", old(&after), old(&background));
}

#[test]
fn history_survives_a_resize() {
    let Some((device, queue)) = gpu() else { return };
    let small = scene(&device, &queue, (64, 32), |_, _| Some(10.0));
    let large = scene(&device, &queue, (96, 48), |_, _| Some(10.0));
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut ball = smoke_box(Vec3::new(0.0, 1.0, -4.0), Vec3::new(1.2, 1.2, 1.2), 0.0);
    ball.fill = SmokeFill::Sphere { center: Vec3::new(0.0, 1.0, -4.0), radius: 0.5, density: 5.0, temperature: 0.0 };
    sim.add(ball);
    let mut with = fog(0.02, 0.5);
    with.smoke_render.temporal = true;
    with.set_smoke(Some(&sim.handle()));
    render(&device, &queue, &mut with, &small, 5);
    let got = render(&device, &queue, &mut with, &large, 1);
    assert!(got.iter().all(|c| c.iter().all(|v| v.is_finite())), "a pixel is not finite");
    let mut still = fog(0.02, 0.5);
    still.set_smoke(Some(&sim.handle()));
    let want = render(&device, &queue, &mut still, &large, 1);
    let m = |img: &[[f32; 4]]| mean(img, large.w, (48, 24), (16, 12));
    eprintln!("after the resize {}, without history {}", m(&got), m(&want));
    assert!((m(&got) / m(&want) - 1.0).abs() < 0.05, "{} vs {}", m(&got), m(&want));
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a < 6.1e-5 {
        return sign | (a / 2f32.powi(-24)).round() as u16;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    if mant & 0x1000 != 0 {
        h += 1;
    }
    sign | h as u16
}

/// The blob's centroid column: pixels weighted by how far they differ from the background's.
fn centroid_column(img: &[[f32; 4]], background: &[[f32; 4]], w: u32, rows: std::ops::Range<u32>) -> f32 {
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for y in rows {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = (img[i][0] - background[i][0]).abs() as f64;
            sum += d * (x as f64 + 0.5);
            weight += d;
        }
    }
    (sum / weight.max(1e-12)) as f32
}

#[test]
fn the_history_follows_the_gas() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (512, 256), |_, _| Some(10.0));
    let background = render(&device, &queue, &mut fog(0.02, 0.5), &s, 1);
    // a ball of smoke carried along +x at 0.5 m/s: no projection (a closed box's would stop a
    // uniform flow), no forces
    let mut sim = SmokeSimulation::new(&device, &queue);
    let mut settings = FluidSmokeSettings::default();
    (settings.pressure_iterations, settings.buoyancy, settings.weight, settings.vorticity) = (0, 0.0, 0.0, 0.0);
    let mut ball = smoke_box(Vec3::new(0.0, 1.0, -3.5), Vec3::new(2.0, 2.0, 2.0), 0.0);
    ball.motion = SmokeMotion::Fluid(settings);
    ball.resolution = 64;
    ball.fill = SmokeFill::Sphere { center: Vec3::new(0.0, 1.0, -3.5), radius: 0.2, density: 6.0, temperature: 0.0 };
    let id = sim.add(ball);
    let grid = sim.grid_textures(id).unwrap();
    let [gw, gh, gd] = grid.dims;
    let one = f32_to_f16(0.5);
    let texels: Vec<u16> = (0..gw * gh * gd).flat_map(|i| [if i % gw == 0 { 0 } else { one }, 0, 0, 0]).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: grid.velocity.unwrap(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(gw * 8), rows_per_image: Some(gh) },
        wgpu::Extent3d { width: gw, height: gh, depth_or_array_layers: gd },
    );
    let mut with = fog(0.02, 0.5);
    with.smoke_render.temporal = true;
    with.set_smoke(Some(&sim.handle()));
    let mut img = Vec::new();
    for _ in 0..20 {
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
        img = render(&device, &queue, &mut with, &s, 1);
    }
    let mut still = fog(0.02, 0.5);
    still.set_smoke(Some(&sim.handle()));
    let want = render(&device, &queue, &mut still, &s, 1);
    let rows = 118..138;
    let (got, want) = (centroid_column(&img, &background, s.w, rows.clone()), centroid_column(&want, &background, s.w, rows));
    eprintln!("the blob's centroid column: with history {got}, without {want}");
    assert!((got - want).abs() < 1.0, "the history lags: {got} vs {want}");
}

#[test]
fn overlapping_volumes_march_as_one_medium() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| Some(12.0));
    let render_boxes = |boxes: &[(Vec3, Vec3, f32)]| {
        let mut sim = SmokeSimulation::new(&device, &queue);
        for (c, size, d) in boxes {
            sim.add(smoke_box(*c, *size, *d));
        }
        let mut f = fog(0.02, 1.0);
        f.set_smoke(Some(&sim.handle()));
        render(&device, &queue, &mut f, &s, 1)
    };
    // two coincident boxes against one of their summed density
    let (c, size) = (Vec3::new(0.0, 1.0, -5.0), Vec3::new(8.0, 8.0, 3.0));
    let two = render_boxes(&[(c, size, 0.25), (c, size, 0.25)]);
    let one = render_boxes(&[(c, size, 0.5)]);
    let (a, b) = (mean(&two, s.w, (32, 16), (8, 4)), mean(&one, s.w, (32, 16), (8, 4)));
    eprintln!("coincident: two {a}, one {b}");
    assert!((a / b - 1.0).abs() < 0.02, "two {a}, one {b}");
    // two boxes touching at x = 0 against one spanning both: the middle row, column by column
    let left = render_boxes(&[(Vec3::new(-2.0, 1.0, -5.0), Vec3::new(4.0, 8.0, 3.0), 0.5), (Vec3::new(2.0, 1.0, -5.0), Vec3::new(4.0, 8.0, 3.0), 0.5)]);
    let whole = render_boxes(&[(Vec3::new(0.0, 1.0, -5.0), Vec3::new(8.0, 8.0, 3.0), 0.5)]);
    let y = s.h / 2;
    let worst = (0..s.w).map(|x| { let i = (y * s.w + x) as usize; (left[i][0] / whole[i][0] - 1.0).abs() }).fold(0.0f32, f32::max);
    eprintln!("touching: worst column {worst}");
    assert!(worst < 0.02, "a column differs by {worst}");
}

#[test]
fn switching_levels_at_runtime_keeps_rendering() {
    let Some((device, queue)) = gpu() else { return };
    let s = scene(&device, &queue, (64, 32), |_, _| Some(6.0));
    let mut sim = SmokeSimulation::new(&device, &queue);
    let source = SmokeSource { position: Vec3::new(0.0, 0.5, -3.0), radius: 0.05, ..Default::default() };
    let mut plume = SmokeVolume::rising_from(source, 0.8, SmokeMotion::fluid());
    plume.resolution = 32;
    let plume = sim.add(plume);
    let mut layers = SmokeVolume::new(Vec3::new(0.0, 1.0, -3.0), Vec3::new(4.0, 2.0, 4.0), SmokeMotion::fluid());
    (layers.resolution, layers.fill) = (24, SmokeFill::Layers { count: 3, thickness: 0.25, density: 0.6 });
    let layers = sim.add(layers);
    let mut f = fog(0.02, 1.0);
    f.smoke_render.temporal = true;
    f.set_smoke(Some(&sim.handle()));
    let levels = [SmokeMotion::fluid(), SmokeMotion::waves(), SmokeMotion::curl_noise()];
    for frame in 0..12 {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        sim.set_motion(plume, levels[frame % 3]);
        sim.set_motion(layers, levels[(frame + 1) % 3]);
        sim.set_enabled(layers, frame % 2 == 0);
        sim.step(1.0 / 60.0, &SmokeMouse::NONE);
        let img = render(&device, &queue, &mut f, &s, 1);
        let error = pollster::block_on(scope.pop());
        assert!(error.is_none(), "frame {frame}: {error:?}");
        assert!(img.iter().all(|c| c.iter().all(|v| v.is_finite())), "frame {frame}: a pixel is not finite");
    }
}
