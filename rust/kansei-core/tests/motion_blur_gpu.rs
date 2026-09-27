//! Runs MotionBlurEffect on synthetic frames on a real GPU: a still camera or a cut leaves the
//! image untouched, a panning camera blurs by the length its motion predicts (and no longer than
//! the cap), a moving object smears over a static background without halos, and a tracked
//! foreground stays sharp over a background blurred by the pan. Skipped (passes) when no
//! adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{MotionBlurEffect, MotionBlurOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

const W: u32 = 384;
const H: u32 = 192;
const NEAR: f32 = 0.1;
const FAR: f32 = 1000.0;
const FOV: f32 = 40.0;
/// The velocity target's clear value where no material writes one.
const NONE: [f32; 2] = [GBuffer::NO_VELOCITY; 2];

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

/// Focal length in pixels of the test camera (vertical fov over H).
fn focal_px() -> f32 {
    0.5 * H as f32 / (0.5 * FOV.to_radians()).tan()
}

/// A camera that moved `dx` metres to the right since last frame (0: still).
fn camera_moved(dx: f32) -> Camera {
    let mut camera = Camera::new(FOV, NEAR, FAR, W as f32 / H as f32);
    camera.set_position(-dx, 0.0, 0.0);
    camera.update_view_matrix();
    camera.end_frame();
    camera.set_position(0.0, 0.0, 0.0);
    camera.update_view_matrix();
    camera
}

fn effect(amount: f32, max: f32) -> MotionBlurEffect {
    MotionBlurEffect::new(MotionBlurOptions { amount, max, sample_count: 32, target_fps: None })
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The GBuffer's (render) size; the effect's input and output are W x H.
    render: (u32, u32),
    gbuffer: GBuffer,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
}

impl Harness {
    fn new() -> Option<Self> {
        Self::with_render_scale(1.0)
    }

    /// Depth and velocity at `scale` of W x H, as after a temporal upscaler.
    fn with_render_scale(scale: f32) -> Option<Self> {
        let (device, queue) = gpu()?;
        // writes the depth buffer and the velocity target from rgba32float texels (NDC depth,
        // velocity in uv per frame)
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var src : texture_2d<f32>;
                struct FOut { @location(4) velocity : vec2f, @builtin(frag_depth) depth : f32 };
                @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
                    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
                    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
                }
                @fragment fn fs(@builtin(position) pos : vec4f) -> FOut {
                    let t = textureLoad(src, vec2u(pos.xy), 0);
                    return FOut(t.yz, t.x);
                }
                "#
                .into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let mut targets: [Option<wgpu::ColorTargetState>; GBuffer::VELOCITY_TARGET + 1] = Default::default();
        targets[GBuffer::VELOCITY_TARGET] = Some(GBuffer::VELOCITY_FORMAT.into());
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &targets, compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GBuffer::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let render = ((W as f32 * scale).round() as u32, (H as f32 * scale).round() as u32);
        let gbuffer = GBuffer::new(&device, render.0, render.1, 1);
        Some(Self { device, queue, render, gbuffer, pipeline, bgl })
    }

    fn texture(&self, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, (width, height): (u32, u32)) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// Run the effect on `color` (rgb per W x H pixel) with the render-size depth buffer at view
    /// depths `depth_m` (metres) and the velocity target holding `velocity_px` (W x H pixels per
    /// frame, or NONE); rgb out.
    fn run(&self, effect: &mut MotionBlurEffect, camera: &Camera, color: &[[f32; 3]], depth_m: &[f32], velocity_px: &[[f32; 2]]) -> Vec<[f32; 3]> {
        let (d, q) = (&self.device, &self.queue);
        let input = self.texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, (W, H));
        let rgba: Vec<f32> = color.iter().flat_map(|c| [c[0], c[1], c[2], 1.0]).collect();
        q.write_texture(input.as_image_copy(), bytemuck::cast_slice(&rgba), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 16), rows_per_image: None }, input.size());
        let src = self.texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, self.render);
        let texels: Vec<f32> = depth_m
            .iter()
            .zip(velocity_px)
            .flat_map(|(&z, v)| {
                let ndc = if z >= FAR { 1.0 } else { FAR * (z - NEAR) / (z * (FAR - NEAR)) };
                let uv = if v[0] >= GBuffer::NO_VELOCITY { *v } else { [v[0] / W as f32, v[1] / H as f32] };
                [ndc, uv[0], uv[1], 0.0]
            })
            .collect();
        q.write_texture(src.as_image_copy(), bytemuck::cast_slice(&texels), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.render.0 * 16), rows_per_image: None }, src.size());
        let depth = self.texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, self.render);
        let output = self.texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, (W, H));
        let (input_view, depth_view, output_view) = (input.create_view(&Default::default()), depth.create_view(&Default::default()), output.create_view(&Default::default()));
        let src_view = src.create_view(&Default::default());
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) }],
        });

        let mut encoder = d.create_command_encoder(&Default::default());
        {
            let mut attachments: [Option<wgpu::RenderPassColorAttachment>; GBuffer::VELOCITY_TARGET + 1] = Default::default();
            attachments[GBuffer::VELOCITY_TARGET] = Some(wgpu::RenderPassColorAttachment {
                view: &self.gbuffer.velocity_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        effect.render(d, q, &mut encoder, &self.gbuffer, &input_view, &depth_view, &output_view, camera, W, H);
        let row = (W * 8).div_ceil(256) * 256;
        let readback = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
            output.size(),
        );
        q.submit(std::iter::once(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        d.poll(wgpu::Maintain::Wait);
        let data = readback.slice(..).get_mapped_range();
        let mut out = Vec::with_capacity((W * H) as usize);
        for y in 0..H {
            for x in 0..W {
                let o = (y * row + x * 8) as usize;
                let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
                out.push([c(0), c(1), c(2)]);
            }
        }
        out
    }
}

fn at(img: &[[f32; 3]], x: u32, y: u32) -> [f32; 3] {
    img[(y * W + x) as usize]
}

fn worst_difference(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs() / b[i].abs().max(1.0)).fold(0.0, f32::max)).fold(0.0, f32::max)
}

fn checker() -> Vec<[f32; 3]> {
    (0..W * H).map(|i| if (i % W / 3 + i / W / 3).is_multiple_of(2) { [1.0, 0.5, 0.25] } else { [0.0, 0.1, 0.2] }).collect()
}

/// Energy-weighted horizontal spread of the red channel around column `cx`: (energy per row,
/// radius of the box with that variance).
fn horizontal_box(img: &[[f32; 3]], cx: u32) -> (f32, f32) {
    let (mut total, mut m2) = (0.0f32, 0.0f32);
    for y in 0..H {
        for x in 0..W {
            let e = at(img, x, y)[0];
            let dx = x as f32 - cx as f32;
            total += e;
            m2 += e * dx * dx;
        }
    }
    // a box from -r to r (in whole pixels) has variance r (r + 1) / 3
    let var = m2 / total;
    (total / H as f32, (0.25 + 3.0 * var).sqrt() - 0.5)
}

#[test]
fn a_still_camera_or_a_cut_leaves_the_image_untouched() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let color = checker();
    let depth = vec![10.0; (W * H) as usize];
    let none = vec![NONE; (W * H) as usize];
    let mut fx = effect(1.0, 0.05);
    let still = h.run(&mut fx, &camera_moved(0.0), &color, &depth, &none);
    assert!(worst_difference(&still, &color) < 2e-3, "a still camera changed pixels by {}", worst_difference(&still, &color));
    // the camera jumped 2 m, but it is a cut
    let mut cut = camera_moved(2.0);
    cut.reset_motion();
    let out = h.run(&mut fx, &cut, &color, &depth, &none);
    assert!(worst_difference(&out, &color) < 2e-3, "a cut blurred by {}", worst_difference(&out, &color));
    // as is the frame after reset()
    fx.reset();
    let out = h.run(&mut fx, &camera_moved(2.0), &color, &depth, &none);
    assert!(worst_difference(&out, &color) < 2e-3, "a reset frame blurred by {}", worst_difference(&out, &color));
    // and the frame after that blurs again
    let out = h.run(&mut fx, &camera_moved(2.0), &color, &depth, &none);
    assert!(worst_difference(&out, &color) > 0.1, "no blur after the reset");
}

#[test]
fn a_panning_camera_blurs_by_the_predicted_length() {
    for scale in [1.0, 0.75, 0.5] {
        panning_camera_blurs_by_the_predicted_length(scale);
    }
}

/// At a render scale below 1 the depth is smaller than the picture, as after a temporal
/// upscaler: the blur is still measured in the picture's pixels.
fn panning_camera_blurs_by_the_predicted_length(scale: f32) {
    let Some(h) = Harness::with_render_scale(scale) else { return eprintln!("no GPU adapter: skipping") };
    let texels = (h.render.0 * h.render.1) as usize;
    // a bright vertical line on a wall 10 m away; the camera slides 32 px worth to the side
    let (cx, depth_m, shift_px) = (W / 2, 10.0, 32.0);
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if i % W == cx { [100.0, 100.0, 100.0] } else { [0.0; 3] }).collect();
    let depth = vec![depth_m; texels];
    let none = vec![NONE; texels];
    let camera = camera_moved(shift_px * depth_m / focal_px());
    for (amount, max, radius) in [(1.0, 0.05, 16.0), (0.5, 0.05, 8.0), (1.0, 0.02, 0.02 * W as f32)] {
        let out = h.run(&mut effect(amount, max), &camera, &color, &depth, &none);
        let (energy, r) = horizontal_box(&out, cx);
        assert!((energy - 100.0).abs() < 10.0 && (r - radius).abs() < 0.06 * radius, "scale {scale}, amount {amount}, max {max}: radius {r} px (predicted {radius}), energy per row {energy}");
        // nothing past the streak's end
        assert!(at(&out, cx + radius as u32 + 3, H / 2)[0] < 1e-3 && at(&out, cx - radius as u32 - 3, H / 2)[0] < 1e-3);
        eprintln!("scale {scale}, amount {amount}, max {max}: radius {r:.2} px (predicted {radius:.2}), energy per row {energy:.1}");
    }
}

/// Whether render texel `i` of a `render`-sized buffer lies over the picture's rectangle
/// [x0, x1) x [y0, y1) (its centre, in picture pixels).
fn texel_inside(i: u32, render: (u32, u32), (x0, x1, y0, y1): (u32, u32, u32, u32)) -> bool {
    let x = (i % render.0) as f32 + 0.5;
    let y = (i / render.0) as f32 + 0.5;
    let (px, py) = (x * W as f32 / render.0 as f32, y * H as f32 / render.1 as f32);
    px >= x0 as f32 && px < x1 as f32 && py >= y0 as f32 && py < y1 as f32
}

#[test]
fn a_moving_object_smears_over_a_static_background_without_halos() {
    for scale in [1.0, 0.5] {
        moving_object_smears_over_a_static_background_without_halos(scale);
    }
}

fn moving_object_smears_over_a_static_background_without_halos(scale: f32) {
    let Some(h) = Harness::with_render_scale(scale) else { return eprintln!("no GPU adapter: skipping") };
    // a red square 5 m away moving 24 px a frame to the right, over a blue-striped wall at 20 m
    let rect = (150u32, 200u32, 70u32, 120u32);
    let (x0, x1, y0, y1) = rect;
    let inside = |i: u32| (x0..x1).contains(&(i % W)) && (y0..y1).contains(&(i / W));
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if inside(i) { [1.0, 0.0, 0.0] } else { [0.0, 0.0, if (i % W / 4).is_multiple_of(2) { 1.0 } else { 0.5 }] }).collect();
    let texels = h.render.0 * h.render.1;
    let depth: Vec<f32> = (0..texels).map(|i| if texel_inside(i, h.render, rect) { 5.0 } else { 20.0 }).collect();
    let velocity: Vec<[f32; 2]> = (0..texels).map(|i| if texel_inside(i, h.render, rect) { [24.0, 0.0] } else { NONE }).collect();
    let out = h.run(&mut effect(1.0, 0.05), &camera_moved(0.0), &color, &depth, &velocity);
    let y = (y0 + y1) / 2;
    // the square's middle stays solid red
    let c = at(&out, (x0 + x1) / 2, y);
    assert!(c[0] > 0.97 && c[2] < 0.03, "middle {c:?}");
    // it smears over the wall on both sides, fading out by the end of its 12 px streak
    let (right, left) = (at(&out, x1 + 3, y), at(&out, x0 - 4, y));
    assert!(right[0] > 0.2 && right[0] < 0.8 && left[0] > 0.2 && left[0] < 0.8, "streaks {left:?} {right:?}");
    assert!(at(&out, x1 + 3, y)[0] > at(&out, x1 + 9, y)[0], "the streak fades");
    for x in [x0 - 16, x1 + 15] {
        assert!(worst_difference(&[at(&out, x, y)], &[at(&color, x, y)]) < 2e-3, "past the streak x={x}: {:?}", at(&out, x, y));
    }
    // the static wall above and below it stays sharp
    for (x, y) in [((x0 + x1) / 2, y0 - 3), ((x0 + x1) / 2, y1 + 2), (x1 + 3, y0 - 3)] {
        assert!(worst_difference(&[at(&out, x, y)], &[at(&color, x, y)]) < 2e-3, "wall at ({x}, {y}): {:?}", at(&out, x, y));
    }
    eprintln!("scale {scale}: streak left {left:?}, right {right:?}");
}

#[test]
fn a_tracked_foreground_stays_sharp_over_a_panning_background() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    // the camera pans with a red square 5 m away (its motion vectors say it stays put) over a
    // blue-striped wall 20 m away that slides 20 px
    let (x0, x1, y0, y1) = (150u32, 200u32, 70u32, 120u32);
    let inside = |i: u32| (x0..x1).contains(&(i % W)) && (y0..y1).contains(&(i / W));
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if inside(i) { [1.0, 0.0, 0.0] } else { [0.0, 0.0, if (i % W / 4).is_multiple_of(2) { 1.0 } else { 0.5 }] }).collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if inside(i) { 5.0 } else { 20.0 }).collect();
    let velocity: Vec<[f32; 2]> = (0..W * H).map(|i| if inside(i) { [0.0, 0.0] } else { NONE }).collect();
    let camera = camera_moved(20.0 * 20.0 / focal_px());
    let out = h.run(&mut effect(1.0, 0.05), &camera, &color, &depth, &velocity);
    let y = (y0 + y1) / 2;
    // the square is sharp to its edges: no background over it
    for x in [x0, x0 + 1, (x0 + x1) / 2, x1 - 2, x1 - 1] {
        let c = at(&out, x, y);
        assert!(c[0] > 0.99 && c[2] < 0.01, "square x={x}: {c:?}");
    }
    // and does not bleed onto the background next to it
    for x in (x0 - 6..x0).chain(x1..x1 + 6) {
        assert!(at(&out, x, y)[0] < 0.01, "halo x={x}: {:?}", at(&out, x, y));
    }
    // the background away from it is blurred: its 8 px stripes average out
    for x in [40u32, 100, 300] {
        let b = at(&out, x, y)[2];
        assert!(b > 0.65 && b < 0.85, "background x={x}: {b}");
    }
}

/// Cost at 1920x1080 of a cut (a copy), nothing moving, and everything moving at the cap (the
/// worst case): wall time of submit-and-wait minus an empty submit (median of 100). Run by hand
/// on an idle machine:
/// `cargo test -p kansei-core --release --test motion_blur_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn motion_blur_frame_cost_at_1080p() {
    let Some((device, queue)) = gpu() else { return };
    let (w, h) = (1920u32, 1080u32);
    let tex = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let input = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING);
    let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
    let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
    let (input_view, depth_view, output_view) = (input.create_view(&Default::default()), depth.create_view(&Default::default()), output.create_view(&Default::default()));
    let gbuffer = GBuffer::new(&device, w, h, 1);
    let clear = |velocity: f64| {
        let mut encoder = device.create_command_encoder(&Default::default());
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &gbuffer.velocity_view,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: velocity, g: 0.0, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store },
        })];
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.99), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        queue.submit(std::iter::once(encoder.finish()));
    };
    let time = |f: &mut dyn FnMut()| {
        let mut samples = Vec::new();
        for i in 0..120 {
            let t0 = std::time::Instant::now();
            f();
            device.poll(wgpu::Maintain::Wait);
            if i >= 20 {
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };
    let empty = time(&mut || {
        queue.submit(std::iter::once(device.create_command_encoder(&Default::default()).finish()));
    });
    let run = |fx: &mut MotionBlurEffect, camera: &Camera| {
        time(&mut || {
            let mut encoder = device.create_command_encoder(&Default::default());
            fx.render(&device, &queue, &mut encoder, &gbuffer, &input_view, &depth_view, &output_view, camera, w, h);
            queue.submit(std::iter::once(encoder.finish()));
        }) - empty
    };
    let mut camera = camera_moved(0.0);
    let mut fx = MotionBlurEffect::new(MotionBlurOptions { amount: 0.5, max: 0.04, ..Default::default() });
    clear(0.0);
    let still = run(&mut fx, &camera);
    clear(0.2); // 384 px a frame: every pixel at the cap
    let moving = run(&mut fx, &camera);
    camera.reset_motion();
    let cut = run(&mut fx, &camera);
    eprintln!("MotionBlurEffect at {w}x{h}: cut (a copy) {cut:.3} ms, still {still:.3} ms, everything at the cap {moving:.3} ms (empty submit {empty:.3} ms)");
}
