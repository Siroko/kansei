//! Runs CinematicDepthOfFieldEffect on synthetic images on a real GPU and checks the physics
//! that matters for the look: focus stays sharp, a point of light becomes a disc of its circle
//! of confusion with its energy kept, depth edges don't halo either way, and a blurred
//! foreground spills over what is behind it. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, HighlightOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

const W: u32 = 384;
const H: u32 = 192;
const NEAR: f32 = 0.1;
const FAR: f32 = 1000.0;

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

/// 100 mm f/1 focused at 5 m on a 36 mm filmback: 384 px -> CoC radius 10.9 px at infinity.
fn lens() -> CameraLens {
    CameraLens { focal_length_mm: Some(100.0), f_stop: 1.0, focus_distance_m: 5.0, sensor_width_mm: 36.0, ..Default::default() }
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gbuffer: GBuffer,
    depth_pipeline: wgpu::RenderPipeline,
    depth_bgl: wgpu::BindGroupLayout,
}

impl Harness {
    fn new() -> Option<Self> {
        let (device, queue) = gpu()?;
        // writes a depth buffer from an r32float texture of NDC depths
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var depths : texture_2d<f32>;
                @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
                    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
                    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
                }
                @fragment fn fs(@builtin(position) pos : vec4f) -> @builtin(frag_depth) f32 {
                    return textureLoad(depths, vec2u(pos.xy), 0).r;
                }
                "#
                .into(),
            ),
        });
        let depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&depth_bgl], push_constant_ranges: &[] });
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
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
        let gbuffer = GBuffer::new(&device, W, H, 1);
        Some(Self { device, queue, gbuffer, depth_pipeline, depth_bgl })
    }

    fn texture(&self, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// Run the effect on `color` (rgb per pixel) at view depths `depth_m` (metres); rgb out.
    fn run(&self, effect: &mut CinematicDepthOfFieldEffect, color: &[[f32; 3]], depth_m: &[f32]) -> Vec<[f32; 3]> {
        let (d, q) = (&self.device, &self.queue);
        let input = self.texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let rgba: Vec<f32> = color.iter().flat_map(|c| [c[0], c[1], c[2], 1.0]).collect();
        q.write_texture(input.as_image_copy(), bytemuck::cast_slice(&rgba), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 16), rows_per_image: None }, input.size());
        let ndc = self.texture(wgpu::TextureFormat::R32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let ndc_depth: Vec<f32> = depth_m.iter().map(|&z| if z >= FAR { 1.0 } else { FAR * (z - NEAR) / (z * (FAR - NEAR)) }).collect();
        q.write_texture(ndc.as_image_copy(), bytemuck::cast_slice(&ndc_depth), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: None }, ndc.size());
        let depth = self.texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let output = self.texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let (input_view, depth_view, output_view) = (input.create_view(&Default::default()), depth.create_view(&Default::default()), output.create_view(&Default::default()));
        let ndc_view = ndc.create_view(&Default::default());
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.depth_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ndc_view) }],
        });

        let mut encoder = d.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
        effect.render(d, q, &mut encoder, &self.gbuffer, &input_view, &depth_view, &output_view, &camera, W, H);
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

/// The test pictures are small (W px) with film-sized blur: lift the width-relative cap, leaving
/// the pixel ceiling.
fn options() -> CinematicDepthOfFieldOptions {
    CinematicDepthOfFieldOptions { lens: lens(), max_coc_fraction: 1.0, ..Default::default() }
}

fn effect() -> CinematicDepthOfFieldEffect {
    CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions { sample_count: 96, ..options() })
}

fn at(img: &[[f32; 3]], x: u32, y: u32) -> [f32; 3] {
    img[(y * W + x) as usize]
}

#[test]
fn in_focus_images_pass_through_untouched() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if (i % W / 3 + i / W / 3).is_multiple_of(2) { [1.0, 0.5, 0.25] } else { [0.0, 0.1, 0.2] }).collect();
    let depth = vec![5.0; (W * H) as usize];
    let out = h.run(&mut effect(), &color, &depth);
    let worst = out.iter().zip(&color).map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max);
    assert!(worst < 2e-3, "in-focus pixels changed by {worst}");
}

#[test]
fn a_point_light_becomes_a_disc_of_its_coc_with_its_energy_kept() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = effect();
    let depth_m = 100.0;
    let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
    let coc = fx.coc_radius_px(&camera, W, depth_m);
    let (cx, cy) = (W / 2, H / 2);
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    color[(cy * W + cx) as usize] = [1000.0, 1000.0, 1000.0];
    let out = h.run(&mut fx, &color, &vec![depth_m; (W * H) as usize]);
    let total: f32 = out.iter().map(|c| c[0]).sum();
    assert!((total - 1000.0).abs() < 150.0, "energy {total} (CoC {coc} px)");
    // lit inside the disc, dark well outside it, and roughly flat inside
    let inside = at(&out, cx + (coc * 0.5) as u32, cy)[0];
    let expected = 1000.0 / (std::f32::consts::PI * coc * coc);
    assert!(inside > 0.4 * expected && inside < 2.5 * expected, "inside {inside}, flat disc {expected} (CoC {coc} px)");
    assert!(at(&out, cx + (coc * 1.6) as u32, cy)[0] < 0.1 * expected, "outside {}", at(&out, cx + (coc * 1.6) as u32, cy)[0]);
    eprintln!("CoC {coc:.2} px: energy {total:.0}, inside {inside:.2} (flat disc {expected:.2})");
}

#[test]
fn depth_edges_do_not_halo() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let edge = W / 2;
    // left: a sharp red object on the focus plane; right: a far blue background, textured
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if i % W < edge { [1.0, 0.0, 0.0] } else { [0.0, 0.0, if (i / W / 4).is_multiple_of(2) { 1.0 } else { 0.5 }] }).collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W < edge { 5.0 } else { 200.0 }).collect();
    let out = h.run(&mut effect(), &color, &depth);
    let y = H / 2;
    // the sharp object stays pure red up to its edge: no background bleeds over it
    for x in edge - 6..edge {
        let c = at(&out, x, y);
        assert!(c[2] < 0.02 && c[0] > 0.98, "sharp side x={x}: {c:?}");
    }
    // the blurred background next to it takes no red from the object in front of it
    for x in edge + 2..edge + 12 {
        let c = at(&out, x, y);
        assert!(c[0] < 0.05, "blurred side x={x}: {c:?}");
    }
    // and it is blurred: its stripes average out
    let b = at(&out, edge + 40, y)[2];
    assert!(b > 0.6 && b < 0.9, "background {b}");
}

#[test]
fn a_blurred_foreground_spills_over_what_is_behind_it() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = effect();
    let edge = W / 2;
    // left: a green object 2.5 m away, well in front of the 5 m focus; right: white, in focus
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if i % W < edge { [0.0, 1.0, 0.0] } else { [1.0, 1.0, 1.0] }).collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W < edge { 2.5 } else { 5.0 }).collect();
    let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
    let coc = -fx.coc_radius_px(&camera, W, 2.5);
    let out = h.run(&mut fx, &color, &depth);
    let y = H / 2;
    // just past the silhouette the in-focus white is partly covered by the green's blur...
    let near = at(&out, edge + 2, y);
    assert!(near[0] < 0.9 && near[0] > 0.1, "next to the edge {near:?} (CoC {coc} px)");
    // ...and fully clear beyond its reach, where the white is untouched and sharp
    let far = at(&out, edge + (coc * 1.5) as u32 + 2, y);
    assert!(far.iter().all(|&c| (c - 1.0).abs() < 0.01), "beyond the blur {far:?}");
    eprintln!("foreground CoC {coc:.2} px: next to the edge {near:?}");
}

#[test]
fn a_near_point_light_becomes_a_flat_disc_over_what_is_behind_it() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = effect();
    // a light 2.5 m away, in front of the 5 m focus, over an in-focus dark wall
    let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
    let coc = -fx.coc_radius_px(&camera, W, 2.5);
    let (cx, cy) = (W / 2, H / 2);
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    let mut depth = vec![5.0f32; (W * H) as usize];
    color[(cy * W + cx) as usize] = [1000.0, 1000.0, 1000.0];
    depth[(cy * W + cx) as usize] = 2.5;
    let out = h.run(&mut fx, &color, &depth);
    let total: f32 = out.iter().map(|c| c[0]).sum();
    assert!((total - 1000.0).abs() < 150.0, "energy {total} (CoC {coc} px)");
    let expected = 1000.0 / (std::f32::consts::PI * coc * coc);
    // flat: no bright core left at the light's own pixel
    let centre = at(&out, cx, cy)[0];
    let half = at(&out, cx + (coc * 0.5) as u32, cy)[0];
    assert!(centre < 2.0 * expected && half > 0.4 * expected && half < 2.0 * expected, "centre {centre}, half radius {half}, flat {expected}");
    eprintln!("near CoC {coc:.2} px: energy {total:.0}, centre {centre:.2}, half radius {half:.2} (flat {expected:.2})");
}

#[test]
fn large_bokeh_keep_their_energy() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    // f/0.28: a CoC of about 37 px, where the gather reads the coarsest level
    let mut fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
        lens: CameraLens { f_stop: 0.28, ..lens() },
        sample_count: 72,
        ..options()
    });
    let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
    let coc = fx.coc_radius_px(&camera, W, 100.0);
    let (cx, cy) = (W / 2, H / 2);
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    color[(cy * W + cx) as usize] = [10000.0, 10000.0, 10000.0];
    let out = h.run(&mut fx, &color, &vec![100.0; (W * H) as usize]);
    let total: f32 = out.iter().map(|c| c[0]).sum();
    assert!(coc > 30.0 && (total - 10000.0).abs() < 1500.0, "energy {total} (CoC {coc} px)");
    eprintln!("CoC {coc:.2} px: energy {total:.0} of 10000");
}

/// Cost of the effect at 1920x1080: the worst case (the largest blur on every pixel) and a
/// typical frame (in focus but for a blurred band), with and without scattered highlights. Wall
/// time of ten frames per submit-and-wait, minus an empty submit, per frame. Run by hand:
/// `cargo test -p kansei-core --release --test dof_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn dof_frame_cost_at_1080p() {
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
    let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
    let (input_view, output_view) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
    let depth_at = |value: f32| {
        let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(value), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        queue.submit(std::iter::once(encoder.finish()));
        view
    };
    // depth 1.0: the sky, the largest CoC; the focus plane at 2 m: in focus
    let sky = depth_at(1.0);
    let focus_ndc = FAR * (2.0 - NEAR) / (2.0 * (FAR - NEAR));
    let in_focus = depth_at(focus_ndc);
    let gbuffer = GBuffer::new(&device, w, h, 1);
    let camera = Camera::new(40.0, NEAR, FAR, w as f32 / h as f32);
    let time = |f: &mut dyn FnMut()| {
        let mut samples = Vec::new();
        for i in 0..60 {
            let t0 = std::time::Instant::now();
            f();
            device.poll(wgpu::Maintain::Wait);
            if i >= 10 {
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };
    let empty = time(&mut || {
        queue.submit(std::iter::once(device.create_command_encoder(&Default::default()).finish()));
    });
    const FRAMES: u32 = 10;
    for (label, depth, scatter, samples) in [
        ("worst case, 48 px everywhere", &sky, true, 72),
        ("worst case, no scatter", &sky, false, 72),
        ("worst case, 48 samples", &sky, true, 48),
        ("all in focus", &in_focus, true, 72),
    ] {
        let mut fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
            lens: CameraLens { focal_length_mm: Some(85.0), f_stop: 1.4, focus_distance_m: 2.0, sensor_width_mm: 23.76, ..Default::default() },
            sample_count: samples,
            highlights: HighlightOptions { enabled: scatter, ..Default::default() },
            ..Default::default()
        });
        let t = time(&mut || {
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..FRAMES {
                fx.render(&device, &queue, &mut encoder, &gbuffer, &input_view, depth, &output_view, &camera, w, h);
            }
            queue.submit(std::iter::once(encoder.finish()));
        });
        eprintln!("CinematicDepthOfFieldEffect at {w}x{h}, {label}: {:.3} ms per frame", (t - empty) / FRAMES as f64);
    }
}

#[test]
fn the_gather_alone_keeps_a_point_light_energy_too() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
        sample_count: 96,
        highlights: HighlightOptions { enabled: false, ..Default::default() },
        ..options()
    });
    let (cx, cy) = (W / 2, H / 2);
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    color[(cy * W + cx) as usize] = [1000.0, 1000.0, 1000.0];
    let out = h.run(&mut fx, &color, &vec![100.0; (W * H) as usize]);
    let total: f32 = out.iter().map(|c| c[0]).sum();
    assert!((total - 1000.0).abs() < 150.0, "energy {total}");
}

#[test]
fn a_scattered_highlight_does_not_shine_over_a_sharper_object_in_front() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let edge = W / 2;
    let y = H / 2;
    // left: a dark object on the focus plane; right: a black far background with one bright light
    // 4 px past the edge
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    color[(y * W + edge + 4) as usize] = [1000.0, 1000.0, 1000.0];
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W < edge { 5.0 } else { 100.0 }).collect();
    let out = h.run(&mut effect(), &color, &depth);
    for x in edge - 8..edge {
        let c = at(&out, x, y)[0];
        assert!(c < 0.05, "sharp side x={x}: {c}");
    }
    let disc = at(&out, edge + 8, y)[0];
    assert!(disc > 0.5, "the bokeh behind it {disc}");
}

/// Largest change between neighbouring pixels of a row, over [x0, x1).
fn max_step(img: &[[f32; 3]], y: u32, x0: u32, x1: u32, channel: usize) -> f32 {
    (x0..x1 - 1).map(|x| (at(img, x + 1, y)[channel] - at(img, x, y)[channel]).abs()).fold(0.0, f32::max)
}

#[test]
fn a_blurred_foreground_fades_smoothly_across_its_silhouette() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = effect();
    let edge = W / 2;
    // an opaque green object 2.5 m away (CoC ~11 px) over an in-focus white wall
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if i % W < edge { [0.0, 1.0, 0.0] } else { [1.0, 1.0, 1.0] }).collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W < edge { 2.5 } else { 5.0 }).collect();
    let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
    let coc = -fx.coc_radius_px(&camera, W, 2.5);
    let out = h.run(&mut fx, &color, &depth);
    let y = H / 2;
    let (x0, x1) = (edge - 2 * coc as u32, edge + 2 * coc as u32);
    // the red channel ramps from the object (0) to the wall (1) with no step at the silhouette
    let step = max_step(&out, y, x0, x1, 0);
    assert!(step < 0.12, "largest step {step} across a {coc:.1} px blur");
    assert!(at(&out, x0, y)[0] < 0.05 && at(&out, x1, y)[0] > 0.95);
    eprintln!("opaque near edge: largest step {step:.3}");
}

#[test]
fn a_porous_foreground_shows_the_background_through_it_without_a_seam() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let mut fx = effect();
    let edge = W / 2;
    // left half: a green screen 2.5 m away with every other column open (like leaves or a fence),
    // over an in-focus white wall seen through it; right half: the wall alone
    let screen = |i: u32| i % W < edge && (i % W).is_multiple_of(2);
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if screen(i) { [0.0, 1.0, 0.0] } else { [1.0, 1.0, 1.0] }).collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if screen(i) { 2.5 } else { 5.0 }).collect();
    let out = h.run(&mut fx, &color, &depth);
    let y = H / 2;
    // well inside, half the aperture sees the wall through the gaps
    let inside = at(&out, edge / 2, y)[0];
    assert!((inside - 0.5).abs() < 0.12, "through the screen {inside}");
    // across the screen's edge, a smooth fade: no seam where the gaps end
    let step = max_step(&out, y, edge - 30, edge + 30, 0);
    assert!(step < 0.08, "largest step {step} across the screen's edge");
    assert!(at(&out, edge + 30, y)[0] > 0.95, "past the blur {:?}", at(&out, edge + 30, y));
    eprintln!("porous near screen: {inside:.3} through it, largest step across its edge {step:.3}");
}

#[test]
fn a_thin_foreground_line_keeps_its_occlusion_when_spread() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let x_line = W / 2;
    // a 1 px black line (a wire, a twig) 2.5 m away over an in-focus white wall
    let color: Vec<[f32; 3]> = (0..W * H).map(|i| if i % W == x_line { [0.0, 0.0, 0.0] } else { [1.0, 1.0, 1.0] }).collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W == x_line { 2.5 } else { 5.0 }).collect();
    let out = h.run(&mut effect(), &color, &depth);
    let y = H / 2;
    let occlusion: f32 = (0..W).map(|x| 1.0 - at(&out, x, y)[0]).sum();
    let deepest = (0..W).map(|x| 1.0 - at(&out, x, y)[0]).fold(0.0, f32::max);
    assert!((occlusion - 1.0).abs() < 0.25 && deepest < 0.2, "occlusion {occlusion} px, deepest {deepest}");
    eprintln!("thin near line: occlusion {occlusion:.3} px, deepest {deepest:.3}");
}

#[test]
fn the_background_hidden_behind_a_blurred_foreground_continues_the_visible_one() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let edge = W / 2;
    // left: a green object 2.5 m away (CoC ~11 px); right: the in-focus background, a dark band
    // 6 px wide along the object's silhouette, then white. Behind the object's blurred edge the
    // hidden background must continue the dark band: if it is filled from the white beyond, the
    // silhouette shows through the blur as a hard line.
    let color: Vec<[f32; 3]> = (0..W * H)
        .map(|i| {
            let x = i % W;
            if x < edge { [0.0, 1.0, 0.0] } else if x < edge + 6 { [0.1, 0.1, 0.1] } else { [1.0, 1.0, 1.0] }
        })
        .collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W < edge { 2.5 } else { 5.0 }).collect();
    let out = h.run(&mut effect(), &color, &depth);
    let y = H / 2;
    // red: 0 in the object, 0.1 in the band, 1 beyond; across the silhouette it must not jump
    let step = (edge - 3..edge + 3).map(|x| (at(&out, x + 1, y)[0] - at(&out, x, y)[0]).abs()).fold(0.0, f32::max);
    assert!(step < 0.08, "largest step at the silhouette {step}: {:?}", (edge - 3..edge + 4).map(|x| at(&out, x, y)[0]).collect::<Vec<_>>());
    eprintln!("hidden background: largest step at the silhouette {step:.3}");
}

#[test]
fn a_blurred_distance_does_not_flood_the_background_hidden_behind_a_near_object() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    let edge = W / 2;
    // left: a green object 2.5 m away (CoC ~11 px); right: an in-focus dark pillar 6 px wide
    // along its silhouette, then a white distance 100 m away (CoC ~10 px). The pillar hides the
    // distance, so its blur must not reach the pillar, visible or hidden behind the object: if it
    // floods the hidden part, the silhouette shows through the object's blur as a hard line.
    let color: Vec<[f32; 3]> = (0..W * H)
        .map(|i| {
            let x = i % W;
            if x < edge { [0.0, 1.0, 0.0] } else if x < edge + 6 { [0.1, 0.1, 0.1] } else { [1.0, 1.0, 1.0] }
        })
        .collect();
    let depth: Vec<f32> = (0..W * H).map(|i| if i % W < edge { 2.5 } else if i % W < edge + 6 { 5.0 } else { 100.0 }).collect();
    let out = h.run(&mut effect(), &color, &depth);
    let y = H / 2;
    let step = (edge - 6..edge + 3).map(|x| (at(&out, x + 1, y)[0] - at(&out, x, y)[0]).abs()).fold(0.0, f32::max);
    assert!(step < 0.08, "largest step at the silhouette {step}: {:?}", (edge - 6..edge + 4).map(|x| at(&out, x, y)[0]).collect::<Vec<_>>());
    eprintln!("blurred distance behind a pillar: largest step at the silhouette {step:.3}");
}

#[test]
fn a_dense_field_of_highlights_keeps_its_energy_when_the_bins_overflow() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    // 256 small lights 4 px apart, 100 m away (CoC ~10 px): more than a sprite bin lists, so
    // some are gathered instead of scattered, and none of their light is lost
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    let (x0, y0) = (W / 2 - 32, H / 2 - 32);
    for j in 0..16 {
        for i in 0..16 {
            color[((y0 + 4 * j) * W + x0 + 4 * i) as usize] = [100.0, 100.0, 100.0];
        }
    }
    let out = h.run(&mut effect(), &color, &vec![100.0; (W * H) as usize]);
    let total: f32 = out.iter().map(|c| c[0]).sum();
    assert!((total - 25600.0).abs() < 0.1 * 25600.0, "energy {total} of 25600");
    eprintln!("dense highlights: energy {total:.0} of 25600");
}

#[test]
fn bokeh_bigger_than_the_cap_stop_at_a_fraction_of_the_width() {
    let Some(h) = Harness::new() else { return eprintln!("no GPU adapter: skipping") };
    // f/0.28 gives a light 100 m away a CoC of about 37 px; the cap is 5 % of 384 px
    let mut fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
        lens: CameraLens { f_stop: 0.28, ..lens() },
        max_coc_fraction: 0.05,
        ..Default::default()
    });
    let camera = Camera::new(40.0, NEAR, FAR, W as f32 / H as f32);
    let cap = 0.05 * W as f32;
    assert_eq!(fx.coc_radius_px(&camera, W, 100.0), cap);
    let (cx, cy) = (W / 2, H / 2);
    let mut color = vec![[0.0f32; 3]; (W * H) as usize];
    color[(cy * W + cx) as usize] = [10000.0, 10000.0, 10000.0];
    let out = h.run(&mut fx, &color, &vec![100.0; (W * H) as usize]);
    // a flat disc's mean squared radius is half its radius squared
    let (mut total, mut r2) = (0.0f32, 0.0f32);
    for y in 0..H {
        for x in 0..W {
            let e = at(&out, x, y)[0];
            let (dx, dy) = (x as f32 + 0.5 - (cx as f32 + 0.5), y as f32 + 0.5 - (cy as f32 + 0.5));
            total += e;
            r2 += e * (dx * dx + dy * dy);
        }
    }
    let radius = (2.0 * r2 / total).sqrt();
    // (a one-texel light's energy is estimated from the samples that land on it: within 25 % here)
    assert!((radius - cap).abs() < 0.1 * cap && (total - 10000.0).abs() < 2500.0, "disc radius {radius} px (cap {cap}), energy {total}");
    eprintln!("capped bokeh: radius {radius:.2} px (cap {cap:.2}), energy {total:.0}");
}
