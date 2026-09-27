//! Runs CinematicDepthOfFieldEffect on synthetic images on a real GPU and checks the physics
//! that matters for the look: focus stays sharp, a point of light becomes a disc of its circle
//! of confusion with its energy kept, depth edges don't halo either way, and a blurred
//! foreground spills over what is behind it. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions};
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

fn effect() -> CinematicDepthOfFieldEffect {
    CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions { lens: lens(), sample_count: 96, ..Default::default() })
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
    // f/0.28: a CoC near the 48 px cap, where the gather takes more samples
    let mut fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
        lens: CameraLens { f_stop: 0.28, ..lens() },
        sample_count: 72,
        ..Default::default()
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

/// Cost of the effect at 1920x1080 with the largest blur everywhere (the worst case): wall time
/// of submit-and-wait minus an empty submit. Run by hand:
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
    let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
    let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
    let (input_view, depth_view, output_view) = (input.create_view(&Default::default()), depth.create_view(&Default::default()), output.create_view(&Default::default()));
    // depth 1.0 everywhere: the sky, with the largest CoC
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    queue.submit(std::iter::once(encoder.finish()));
    let gbuffer = GBuffer::new(&device, w, h, 1);
    let camera = Camera::new(40.0, NEAR, FAR, w as f32 / h as f32);
    let mut fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
        lens: CameraLens { focal_length_mm: Some(85.0), f_stop: 1.4, focus_distance_m: 2.0, sensor_width_mm: 23.76, ..Default::default() },
        ..Default::default()
    });
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
    let dof = time(&mut || {
        let mut encoder = device.create_command_encoder(&Default::default());
        fx.render(&device, &queue, &mut encoder, &gbuffer, &input_view, &depth_view, &output_view, &camera, w, h);
        queue.submit(std::iter::once(encoder.finish()));
    });
    eprintln!("CinematicDepthOfFieldEffect at {w}x{h}, CoC {:.0} px everywhere: {:.3} ms (empty submit {:.3} ms)", fx.coc_radius_px(&camera, w, FAR), dof - empty, empty);
}
