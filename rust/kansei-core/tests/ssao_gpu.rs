//! Runs SSAOEffect on analytic scenes on a real GPU (their depth, normals, albedo and emission
//! written into a GBuffer, their lit colour as the input): flat surfaces facing the camera or
//! seen at a slant read unoccluded and keep their colour; where a floor meets a wall the
//! occlusion takes out only the ambient, never the direct light or the emission, at a room's
//! scale and at a tabletop's. Skipped (passes) when no adapter is available.

use glam::{Vec2, Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{SSAOAmbient, SSAOEffect, SSAOOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

const W: u32 = 320;
const H: u32 = 240;
/// Every surface's light: the direct part, its albedo, and the ambient (`AMBIENT` times the
/// albedo, a constant hemisphere).
const DIRECT: f32 = 0.3;
const ALBEDO: f32 = 0.5;
const AMBIENT: f32 = 0.8;

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

/// What a camera ray meets: how far, and the surface's normal there.
struct Hit {
    t: f32,
    normal: Vec3,
}

/// A camera at `eye` looking at `target`.
fn camera(eye: Vec3, target: Vec3, near: f32) -> Camera {
    let mut camera = Camera::new(60.0, near, near * 1000.0, W as f32 / H as f32);
    camera.set_position(eye.x, eye.y, eye.z);
    camera.look_at(&kansei_core::math::Vec3::new(target.x, target.y, target.z));
    camera.update_view_matrix();
    camera
}

/// The world ray through a pixel's centre.
fn pixel_ray(camera: &Camera, x: u32, y: u32) -> (Vec3, Vec3) {
    let inv_vp = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
    let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
    let ndc = Vec2::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0);
    let far = inv_vp * Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    (eye, (far.truncate() / far.w - eye).normalize())
}

/// Each pixel's input colour and the effect's output (rgb), and the hit there (none: sky).
struct Frame {
    input: Vec<[f32; 4]>,
    output: Vec<[f32; 4]>,
    hits: Vec<Option<Hit>>,
}

/// Renders `scene` into a GBuffer (`gbuffer_normals`: its normals written, else rebuilt from
/// depth by the effect; `emission` added to every surface and written to the emissive target)
/// and runs `fx` once.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, camera: &Camera, scene: &dyn Fn(Vec3, Vec3) -> Option<Hit>, gbuffer_normals: bool, emission: f32, fx: &mut SSAOEffect) -> Frame {
    let gbuffer = GBuffer::new(device, W, H, 1);
    let vp = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
    let n = (W * H) as usize;
    let (mut depth, mut normal, mut albedo, mut emissive, mut input) = (vec![1.0f32; n], vec![[0.0f32; 4]; n], vec![[0.0f32; 4]; n], vec![[0.0f32; 4]; n], vec![[0.0f32; 4]; n]);
    let mut hits = Vec::with_capacity(n);
    for y in 0..H {
        for x in 0..W {
            let (eye, dir) = pixel_ray(camera, x, y);
            let hit = scene(eye, dir);
            if let Some(hit) = &hit {
                let i = (y * W + x) as usize;
                let clip = vp * (eye + dir * hit.t).extend(1.0);
                depth[i] = clip.z / clip.w;
                if gbuffer_normals {
                    let e = hit.normal * 0.5 + 0.5;
                    normal[i] = [e.x, e.y, e.z, 1.0];
                }
                albedo[i] = [ALBEDO, ALBEDO, ALBEDO, 1.0];
                emissive[i] = [emission, emission, emission, 0.0];
                let c = DIRECT + ALBEDO * AMBIENT + emission;
                input[i] = [c, c, c, 1.0];
            }
            hits.push(hit);
        }
    }
    let buffer = |data: &[u8]| {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: wgpu::BufferUsages::STORAGE })
    };
    let (depth_buf, normal_buf, albedo_buf, emissive_buf) = (buffer(bytemuck::cast_slice(&depth)), buffer(bytemuck::cast_slice(&normal)), buffer(bytemuck::cast_slice(&albedo)), buffer(bytemuck::cast_slice(&emissive)));
    let code = format!(
        "@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
        @group(0) @binding(1) var<storage, read> n : array<vec4f>;\n\
        @group(0) @binding(2) var<storage, read> a : array<vec4f>;\n\
        @group(0) @binding(3) var<storage, read> e : array<vec4f>;\n\
        @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
        struct O {{ @builtin(frag_depth) depth : f32, @location(0) emissive : vec4f, @location(1) normal : vec4f, @location(2) albedo : vec4f }};\n\
        @fragment fn fs(@builtin(position) pos : vec4f) -> O {{ var o : O; let i = u32(pos.y) * {W}u + u32(pos.x); o.depth = d[i]; o.emissive = e[i]; o.normal = n[i]; o.albedo = a[i]; return o; }}"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
    let target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            targets: &[target(GBuffer::MRT_FORMATS[1]), target(GBuffer::MRT_FORMATS[2]), target(GBuffer::ALBEDO_FORMAT)],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState { format: GBuffer::DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: depth_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: normal_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: albedo_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: emissive_buf.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let attachment = |view| Some(wgpu::RenderPassColorAttachment { view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[attachment(&gbuffer.emissive_view), attachment(&gbuffer.normal_view), attachment(&gbuffer.albedo_view)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &gbuffer.depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit([encoder.finish()]);

    let extent = wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 };
    let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: extent, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
    let colour = texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &colour, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&input),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 16), rows_per_image: Some(H) },
        extent,
    );
    let output = texture(GBuffer::COLOR_FORMAT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
    let (colour_view, output_view) = (colour.create_view(&Default::default()), output.create_view(&Default::default()));
    fx.initialize(device, &gbuffer, camera);
    let mut encoder = device.create_command_encoder(&Default::default());
    fx.render(device, queue, &mut encoder, &gbuffer, &colour_view, &gbuffer.depth_view, &output_view, camera, W, H);
    let row = (W * 8).div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
        extent,
    );
    queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = staging.slice(..).get_mapped_range();
    let half = |o: usize| f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]));
    let output = (0..H * W)
        .map(|i| {
            let o = ((i / W) * row + (i % W) * 8) as usize;
            [half(o), half(o + 2), half(o + 4), half(o + 6)]
        })
        .collect();
    Frame { input, output, hits }
}

/// The floor y = 0 and, if `wall_z` is finite, a wall z = wall_z facing +z.
fn floor_and_wall(wall_z: f32) -> impl Fn(Vec3, Vec3) -> Option<Hit> {
    move |eye, dir| {
        let floor = (dir.y < 0.0).then(|| -eye.y / dir.y).map(|t| Hit { t, normal: Vec3::Y });
        let wall = (wall_z.is_finite() && dir.z < 0.0).then(|| (wall_z - eye.z) / dir.z).map(|t| Hit { t, normal: Vec3::Z });
        match (floor, wall) {
            (Some(f), Some(w)) => Some(if f.t < w.t { f } else { w }),
            (f, w) => f.or(w),
        }
    }
}

fn options(radius: f32) -> SSAOOptions {
    SSAOOptions { radius, kernel_size: 32, ..SSAOOptions::new(SSAOAmbient::Hemisphere { up: [AMBIENT; 3], down: [AMBIENT; 3] }) }
}

/// The largest change the effect made to any pixel's red channel.
fn largest_change(frame: &Frame) -> f32 {
    frame.input.iter().zip(&frame.output).map(|(i, o)| (i[0] - o[0]).abs()).fold(0.0, f32::max)
}

/// A wall facing the camera, and an open floor seen from above at a slant, are not occluded:
/// the colour comes out as it went in, with the GBuffer's normals and with normals rebuilt from
/// depth, at a room's scale and at a tabletop's.
#[test]
fn flat_surfaces_read_unoccluded() {
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    for scale in [1.0, 0.03] {
        let wall = |eye: Vec3, dir: Vec3| (dir.z < 0.0).then(|| Hit { t: (-5.0 * scale - eye.z) / dir.z, normal: Vec3::Z });
        let facing = camera(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), 0.1 * scale);
        let above = camera(Vec3::new(0.0, 1.5, 4.0) * scale, Vec3::new(0.0, 0.0, -3.0) * scale, 0.1 * scale);
        for (name, camera, scene) in [("wall", &facing, &wall as &dyn Fn(Vec3, Vec3) -> Option<Hit>), ("floor", &above, &floor_and_wall(f32::INFINITY))] {
            for gbuffer_normals in [true, false] {
                let mut fx = SSAOEffect::new(options(0.5 * scale));
                let frame = run(&device, &queue, camera, scene, gbuffer_normals, 0.0, &mut fx);
                let change = largest_change(&frame);
                assert!(change < 2e-3, "{name} at scale {scale} (GBuffer normals {gbuffer_normals}) changed by {change}");
            }
        }
    }
}

/// Where the floor meets a wall the floor and the wall are occluded, and the occlusion takes
/// out only ambient light: no pixel loses more than its ambient (albedo x `AMBIENT`) or any of
/// its direct light or emission. Far from the crease the colour is untouched. The same at a
/// tabletop's scale with a radius to match.
#[test]
fn a_crease_loses_only_its_ambient() {
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let emission = 0.25;
    for scale in [1.0, 0.03] {
        let camera = camera(Vec3::new(0.0, 1.2, 3.0) * scale, Vec3::new(0.0, 0.3, 0.0) * scale, 0.1 * scale);
        let scene = floor_and_wall(0.0);
        let mut fx = SSAOEffect::new(options(0.5 * scale));
        let frame = run(&device, &queue, &camera, &scene, true, emission, &mut fx);
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        let (mut crease, mut crease_n, mut open_change) = (0.0f32, 0, 0.0f32);
        for (i, hit) in frame.hits.iter().enumerate() {
            let Some(hit) = hit else { continue };
            let (x, y) = (i as u32 % W, i as u32 / W);
            let (_, dir) = pixel_ray(&camera, x, y);
            let p = (eye + dir * hit.t) / scale;
            let lost = frame.input[i][0] - frame.output[i][0];
            assert!(lost <= ALBEDO * AMBIENT + 2e-3, "scale {scale}: pixel {x},{y} lost {lost}, more than its ambient");
            assert!(frame.output[i][0] >= DIRECT + emission - 2e-3);
            // (in units of `scale`) the floor within 0.05 of the wall, the wall within 0.05 of the floor
            if hit.normal == Vec3::Y && p.z < 0.05 || hit.normal == Vec3::Z && p.y < 0.05 {
                crease += lost / (ALBEDO * AMBIENT);
                crease_n += 1;
            }
            if hit.normal == Vec3::Y && p.z > 1.0 || hit.normal == Vec3::Z && p.y > 1.0 {
                open_change = open_change.max(lost.abs());
            }
        }
        assert!(crease_n > 20, "scale {scale}: {crease_n} crease pixels");
        let crease = crease / crease_n as f32;
        assert!(crease > 0.2, "scale {scale}: the crease lost {crease} of its ambient");
        assert!(open_change < 2e-3, "scale {scale}: open surfaces changed by {open_change}");
    }
}

/// With an unlit ambient the colour less its emission is what the occlusion darkens: the
/// emission stays whole.
#[test]
fn unlit_ambient_keeps_the_emission() {
    let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
    let emission = 0.4;
    let camera = camera(Vec3::new(0.0, 1.2, 3.0), Vec3::new(0.0, 0.3, 0.0), 0.1);
    let mut fx = SSAOEffect::new(SSAOOptions { radius: 0.5, kernel_size: 32, ..SSAOOptions::new(SSAOAmbient::Unlit) });
    let frame = run(&device, &queue, &camera, &floor_and_wall(0.0), true, emission, &mut fx);
    let darkest = frame.hits.iter().zip(&frame.output).filter(|(h, _)| h.is_some()).map(|(_, o)| o[0]).fold(f32::INFINITY, f32::min);
    assert!(darkest >= emission - 2e-3, "darkened below the emission: {darkest}");
    assert!(darkest < DIRECT + ALBEDO * AMBIENT + emission - 0.05, "nothing occluded: {darkest}");
}
