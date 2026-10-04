//! The path tracer's denoisers, TLAS, probe grid and effect on a real GPU: the spatial filter
//! widens its step every iteration, the temporal filter keeps its history across same-size
//! resizes, the TLAS leaves follow the Morton sort, the probe grid hands out the SH it just wrote,
//! and `PathTracerEffect` in a post chain writes the scene lit by its lights. Skipped (passes)
//! when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Material, MaterialOptions, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pathtracer::{BVHBuilder, PathTracerEffect, PathTracerMaterial, ProbeGrid, SpatialDenoise, TLASBuilder, TemporalDenoise};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

const W: u32 = 64;
const H: u32 = 64;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig::default());
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn texture(device: &wgpu::Device, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
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

/// An rgba32float texture holding `rgba` (W×H, row-major).
fn rgba_texture(renderer: &Renderer, rgba: &[[f32; 4]]) -> wgpu::Texture {
    let tex = texture(renderer.device(), wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    renderer.queue().write_texture(
        tex.as_image_copy(),
        bytemuck::cast_slice(rgba),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 16), rows_per_image: None },
        tex.size(),
    );
    tex
}

/// A flat GBuffer: depth 0.5 everywhere (not sky) and every normal facing +z.
fn flat_gbuffer(renderer: &Renderer) -> (wgpu::TextureView, wgpu::TextureView) {
    let depth = texture(renderer.device(), wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
    let depth_view = depth.create_view(&Default::default());
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.5), store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    renderer.queue().submit(Some(encoder.finish()));
    let normal = rgba_texture(renderer, &vec![[0.5, 0.5, 1.0, 1.0]; (W * H) as usize]);
    (depth_view, normal.create_view(&Default::default()))
}

/// Read a W×H float texture back through a compute copy (the denoisers' outputs are private).
fn read_view(renderer: &Renderer, view: &wgpu::TextureView) -> Vec<[f32; 4]> {
    let device = renderer.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var src : texture_2d<f32>;
            @group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
            @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id : vec3u) {
                let size = textureDimensions(src);
                if (id.x >= size.x || id.y >= size.y) { return; }
                dst[id.y * size.x + id.x] = textureLoad(src, id.xy, 0);
            }
            "#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let size = (W * H * 16) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(W / 8, H / 8, 1);
    }
    renderer.queue().submit(Some(encoder.finish()));
    renderer.read_back_buffer_sync(&buf, size)
}

fn px(x: u32, y: u32) -> usize {
    (y * W + x) as usize
}

#[test]
fn spatial_denoise_doubles_its_step_every_iteration() {
    let Some(renderer) = renderer() else { return };
    let (depth, normal) = flat_gbuffer(&renderer);
    // one lit pixel; a huge variance so luminance never stops the filter
    let mut gi = vec![[0.0f32; 4]; (W * H) as usize];
    gi[px(32, 32)] = [1.0, 1.0, 1.0, 1.0];
    let input = rgba_texture(&renderer, &gi).create_view(&Default::default());
    let moments = rgba_texture(&renderer, &vec![[0.0, 0.0, 1.0, 1e4]; (W * H) as usize]).create_view(&Default::default());

    let mut spatial = SpatialDenoise::new(&renderer);
    spatial.resize(W, H);
    // more iterations than the default, so the per-iteration parameters outgrow their first buffer
    spatial.iterations = 4;
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    let out = spatial.denoise(&mut encoder, &input, &depth, &normal, &moments);
    renderer.queue().submit(Some(encoder.finish()));
    let out = read_view(&renderer, out);

    // steps 1, 2, 4, 8 reach 2·(1+2+4+8) = 30 px and every offset in between; a single step
    // size for all four (8) would reach only multiples of 8, out to 64 px
    let at = |dx: u32| out[px(32 + dx, 32)][0];
    assert!(at(1) > 0.0, "the step-1 iteration never ran: {}", at(1));
    assert!(at(30) > 0.0, "the filter stops short of 30 px: {}", at(30));
    assert_eq!(at(31), 0.0, "the filter reaches past 30 px");
}

#[test]
fn temporal_denoise_keeps_its_history_when_resized_to_the_same_size() {
    let Some(renderer) = renderer() else { return };
    let (depth, normal) = flat_gbuffer(&renderer);
    // a checkerboard that flips every frame: the neighbourhood clamp keeps all of [0, 1], so
    // the history survives and blends in at 1 - blend
    let checker = |phase: u32| -> wgpu::TextureView {
        let rgba: Vec<[f32; 4]> = (0..W * H)
            .map(|i| {
                let v = ((i % W + i / W + phase) % 2) as f32;
                [v, v, v, 1.0]
            })
            .collect();
        rgba_texture(&renderer, &rgba).create_view(&Default::default())
    };
    let identity = glam::Mat4::IDENTITY.to_cols_array();

    let mut temporal = TemporalDenoise::new(&renderer);
    for frame in 0..2 {
        // PathTracerEffect resizes every frame
        temporal.resize(W, H);
        let gi = checker(frame);
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        temporal.denoise(&mut encoder, &gi, &depth, &normal, &identity, &identity, frame);
        renderer.queue().submit(Some(encoder.finish()));
    }
    let out = read_view(&renderer, temporal.output_view().unwrap());

    // (1, 0) was 1 in frame 0 and is 0 now: blended at the default 0.1, 0.9 of the history stays
    let v = out[px(1, 0)][0];
    assert!((v - 0.9).abs() < 0.01, "history lost: (1, 0) = {v}, expected 0.9");
    let v = out[px(0, 0)][0];
    assert!((v - 0.1).abs() < 0.01, "history lost: (0, 0) = {v}, expected 0.1");
}

#[test]
fn temporal_denoise_hands_out_the_moments_it_just_wrote() {
    let Some(renderer) = renderer() else { return };
    let (depth, normal) = flat_gbuffer(&renderer);
    let gi = rgba_texture(&renderer, &vec![[0.5, 0.5, 0.5, 1.0]; (W * H) as usize]).create_view(&Default::default());
    let identity = glam::Mat4::IDENTITY.to_cols_array();

    let mut temporal = TemporalDenoise::new(&renderer);
    temporal.resize(W, H);
    for frame in 0..3 {
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        temporal.denoise(&mut encoder, &gi, &depth, &normal, &identity, &identity, frame);
        renderer.queue().submit(Some(encoder.finish()));
        let moments = read_view(&renderer, temporal.moments_view().unwrap());
        // moments are (m1, m2, history length, variance): frame n leaves a history of n + 1
        let len = moments[px(10, 10)][2];
        assert_eq!(len, (frame + 1) as f32, "frame {frame}: moments_view shows history length {len}");
    }
}

fn box_material() -> Material {
    Material::new("Box", "", vec![], MaterialOptions::default())
}

/// Build a TLAS over `n` unit boxes on a line, their index order scrambled against their x order,
/// and check every leaf holds four x-neighbours and every instance appears once.
fn check_tlas_leaves(renderer: &Renderer, n: u32) {
    let rank = |i: u32| (i * 37) % n;
    let mut scene = Scene::new();
    for i in 0..n {
        let mut r = Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), box_material());
        r.object.set_position(rank(i) as f32 * 2.0, 0.0, 0.0);
        scene.add(SceneNode::Renderable(r));
    }
    scene.prepare(&Vec3::new(0.0, 0.0, 10.0));
    let mut tlas = TLASBuilder::new(renderer);
    let data = BVHBuilder::new().build_full(renderer, &scene, &mut tlas);
    assert_eq!(data.instance_count, n);

    let total = TLASBuilder::total_nodes(n);
    let nodes: Vec<[i32; 4]> = renderer.read_back_buffer_sync(tlas.tlas_nodes_buf.as_ref().unwrap(), total as u64 * 128);
    // levels are stored root first, so the leaves are the last nodes
    let leaves = n.div_ceil(4);
    let mut seen = vec![0u32; n as usize];
    for leaf in total - leaves..total {
        let children = nodes[(leaf * 8 + 6) as usize];
        let counts = nodes[(leaf * 8 + 7) as usize];
        let ranks: Vec<u32> = (0..4)
            .filter(|&j| counts[j] != 0)
            .map(|j| {
                let inst = (-children[j] - 1) as u32;
                seen[inst as usize] += 1;
                rank(inst)
            })
            .collect();
        let spread = ranks.iter().max().unwrap() - ranks.iter().min().unwrap();
        assert_eq!(spread, 3, "{n} instances: leaf {leaf} groups x ranks {ranks:?}, not 4 neighbours");
    }
    assert!(seen.iter().all(|&c| c == 1), "{n} instances: leaves don't hold every instance once: {seen:?}");
}

#[test]
fn tlas_leaves_group_instances_in_morton_order() {
    let Some(renderer) = renderer() else { return };
    // one radix-sort workgroup, then three (a bin count that is not a power of two)
    check_tlas_leaves(&renderer, 64);
    check_tlas_leaves(&renderer, 600);
}

#[test]
fn probe_grid_returns_the_sh_it_just_wrote() {
    let Some(renderer) = renderer() else { return };
    // probes inside one emissive box, so every ray hits radiance 2
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(Renderable::new(BoxGeometry::new(4.0, 4.0, 4.0), box_material())));
    scene.prepare(&Vec3::new(0.0, 0.0, 10.0));
    let mut tlas = TLASBuilder::new(&renderer);
    let data = BVHBuilder::new().build_full(&renderer, &scene, &mut tlas);

    let device = renderer.device();
    let storage = |bytes: &[u8]| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        renderer.queue().write_buffer(&buf, 0, bytes);
        buf
    };
    let materials = storage(bytemuck::bytes_of(&PathTracerMaterial::emissive([1.0, 1.0, 1.0], 2.0)));
    let lights = storage(&[0u8; 64]);

    let mut probes = ProbeGrid::new(&renderer);
    probes.configure([-1.0; 3], [1.0; 3], 2.0);
    let mut encoder = device.create_command_encoder(&Default::default());
    probes.update(&mut encoder, &data, tlas.tlas_nodes_buf.as_ref().unwrap(), &materials, &lights, 0, 0);
    renderer.queue().submit(Some(encoder.finish()));

    let count = probes.probe_count;
    assert_eq!(count, 8);
    let sh: Vec<[f32; 4]> = renderer.read_back_buffer_sync(probes.sh_buffer().unwrap(), count as u64 * 9 * 16);
    // constant radiance L projects to L0 = 4π · Y00 · L = 2 · 2·√π ≈ 7.09 (plus the albedo-weighted indirect, 0 on frame 0)
    for p in 0..count as usize {
        let l0 = sh[p * 9][0];
        assert!((l0 - 4.0 * std::f32::consts::PI.sqrt()).abs() < 0.1, "probe {p}: L0 = {l0}, expected ≈ 7.09");
    }
}

/// The mean of the image's middle quarter after `frames` frames of a floor seen from above,
/// through `PathTracerEffect` as a post-processing effect, with the scene's sun or without lights.
fn traced_floor(renderer: &mut Renderer, sun: bool, frames: u32) -> [f32; 3] {
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), Material::standard_lit("Floor", &StandardLitOptions::default()));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 3.0))));
    let mut camera = Camera::new(40.0, 0.1, 50.0, 1.0);
    camera.set_position(0.0, 3.0, 0.01);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);

    let mut effect = PathTracerEffect::new(renderer, &scene);
    if sun {
        effect.set_lights_from_scene(&scene);
    }
    effect.initialize(renderer.device(), &gbuffer, &camera);
    let output = texture(renderer.device(), wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING);
    let output_view = output.create_view(&Default::default());
    for _ in 0..frames {
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
    }
    let image = read_view(renderer, &output_view);
    let mut sum = [0.0f32; 3];
    for y in H / 4..H * 3 / 4 {
        for x in W / 4..W * 3 / 4 {
            let p = image[px(x, y)];
            assert!(p.iter().all(|c| c.is_finite()), "pixel ({x}, {y}) is {p:?}");
            (0..3).for_each(|c| sum[c] += p[c]);
        }
    }
    sum.map(|c| c / (W * H / 4) as f32)
}

#[test]
fn the_effect_writes_the_floor_lit_by_the_scenes_lights() {
    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else { return };
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    let sky_only = traced_floor(&mut renderer, false, 8);
    let sunlit = traced_floor(&mut renderer, true, 8);
    // the dim default sky leaves a little; a white sun of 3 on the default albedo of 0.8 lifts
    // the floor far above that, in HDR (no tone curve), and never past albedo times the sun
    assert!(sky_only[1] > 0.0, "nothing written: {sky_only:?}");
    assert!(sunlit[1] > sky_only[1] + 0.3, "the sun adds too little: {sunlit:?} against {sky_only:?}");
    assert!(sunlit[1] > 1.0 && sunlit[1] < 0.8 * 3.0 + 0.2, "not the HDR radiance: {sunlit:?}");
}
