//! The path tracer's denoisers, TLAS, probe grid and effect on a real GPU: the spatial filter
//! widens its step every iteration, the temporal filter keeps its history across same-size
//! resizes, the TLAS leaves follow the Morton sort, the probe grid hands out the SH it just wrote,
//! `PathTracerEffect` in a post chain writes the scene lit by its lights, the tracer's guide
//! buffers hold the surface and its motion, and `PathTracerDenoise` keeps the light while it
//! removes the noise. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Material, MaterialOptions, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pathtracer::{BVHBuilder, PathTracerDenoise, PathTracerEffect, PathTracerMaterial, ProbeGrid, SpatialDenoise, TLASBuilder, TemporalDenoise};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

const W: u32 = 64;
const H: u32 = 64;

/// The fast adapter on `WGPU_BACKEND`'s backend (the discrete GPU beside an integrated one, so the
/// scene goes to hardware ray queries where it has them), named on stderr.
fn adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    let info = adapter.get_info();
    eprintln!("{} ({:?})", info.name, info.backend);
    Some(adapter)
}

fn renderer() -> Option<Renderer> {
    let adapter = adapter()?;
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
        multiview_mask: None,
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
    let Some(adapter) = adapter() else { return };
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    let sky_only = traced_floor(&mut renderer, false, 8);
    let sunlit = traced_floor(&mut renderer, true, 8);
    // the dim default sky leaves a little; a white sun of 3 straight down on the default albedo
    // of 0.8 adds a Lambertian 0.8 * 3 / pi (as the raster lights it), in HDR (no tone curve)
    assert!(sky_only[1] > 0.0, "nothing written: {sky_only:?}");
    let sun = sunlit[1] - sky_only[1];
    let lambert = 0.8 * 3.0 / std::f32::consts::PI;
    assert!((sun - lambert).abs() < 0.1 * lambert, "the sun adds {sun}, not {lambert}: {sunlit:?} against {sky_only:?}");
}

/// Read a W×H rgba32uint texture back (the tracer's guide buffers).
fn read_uint_view(renderer: &Renderer, view: &wgpu::TextureView) -> Vec<[u32; 4]> {
    let device = renderer.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var src : texture_2d<u32>;
            @group(0) @binding(1) var<storage, read_write> dst : array<vec4u>;
            @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id : vec3u) {
                let size = textureDimensions(src);
                if (id.x >= size.x || id.y >= size.y) { return; }
                dst[id.y * size.x + id.x] = textureLoad(src, id.xy, 0);
            }
            "#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let size = (W * H * 16) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }, wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() }],
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

fn half(word: u32, high: bool) -> f32 {
    let h = if high { word >> 16 } else { word & 0xffff } as u16;
    // f16 -> f32 (normal numbers and zero suffice here)
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    if exp == 0 {
        return sign * mant * 2f32.powi(-24);
    }
    sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
}

fn snorm(word: u32, high: bool) -> f32 {
    let v = if high { (word >> 16) as u16 as i16 } else { word as u16 as i16 };
    (v as f32 / 32767.0).max(-1.0)
}

/// A 6 x 6 floor under a white sun straight down, seen from 3 m above, traced with guides.
struct Floor {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    tracer: kansei_core::pathtracer::PathTracer,
    bvh: BVHBuilder,
    data: kansei_core::pathtracer::GPUBVHData,
    tlas: TLASBuilder,
    lights: u32,
}

impl Floor {
    fn new() -> Option<Self> {
        let renderer = renderer()?;
        let mut scene = Scene::new();
        let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), box_material());
        floor.object.set_position(0.0, -0.1, 0.0);
        scene.add(SceneNode::Renderable(floor));
        scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 3.0))));
        let mut camera = Camera::new(40.0, 0.1, 50.0, 1.0);
        Self::look_from(&mut camera, 0.0);
        scene.prepare(&camera.position());
        let mut bvh = BVHBuilder::new();
        let mut tlas = TLASBuilder::new(&renderer);
        let data = bvh.build_full(&renderer, &scene, &mut tlas);
        let mut tracer = kansei_core::pathtracer::PathTracer::new(&renderer);
        tracer.resize(W, H);
        tracer.set_materials(&BVHBuilder::scene_materials(&scene));
        let lights = tracer.set_lights_from_scene(&scene);
        tracer.set_guides(true);
        Some(Self { renderer, scene, camera, tracer, bvh, data, tlas, lights })
    }

    /// Straight down from 3 m above (x, 0).
    fn look_from(camera: &mut Camera, x: f32) {
        camera.set_position(x, 3.0, 0.0);
        camera.look_at(&Vec3::new(x, 0.0, -0.001));
        camera.update_projection_matrix();
        camera.update_view_matrix();
    }

    fn trace(&mut self, denoise: Option<&mut PathTracerDenoise>) {
        let mut encoder = self.renderer.device().create_command_encoder(&Default::default());
        self.tracer.reset_accumulation();
        self.tracer.trace(&mut encoder, &self.data, self.tlas.tlas_nodes_buf.as_ref().unwrap(), &self.camera, self.lights);
        if let Some(d) = denoise {
            d.denoise(&mut encoder, &self.tracer, &self.camera);
        }
        self.renderer.queue().submit(Some(encoder.finish()));
    }

    fn surface(&self) -> Vec<[u32; 4]> {
        read_uint_view(&self.renderer, self.tracer.guides().unwrap().surface)
    }
}

/// The guides hold the floor's albedos (its base colour through Fresnel at the view), its
/// normal and depth, and the motion of the camera and of the floor itself.
#[test]
fn guides_hold_the_surface_and_its_motion() {
    let Some(mut f) = Floor::new() else { return };
    f.tracer.set_jitter(false);
    f.trace(None);
    let c = px(W / 2, H / 2);
    let albedo = read_uint_view(&f.renderer, f.tracer.guides().unwrap().albedo)[c];
    // straight on, Schlick's Fresnel is F0 = 0.04: 0.8 * 0.96 diffuse, 0.04 specular
    let (da, sa) = (half(albedo[0], false), half(albedo[1], true));
    assert!((da - 0.768).abs() < 0.005 && (sa - 0.04).abs() < 0.002, "albedos {da} {sa}");
    let s = f.surface()[c];
    let (z, ny) = (f32::from_bits(s[0]), snorm(s[1], true));
    assert!((z - 3.0).abs() < 0.01 && ny > 0.99, "depth {z}, normal y {ny}");
    let still = (half(s[2], false), half(s[2], true));
    assert!(still.0.abs() < 1e-5 && still.1.abs() < 1e-5, "a still camera's motion {still:?}");

    // the camera 0.3 m right: the floor moves left by 0.3 m of the 2 * 3 * tan(20°) the view spans
    let shift = 0.3 / (6.0 * 20f32.to_radians().tan());
    Floor::look_from(&mut f.camera, 0.3);
    f.trace(None);
    let m = half(f.surface()[c][2], false);
    assert!((m + shift).abs() < 0.002, "camera motion {m}, expected {}", -shift);

    // the camera still, the floor 0.3 m right: it moves right as much (last frame's transform)
    f.trace(None);
    let idx = f.scene.ordered_indices().next().unwrap();
    let r = f.scene.get_renderable_mut(idx).unwrap();
    r.object.set_position(0.3, -0.1, 0.0);
    r.object.update_model_matrix();
    r.object.update_world_matrix(None);
    f.bvh.refresh_transforms(&f.renderer, &f.scene, &mut f.data, &mut f.tlas);
    f.trace(None);
    let m = half(f.surface()[c][2], false);
    assert!((m - shift).abs() < 0.002, "object motion {m}, expected {shift}");
}

/// Over frames the denoiser keeps the floor's light and takes away most of its noise (the
/// specular lobe picked 1 time in 20).
#[test]
fn the_denoiser_keeps_the_light_and_removes_the_noise() {
    let Some(mut f) = Floor::new() else { return };
    let mut denoise = PathTracerDenoise::new(&f.renderer);
    for _ in 0..16 {
        f.trace(Some(&mut denoise));
    }
    let noisy = read_view(&f.renderer, f.tracer.output_view().unwrap());
    let denoised = read_view(&f.renderer, denoise.output_view().unwrap());
    let stats = |img: &[[f32; 4]]| {
        let v: Vec<f32> = (H / 4..H * 3 / 4).flat_map(|y| (W / 4..W * 3 / 4).map(move |x| (x, y))).map(|(x, y)| img[px(x, y)][1]).collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        (mean, sd)
    };
    let (nm, nsd) = stats(&noisy);
    let (dm, dsd) = stats(&denoised);
    assert!(denoised.iter().all(|p| p.iter().all(|c| c.is_finite())), "non-finite output");
    assert!((dm - nm).abs() < 0.05 * nm, "the light changed: {dm} against {nm}");
    assert!(dsd < 0.2 * nsd, "noise left: {dsd} of {nsd}");
}
