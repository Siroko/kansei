//! The path tracer's hardware traversal against its BVH4s, on the path tracer's test scene (the
//! Cornell box of `pathtracer_test` and the web demo: walls, two boxes turned and scaled, a glass
//! Stanford dragon, an emissive panel): ray for ray (closest hits, any hits, the hit's triangle,
//! instance, normal and area), then the traced image at the same sample counts. Skipped (passes)
//! where the adapter has no ray queries.

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, SphereGeometry};
use kansei_core::lights::{AreaLight, DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::Material;
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pathtracer::{BVHBuilder, GPUBVHData, PathTracer, PathTracerMaterial, TLASBuilder};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::RtBackend;

/// A headless renderer whose device has hardware ray queries (the fast adapter, `WGPU_BACKEND`'s
/// backend: DX12 too, where `Auto` would trace in software).
fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    let mut renderer = Renderer::new(RendererConfig::default());
    pollster::block_on(renderer.initialize_headless(&adapter));
    let info = renderer.device().adapter_info();
    if !renderer.device().features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        eprintln!("{} ({:?}) has no ray queries: skipped", info.name, info.backend);
        return None;
    }
    eprintln!("{} ({:?})", info.name, info.backend);
    Some(renderer)
}

fn material() -> Material {
    Material::basic_lit("Surface", [0.8, 0.8, 0.8, 1.0], [0.1, 0.1, 0.1, 0.5])
}

/// `pathtracer_test`'s Cornell box: 8 x 6 x 8, a red and a green wall, a tall box and a cube
/// turned and scaled, the glass dragon, an emissive panel under the ceiling, a sun and an area
/// light; and the tracer's materials, one a renderable.
fn cornell() -> (Scene, Vec<PathTracerMaterial>, Vec<Light>) {
    let mut scene = Scene::new();
    let mut materials = Vec::new();
    let white = PathTracerMaterial { albedo: [0.73, 0.73, 0.73], ..Default::default() };
    let mut add = |scene: &mut Scene, mut r: Renderable, m: PathTracerMaterial, at: [f32; 3], rotation: [f32; 3], scale: [f32; 3]| {
        r.object.position = Vec3::new(at[0], at[1], at[2]);
        r.object.rotation = Vec3::new(rotation[0], rotation[1], rotation[2]);
        r.object.scale = Vec3::new(scale[0], scale[1], scale[2]);
        r.object.update_model_matrix();
        r.object.update_world_matrix(None);
        scene.add(SceneNode::Renderable(r));
        materials.push(m);
    };
    let one = [1.0; 3];
    let flat = [0.0; 3];
    add(&mut scene, Renderable::new(BoxGeometry::new(8.0, 0.1, 8.0), material()), white, [0.0, -0.05, 0.0], flat, one);
    add(&mut scene, Renderable::new(BoxGeometry::new(8.0, 0.1, 8.0), material()), white, [0.0, 6.05, 0.0], flat, one);
    add(&mut scene, Renderable::new(BoxGeometry::new(8.0, 6.0, 0.1), material()), white, [0.0, 3.0, -4.05], flat, one);
    add(&mut scene, Renderable::new(BoxGeometry::new(0.1, 6.0, 8.0), material()), PathTracerMaterial { albedo: [0.65, 0.05, 0.05], ..Default::default() }, [-4.05, 3.0, 0.0], flat, one);
    add(&mut scene, Renderable::new(BoxGeometry::new(0.1, 6.0, 8.0), material()), PathTracerMaterial { albedo: [0.12, 0.45, 0.15], ..Default::default() }, [4.05, 3.0, 0.0], flat, one);
    // a tall box turned about y, a cube turned about all three axes and stretched
    add(&mut scene, Renderable::new(BoxGeometry::new(1.0, 2.0, 1.0), material()), PathTracerMaterial { albedo: [0.9, 0.2, 0.2], ..Default::default() }, [-1.5, 1.0, -1.0], [0.0, 0.4, 0.0], one);
    add(&mut scene, Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), material()), PathTracerMaterial::metal([0.9, 0.8, 0.6], 0.2), [1.6, 1.4, -0.5], [0.5, 0.9, 0.3], [1.4, 0.8, 1.1]);
    // a ball, and a panel under the ceiling that glows
    add(&mut scene, Renderable::new(SphereGeometry::new(0.6, 24, 16), material()), PathTracerMaterial { albedo: [0.2, 0.3, 0.9], roughness: 0.4, ..Default::default() }, [2.2, 0.6, 1.8], flat, one);
    add(&mut scene, Renderable::new(BoxGeometry::new(2.0, 0.05, 1.0), material()), PathTracerMaterial::emissive([1.0, 0.9, 0.7], 6.0), [0.0, 5.9, 0.0], flat, one);
    // the dragon in glass, as pathtracer_test places it
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/assets/geom/stanford_dragon_pbr.glb");
    let dragon = GLTFLoader::load(path).expect("the Stanford dragon");
    for part in dragon.renderables {
        let s = 0.03;
        let p = part.position;
        add(
            &mut scene,
            Renderable::new(part.geometry, material()),
            PathTracerMaterial::glass(1.5),
            [p.x, p.y, p.z + 2.0],
            [part.rotation.x, part.rotation.y, part.rotation.z],
            [part.scale.x * s, part.scale.y * s, part.scale.z * s],
        );
    }
    let lights = vec![
        Light::Directional(DirectionalLight::new(Vec3::new(-0.5, -1.0, -0.3).normalize(), Vec3::new(1.0, 0.95, 0.9), 3.0)),
        Light::Area(AreaLight::new(Vec3::new(0.0, 5.8, 1.5), Vec3::new(0.0, 0.0, 1.5), Vec3::new(1.0, 1.0, 1.0), 8.0, 1.0, 1.0)),
    ];
    scene.prepare(&Vec3::new(0.0, 3.0, 8.0));
    (scene, materials, lights)
}

/// The scene ready for `backend`, with the BVH4 TLAS (the passes take it either way).
fn build(renderer: &Renderer, scene: &Scene, backend: RtBackend) -> (GPUBVHData, TLASBuilder, BVHBuilder) {
    let mut bvh = BVHBuilder::new();
    bvh.set_backend(backend);
    let mut tlas = TLASBuilder::new(renderer);
    let data = bvh.build_full(renderer, scene, &mut tlas);
    assert_eq!(data.backend(), backend, "asked for {backend:?}");
    (data, tlas, bvh)
}

/// What each ray of `queries` (origin + any-hit distance, direction) finds through `data`: per
/// ray four vec4s, (closest t or -1, triangle, instance, any hit within the distance), (normal,
/// material), (position, u), (v, the triangle's world area, 0, 0).
fn trace(renderer: &Renderer, data: &GPUBVHData, tlas: &TLASBuilder, queries: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let device = renderer.device();
    let code = format!(
        "{}\n\
         @group(0) @binding(4) var<storage, read> queries : array<vec4f>;\n\
         @group(0) @binding(5) var<storage, read_write> results : array<vec4f>;\n\
         @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id : vec3u) {{\n\
             let i = id.x;\n\
             if (i >= arrayLength(&queries) / 2u) {{ return; }}\n\
             var ray : Ray;\n\
             ray.origin = queries[2u * i].xyz;\n\
             ray.dir = queries[2u * i + 1u].xyz;\n\
             let h = trace_bvh(ray);\n\
             let shadow = trace_bvh_shadow(ray, queries[2u * i].w);\n\
             var area = 0.0;\n\
             if (h.hit) {{ area = get_world_tri_area(h.tri_index, h.instance_id); }}\n\
             results[4u * i] = vec4f(select(-1.0, h.t, h.hit), bitcast<f32>(h.tri_index), bitcast<f32>(h.instance_id), select(0.0, 1.0, shadow));\n\
             results[4u * i + 1u] = vec4f(h.world_norm, bitcast<f32>(h.mat_index));\n\
             results[4u * i + 2u] = vec4f(h.world_pos, h.u);\n\
             results[4u * i + 3u] = vec4f(h.v, area, 0.0, 0.0);\n\
         }}",
        data.traversal_wgsl(0)
    );
    let module = data.shader_module(device, "PtParity", &code);
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let mut entries = data.layout_entries(0);
    entries.extend([storage(4, true), storage(5, false)]);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("PtParity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (queries.len() * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    renderer.queue().write_buffer(&input, 0, bytemuck::cast_slice(queries));
    let rays = queries.len() / 2;
    let size = (rays * 64) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let mut bind = data.bind_group_entries(tlas.tlas_nodes_buf.as_ref().unwrap(), 0);
    bind.push(wgpu::BindGroupEntry { binding: 4, resource: input.as_entire_binding() });
    bind.push(wgpu::BindGroupEntry { binding: 5, resource: output.as_entire_binding() });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &bind });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((rays as u32).div_ceil(64), 1, 1);
    }
    renderer.queue().submit(Some(encoder.finish()));
    renderer.read_back_buffer_sync(&output, size)
}

/// A small deterministic generator in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Rays through the box: a camera's primary rays from the open side, and rays from random points
/// inside in random directions (bounces, shadow rays), each with a random any-hit distance.
fn queries() -> Vec<[f32; 4]> {
    let mut rng = Lcg(7);
    let mut q = Vec::new();
    let eye = glam::Vec3::new(0.0, 3.0, 8.0);
    let (w, h) = (160, 120);
    for y in 0..h {
        for x in 0..w {
            let dir = glam::Vec3::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 0.75 - (y as f32 + 0.5) / h as f32 * 1.5, -1.6).normalize();
            q.push([eye.x, eye.y, eye.z, 2.0 + rng.next() * 10.0]);
            q.push([dir.x, dir.y, dir.z, 0.0]);
        }
    }
    for _ in 0..20000 {
        let o = glam::Vec3::new(-3.9 + 7.8 * rng.next(), 0.05 + 5.9 * rng.next(), -3.9 + 7.8 * rng.next());
        let z = 2.0 * rng.next() - 1.0;
        let a = std::f32::consts::TAU * rng.next();
        let r = (1.0 - z * z).sqrt();
        q.push([o.x, o.y, o.z, 0.2 + rng.next() * 6.0]);
        q.push([r * a.cos(), z, r * a.sin(), 0.0]);
    }
    q
}

/// Compare what both backends found for `q`'s rays: report the differences among the robust
/// rays, and fail if more than one in 2000 differ.
fn compare(q: &[[f32; 4]], sw: &[[f32; 4]], hw: &[[f32; 4]]) {
    let rays = q.len() / 2;
    // a hit within a thousandth of a triangle's edge, or an any-hit distance within a thousandth
    // of the closest hit, may go either way: not robust; nor which of two coplanar faces at the
    // same distance a ray hits (the tall box stands on the floor): a tie
    let edge = |r: &[[f32; 4]], i: usize| {
        let (u, v) = (r[4 * i + 2][3], r[4 * i + 3][0]);
        r[4 * i][0] >= 0.0 && u.min(v).min(1.0 - u - v) < 1e-3
    };
    let (mut robust, mut hits, mut ties, mut differ) = (0, 0, 0, Vec::new());
    for i in 0..rays {
        if edge(sw, i) || edge(hw, i) {
            continue;
        }
        robust += 1;
        let (s, h) = (sw[4 * i], hw[4 * i]);
        let max_dist = q[2 * i][3];
        let mut why = Vec::new();
        if (s[0] < 0.0) != (h[0] < 0.0) {
            why.push("hit/miss");
        } else if s[0] >= 0.0 {
            hits += 1;
            let t = s[0];
            if (t - h[0]).abs() > 1e-4 * t.max(1.0) {
                why.push("distance");
            }
            let same = s[1].to_bits() == h[1].to_bits() && s[2].to_bits() == h[2].to_bits() && sw[4 * i + 1][3].to_bits() == hw[4 * i + 1][3].to_bits();
            if !same && (t - h[0]).abs() <= 1e-5 * t.max(1.0) {
                ties += 1;
            } else if !same {
                why.push("triangle/instance/material");
            } else {
                let (ns, nh) = (glam::Vec4::from(sw[4 * i + 1]).truncate(), glam::Vec4::from(hw[4 * i + 1]).truncate());
                if ns.dot(nh) < 0.9999 {
                    why.push("normal");
                }
                let (ps, ph) = (glam::Vec4::from(sw[4 * i + 2]).truncate(), glam::Vec4::from(hw[4 * i + 2]).truncate());
                if ps.distance(ph) > 1e-4 * t.max(1.0) {
                    why.push("position");
                }
                let (a, b) = (sw[4 * i + 3][1], hw[4 * i + 3][1]);
                if (a - b).abs() > 1e-5 * a {
                    why.push("area");
                }
            }
        }
        let near_max = s[0] >= 0.0 && (s[0] - max_dist).abs() < 1e-3 * max_dist;
        if s[3] != h[3] && !near_max {
            why.push("any hit");
        }
        if !why.is_empty() {
            differ.push((i, why, s, h));
        }
    }
    for (i, why, s, h) in differ.iter().take(10) {
        eprintln!("ray {i} {:?} -> {:?}: {why:?}\n  software {s:?}\n  hardware {h:?}", q[2 * i], q[2 * i + 1]);
    }
    eprintln!("{robust} robust rays of {rays} ({hits} hits, {ties} ties between coplanar faces): {} differ", differ.len());
    assert!(hits > robust / 2, "too few hits to compare: {hits} of {robust}");
    assert!(differ.len() * 2000 <= robust, "{} of {robust} robust rays differ", differ.len());
}

#[test]
fn hardware_hits_match_the_bvh() {
    let Some(renderer) = renderer() else { return };
    let (scene, _, _) = cornell();
    let q = queries();
    let (sw_data, sw_tlas, _) = build(&renderer, &scene, RtBackend::Software);
    let (hw_data, hw_tlas, _) = build(&renderer, &scene, RtBackend::Hardware);
    eprintln!("{} triangles, {} instances, {} rays", sw_data.triangle_count, sw_data.instance_count, q.len() / 2);
    compare(&q, &trace(&renderer, &sw_data, &sw_tlas, &q), &trace(&renderer, &hw_data, &hw_tlas, &q));
}

#[test]
fn moved_instances_match_after_a_refresh() {
    let Some(renderer) = renderer() else { return };
    let (mut scene, _, _) = cornell();
    let q = queries();
    let (mut sw_data, mut sw_tlas, mut sw_bvh) = build(&renderer, &scene, RtBackend::Software);
    let (mut hw_data, mut hw_tlas, mut hw_bvh) = build(&renderer, &scene, RtBackend::Hardware);
    // the cube flies up and turns, the ball rolls across, as the web demo's box does
    let mut moved = 0;
    for (i, at, rotation) in [(6, [0.5, 3.0, 1.0], [1.2, 0.3, 0.8]), (7, [-2.5, 0.6, 2.5], [0.0, 2.0, 0.0])] {
        let r = scene.get_renderable_mut(i).expect("a renderable");
        r.object.position = Vec3::new(at[0], at[1], at[2]);
        r.object.rotation = Vec3::new(rotation[0], rotation[1], rotation[2]);
        r.object.update_model_matrix();
        r.object.update_world_matrix(None);
        moved += 1;
    }
    assert_eq!(moved, 2);
    sw_bvh.refresh_transforms(&renderer, &scene, &mut sw_data, &mut sw_tlas);
    hw_bvh.refresh_transforms(&renderer, &scene, &mut hw_data, &mut hw_tlas);
    let sw = trace(&renderer, &sw_data, &sw_tlas, &q);
    compare(&q, &sw, &trace(&renderer, &hw_data, &hw_tlas, &q));
    // and the move is seen: the cube's instance is hit where it went
    let ray = [[0.0, 3.0, 8.0, 100.0], [0.5 / 7.0, 0.0, -1.0, 0.0]];
    let hit = trace(&renderer, &hw_data, &hw_tlas, &ray);
    assert_eq!(hit[0][2].to_bits(), 6, "the moved cube isn't in front of the eye: {:?}", hit[0]);
}

/// The scene traced through `data` at `width` x `height` for `frames` frames of `spp` samples,
/// accumulated, read back.
#[allow(clippy::too_many_arguments)]
fn image(renderer: &Renderer, data: &GPUBVHData, tlas: &TLASBuilder, materials: &[PathTracerMaterial], lights: &[Light], camera: &Camera, (width, height): (u32, u32), frames: u32, blue_noise: bool) -> Vec<[f32; 4]> {
    let mut tracer = PathTracer::new(renderer);
    tracer.resize(width, height);
    tracer.set_spp(4);
    tracer.set_max_bounces(6);
    tracer.set_use_blue_noise(blue_noise);
    tracer.set_materials(materials);
    let light_count = tracer.set_lights(lights);
    for _ in 0..frames {
        tracer.trace_frame(renderer, data, tlas, camera, light_count);
    }
    read_view(renderer, tracer.output_view().unwrap(), width, height)
}

/// Read a float texture back through a compute copy.
fn read_view(renderer: &Renderer, view: &wgpu::TextureView, width: u32, height: u32) -> Vec<[f32; 4]> {
    let device = renderer.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var src : texture_2d<f32>;
             @group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
             @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id : vec3u) {
                 let size = textureDimensions(src);
                 if (id.x >= size.x || id.y >= size.y) { return; }
                 dst[id.y * size.x + id.x] = textureLoad(src, id.xy, 0);
             }"
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let size = (width * height * 16) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }, wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    renderer.queue().submit(Some(encoder.finish()));
    renderer.read_back_buffer_sync(&buf, size)
}

fn luminance(p: [f32; 4]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

#[test]
fn hardware_image_matches_the_bvh() {
    let Some(renderer) = renderer() else { return };
    let (scene, materials, lights) = cornell();
    let size = (160, 120);
    let mut camera = Camera::new(45.0, 0.1, 100.0, size.0 as f32 / size.1 as f32);
    camera.set_position(0.0, 3.0, 8.0);
    camera.look_at(&Vec3::new(0.0, 2.5, 0.0));
    camera.update_projection_matrix();
    camera.update_view_matrix();
    let (sw_data, sw_tlas, _) = build(&renderer, &scene, RtBackend::Software);
    let (hw_data, hw_tlas, _) = build(&renderer, &scene, RtBackend::Hardware);
    let frames = 16;
    let sw = image(&renderer, &sw_data, &sw_tlas, &materials, &lights, &camera, size, frames, true);
    let hw = image(&renderer, &hw_data, &hw_tlas, &materials, &lights, &camera, size, frames, true);
    // the same count of other samples (PCG in place of the blue noise): the noise's size
    let other = image(&renderer, &sw_data, &sw_tlas, &materials, &lights, &camera, size, frames, false);

    let n = sw.len() as f32;
    let mean = |img: &[[f32; 4]]| img.iter().map(|&p| luminance(p)).sum::<f32>() / n;
    let mean_diff = |a: &[[f32; 4]], b: &[[f32; 4]]| a.iter().zip(b).map(|(&p, &q)| (luminance(p) - luminance(q)).abs()).sum::<f32>() / n;
    assert!(sw.iter().chain(&hw).all(|p| p.iter().all(|c| c.is_finite())), "a pixel is not finite");
    let (m_sw, m_hw, m_other) = (mean(&sw), mean(&hw), mean(&other));
    let (d_hw, d_noise) = (mean_diff(&sw, &hw), mean_diff(&sw, &other));
    eprintln!("{} spp: mean luminance software {m_sw:.4}, hardware {m_hw:.4}, other samples {m_other:.4}; per pixel |software - hardware| {d_hw:.2e}, |software - other samples| {d_noise:.2e}", frames * 4);
    assert!(m_sw > 0.05, "the image is dark: {m_sw}");
    assert!((m_hw - m_sw).abs() < 0.01 * m_sw, "mean luminance: software {m_sw}, hardware {m_hw}");
    // the same samples down the same paths: well within the noise between two sample sets
    assert!(d_hw < 0.25 * d_noise, "hardware differs by {d_hw} a pixel, other samples by {d_noise}");
}

/// Rays the hardware's query must not be given (its module skips wgpu's check of them): a NaN or
/// an infinity in the origin or direction, or no interval to search. Each misses on both backends.
#[test]
fn rays_with_a_nan_or_an_infinity_miss() {
    let Some(renderer) = renderer() else { return };
    let (scene, _, _) = cornell();
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    // (origin + any-hit distance, direction), each aimed at the back wall from the eye but for its fault
    let q: Vec<[f32; 4]> = [
        ([nan, 3.0, 8.0, 100.0], [0.0, 0.0, -1.0]),
        ([0.0, 3.0, 8.0, 100.0], [0.0, nan, -1.0]),
        ([inf, 3.0, 8.0, 100.0], [0.0, 0.0, -1.0]),
        ([0.0, 3.0, 8.0, 100.0], [0.0, 0.0, -inf]),
        ([0.0, 3.0, 8.0, nan], [0.0, 0.0, -1.0]),
        ([0.0, 3.0, 8.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 3.0, 8.0, -5.0], [0.0, 0.0, -1.0]),
    ]
    .iter()
    .flat_map(|&(o, d)| [o, [d[0], d[1], d[2], 0.0]])
    .collect();
    for backend in [RtBackend::Software, RtBackend::Hardware] {
        let (data, tlas, _) = build(&renderer, &scene, backend);
        let r = trace(&renderer, &data, &tlas, &q);
        for i in 0..q.len() / 2 {
            assert_eq!(r[4 * i][3], 0.0, "{backend:?}: ray {i} {:?} -> {:?} found an any hit", q[2 * i], q[2 * i + 1]);
            // the closest hit has no distance limit: only the rays with a faulty origin or
            // direction (the first four) miss it
            if i < 4 {
                assert!(r[4 * i][0] < 0.0, "{backend:?}: ray {i} {:?} -> {:?} hit at {}", q[2 * i], q[2 * i + 1], r[4 * i][0]);
            }
        }
    }
}
