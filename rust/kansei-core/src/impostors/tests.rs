use super::*;
use super::tests_support::read_texels;

fn validate(name: &str, code: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    module
}

fn struct_size(module: &naga::Module, name: &str) -> usize {
    module
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no struct {name}"))
}

#[test]
fn shaders_validate_and_the_uniforms_match() {
    let impostor = validate("impostor", IMPOSTOR_WGSL);
    assert_eq!(struct_size(&impostor, "KanseiImpostor"), std::mem::size_of::<ImpostorParams>());
    let pack = validate("impostor_pack", include_str!("../shaders/impostor_pack.wgsl"));
    assert_eq!(struct_size(&pack, "Pack"), std::mem::size_of::<bake::PackGpu>());
    validate("impostor_mip", include_str!("../shaders/impostor_mip.wgsl"));
}

fn directions() -> Vec<glam::Vec3> {
    let mut dirs = vec![glam::Vec3::X, glam::Vec3::NEG_X, glam::Vec3::Y, glam::Vec3::NEG_Y, glam::Vec3::Z, glam::Vec3::NEG_Z];
    for i in 0..200 {
        // a spiral over the sphere
        let y = 1.0 - (i as f32 + 0.5) / 100.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let a = i as f32 * 2.399_963;
        dirs.push(glam::Vec3::new(r * a.cos(), y, r * a.sin()));
    }
    dirs
}

#[test]
fn octahedral_mapping_round_trips() {
    for d in directions() {
        let g = encode(d, false);
        assert!(g.abs().max_element() <= 1.0 + 1e-6, "{d} -> {g}");
        assert!(decode(g, false).distance(d) < 1e-5, "{d} -> {g} -> {}", decode(g, false));
        let h = encode(d, true);
        assert!(h.abs().max_element() <= 1.0 + 1e-6, "hemi {d} -> {h}");
        if d.y >= 0.0 {
            assert!(decode(h, true).distance(d) < 1e-5, "hemi {d} -> {h} -> {}", decode(h, true));
        } else if d.x.abs() + d.z.abs() > 1e-3 {
            // below the horizon: onto it (straight down has no nearest point there)
            let flat = glam::Vec3::new(d.x, 0.0, d.z).normalize();
            assert!(decode(h, true).distance(flat) < 1e-4, "hemi {d} -> {}", decode(h, true));
        }
    }
}

#[test]
fn frames_cover_the_sphere_and_their_axes_are_the_bake_cameras() {
    for layout in [ImpostorLayout::Octahedral, ImpostorLayout::HemiOctahedral] {
        let n = 8;
        let frames: Vec<_> = (0..n * n).map(|k| frame_direction(n, layout, k % n, k / n)).collect();
        // every direction (above the horizon, hemi) has a frame within a cell's reach
        for d in directions().into_iter().filter(|d| layout == ImpostorLayout::Octahedral || d.y >= 0.05) {
            let nearest = frames.iter().map(|f| f.dot(d)).fold(-1.0f32, f32::max);
            assert!(nearest > 0.93, "{layout:?}: {d} is {:.1} deg from a frame", nearest.acos().to_degrees());
        }
        for &d in &frames {
            // the bake's view (glam's look_at_rh) has kansei_impostor_basis' axes
            let view = glam::Mat4::look_at_rh(d * 3.0, glam::Vec3::ZERO, up_reference(d));
            let right = up_reference(d).cross(d).normalize();
            let up = d.cross(right);
            assert!(view.transform_vector3(right).distance(glam::Vec3::X) < 1e-5);
            assert!(view.transform_vector3(up).distance(glam::Vec3::Y) < 1e-5);
            assert!(view.transform_vector3(d).distance(glam::Vec3::Z) < 1e-5);
        }
    }
}

#[test]
fn the_billboard_faces_plus_z() {
    let g = billboard_geometry("Billboard");
    let p = |i: u32| glam::Vec3::from_slice(&g.vertices[i as usize].position[..3]);
    for t in g.indices.chunks(3) {
        assert!((p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).z > 0.0, "counter-clockwise from +z");
    }
}

// On a real GPU: a unit sphere whose material writes its normal (as normal and albedo).

const SPHERE_WGSL: &str = r#"
struct Tint { color: vec4<f32> };
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(position.xyz, 1.0);
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
@fragment fn fragment_main(in: VOut) -> FOut {
    let n = normalize(in.normal) * 0.5 + 0.5;
    return FOut(tint.color, vec4<f32>(0.0), vec4<f32>(n, 1.0), vec4<f32>(n, 1.0));
}
"#;

fn headless() -> Option<crate::renderers::Renderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let mut renderer = crate::renderers::Renderer::new(crate::renderers::RendererConfig { width: 64, height: 64, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn baked_sphere(renderer: &mut crate::renderers::Renderer, frames: u32, frame_size: u32) -> Impostor {
    let mut scene = crate::objects::Scene::new();
    let mut material = crate::materials::Material::new(
        "Sphere",
        SPHERE_WGSL,
        vec![crate::materials::Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
        crate::materials::MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    material.set_uniform_bindable(0, "Tint", &[[0.2f32, 0.4, 0.6, 1.0]]);
    let sphere = scene.add(crate::objects::SceneNode::Renderable(crate::objects::Renderable::new(crate::geometries::SphereGeometry::new(1.0, 64, 32), material)));
    renderer.bake_impostor(&mut scene, &[sphere], &ImpostorOptions { frames, frame_size, supersample: 2, ..Default::default() })
}

/// Every frame of a sphere's impostor sees a disc: covered inside, empty in the corners, the
/// frame's own direction as the normal (and the albedo, here) at its centre, and depth 0 there
/// (the sphere touches each frame's near plane).
#[test]
fn a_baked_sphere_is_a_disc_facing_each_frame() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (n, size) = (6u32, 32u32);
    let impostor = baked_sphere(&mut renderer, n, size);
    assert!((impostor.radius - 1.0).abs() < 1e-3 && impostor.center.length() < 1e-3, "bounds of the unit sphere");
    let (device, queue) = (renderer.device(), renderer.queue());
    let albedo = read_texels(device, queue, impostor.albedo_atlas());
    let normal_depth = read_texels(device, queue, impostor.normal_depth_atlas());
    let side = n * size;
    let at = |t: &[[u8; 4]], x: u32, y: u32| t[(y * side + x) as usize];
    let unpack = |t: [u8; 4]| glam::Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32) / 127.5 - 1.0;
    for k in 0..n * n {
        let (column, row) = (k % n, k / n);
        let d = frame_direction(n, ImpostorLayout::Octahedral, column, row);
        let (x0, y0) = (column * size, row * size);
        let covered = (0..size * size).filter(|i| at(&albedo, x0 + i % size, y0 + i / size)[3] > 127).count() as f32 / (size * size) as f32;
        assert!((covered - std::f32::consts::FRAC_PI_4).abs() < 0.05, "frame {k}: {covered} of it covered");
        assert_eq!(at(&albedo, x0, y0)[3], 0, "frame {k}: its corner is empty");
        let centre = at(&normal_depth, x0 + size / 2, y0 + size / 2);
        assert!(unpack(centre).normalize().dot(d) > 0.98, "frame {k}: normal {} for direction {d}", unpack(centre));
        assert!(centre[3] < 16, "frame {k}: depth {} at the centre", centre[3]);
        assert!(unpack(at(&albedo, x0 + size / 2, y0 + size / 2)).normalize().dot(d) > 0.98, "frame {k}: the albedo");
    }
}

/// Sampling the sphere's impostor along rays from anywhere round it (kansei_impostor_sample_lod,
/// in a compute shader): through the billboard's middle the ray hits the sphere where the sample
/// says, with its normal; through the billboard's corner it misses.
#[test]
fn rays_through_the_impostor_find_the_sphere() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let impostor = baked_sphere(&mut renderer, 12, 64);
    let (device, queue) = (renderer.device(), renderer.queue());
    // eyes 6 radii out, all round; for each, a point near the billboard's middle and its corner
    let eyes: Vec<glam::Vec3> = directions().into_iter().step_by(7).map(|d| d * 6.0).collect();
    let shader = format!(
        "{IMPOSTOR_WGSL}
        @group(0) @binding(0) var<uniform> imp: KanseiImpostor;
        @group(0) @binding(1) var albedo: texture_2d<f32>;
        @group(0) @binding(2) var normalDepth: texture_2d<f32>;
        @group(0) @binding(3) var samp: sampler;
        @group(0) @binding(4) var<storage, read> eyes: array<vec4f>;
        @group(0) @binding(5) var<storage, read_write> out: array<vec4f>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {{
            let eye = eyes[id.x].xyz;
            for (var c = 0u; c < 2u; c++) {{
                let corner = select(vec2f(0.95, 0.9), vec2f(0.3, -0.2), c == 0u);
                let p = kansei_impostor_corner(imp, corner, eye);
                let s = kansei_impostor_sample_lod(imp, albedo, normalDepth, samp, p, eye, 0.0);
                out[id.x * 6u + c * 3u] = vec4f(p, 0.0);
                out[id.x * 6u + c * 3u + 1u] = vec4f(s.position, s.alpha);
                out[id.x * 6u + c * 3u + 2u] = vec4f(s.normal, 0.0);
            }}
        }}"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    use wgpu::util::DeviceExt;
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&impostor.params()), usage: wgpu::BufferUsages::UNIFORM });
    let eye_data: Vec<[f32; 4]> = eyes.iter().map(|e| [e.x, e.y, e.z, 0.0]).collect();
    let eye_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&eye_data), usage: wgpu::BufferUsages::STORAGE });
    let out_size = (eyes.len() * 6 * 16) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: out_size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor { mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Linear, ..Default::default() });
    let (albedo, normal_depth) = (impostor.albedo_atlas().create_view(&Default::default()), impostor.normal_depth_atlas().create_view(&Default::default()));
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&albedo) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal_depth) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: eye_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(eyes.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, out_size);
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let values: Vec<[f32; 4]> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    let v3 = |v: [f32; 4]| glam::Vec3::new(v[0], v[1], v[2]);
    for (i, eye) in eyes.iter().enumerate() {
        let [p, hit, normal, far_p, far_hit, _] = std::array::from_fn(|k| values[i * 6 + k]);
        // where the ray from the eye through the billboard point meets the unit sphere
        let ray = (v3(p) - *eye).normalize();
        let b = eye.dot(ray);
        let t = -b - (b * b - (eye.length_squared() - 1.0)).sqrt();
        let expected = *eye + ray * t;
        assert!(hit[3] > 0.9, "eye {eye}: coverage {}", hit[3]);
        assert!(v3(hit).distance(expected) < 0.05, "eye {eye}: surface at {} for {expected}", v3(hit));
        assert!(v3(normal).dot(expected) > 0.97, "eye {eye}: normal {} for {expected}", v3(normal));
        assert!(far_hit[3] < 0.1, "eye {eye}: the billboard's corner ({}) covered {}", v3(far_p), far_hit[3]);
    }
}
