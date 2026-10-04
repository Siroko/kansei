//! The skinning shader on a real GPU: its vertices against the CPU reference, and a skinned mesh
//! drawn by the renderer (vertex pulling through an index buffer, the palette upload, motion
//! vectors from last frame's palette). Skipped (passes) without an adapter.

use glam::{Quat, Vec3};

use super::*;
use crate::cameras::{Camera, MOTION_VECTORS_WGSL};
use crate::geometries::Vertex;
use crate::materials::MaterialOptions;
use crate::objects::{Renderable, Scene, SceneNode};
use crate::renderers::{GBuffer, Renderer, RendererConfig};

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
fn shaders_validate_and_the_uniform_matches() {
    let lit = validate("skinned_lit", SKINNED_LIT_WGSL);
    assert_eq!(struct_size(&lit, "KanseiSkinnedSurface"), std::mem::size_of::<SkinnedLitParams>());
    let textured = validate("skinned_lit_textured", SKINNED_LIT_TEXTURED_WGSL);
    assert_eq!(struct_size(&textured, "KanseiSkinnedTexturedSurface"), std::mem::size_of::<SkinnedLitParams>());
    validate("skin compute", &skin_compute_wgsl());
    validate("strip", &strip_wgsl());
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    pollster::block_on(adapter.request_device(&Default::default(), None)).ok()
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let floats = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    floats
}

/// Runs `kansei_skin` over every vertex: position, normal and last frame's position.
fn skin_compute_wgsl() -> String {
    format!(
        "{SKINNING_WGSL}\n{}",
        r#"
struct In { position: vec4f, normal: vec4f };
@group(0) @binding(3) var<storage, read> vertices: array<In>;
@group(0) @binding(4) var<storage, read_write> skinned: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&vertices)) { return; }
    let s = kansei_skin(i, vertices[i].position.xyz, vertices[i].normal.xyz);
    skinned[3u * i] = vec4f(s.position, 1.0);
    skinned[3u * i + 1u] = vec4f(s.normal, 0.0);
    skinned[3u * i + 2u] = vec4f(s.prev_position, 1.0);
}
"#
    )
}

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

/// A chain of five joints up +y, 0.4 m apart, with 200 vertices around it each weighted to up to
/// four joints.
fn chain_mesh() -> (Skeleton, SkinnedMesh) {
    let n: usize = 5;
    let skeleton = Skeleton::new(
        (0..n).map(|i| format!("j{i}")).collect(),
        (0..n).map(|i| i.checked_sub(1)).collect(),
        (0..n).map(|i| Transform::from_translation_rotation(if i == 0 { Vec3::ZERO } else { Vec3::new(0.0, 0.4, 0.0) }, Quat::IDENTITY)).collect(),
    );
    let bind = skeleton.rest_model();
    let mut vertices = Vec::new();
    let mut joints = Vec::new();
    let mut weights = Vec::new();
    for v in 0..200u32 {
        let y = hash(v) * 1.8;
        let a = hash(v + 1000) * std::f32::consts::TAU;
        vertices.push(Vertex { position: [0.1 * a.cos(), y, 0.1 * a.sin(), 1.0], normal: [a.cos(), 0.0, a.sin()], uv: [0.0; 2] });
        let (j, w) = strongest_influences((0..4).map(|k| ((hash(v * 7 + k) * n as f32) as u16 % n as u16, hash(v * 13 + k) + 0.05)));
        joints.push(j);
        weights.push(w);
    }
    let mesh = SkinnedMesh {
        name: "chain".into(),
        indices: (0..vertices.len() as u32).collect(),
        vertices,
        joints,
        weights,
        skin_joints: (0..n).collect(),
        inverse_bind: bind.iter().map(|t| t.to_mat4().inverse()).collect(),
        material: None,
    };
    (skeleton, mesh)
}

/// The chain bent at every joint by `amount`.
fn bent(skeleton: &Skeleton, amount: f32) -> Vec<Transform> {
    let mut pose = Pose::rest(skeleton);
    for (i, t) in pose.local.iter_mut().enumerate() {
        t.rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.3 * amount * i as f32, 0.2 * amount, 0.4 * amount);
    }
    pose.model(skeleton)
}

#[test]
fn the_gpu_skins_vertices_as_the_cpu_does() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    use wgpu::util::DeviceExt;
    let (skeleton, mesh) = chain_mesh();
    let mut palette = BonePalette::new(mesh.skin_joints.len());
    palette.update(&mesh, &bent(&skeleton, 0.5));
    palette.update(&mesh, &bent(&skeleton, 1.0));

    let storage = |data: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC });
    let bones = storage(palette.as_bytes());
    let skin = storage(bytemuck::cast_slice(&mesh.skin_words()));
    let inputs: Vec<[f32; 8]> = mesh.vertices.iter().map(|v| [v.position[0], v.position[1], v.position[2], 1.0, v.normal[0], v.normal[1], v.normal[2], 0.0]).collect();
    let inputs = storage(bytemuck::cast_slice(&inputs));
    let output = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (mesh.vertices.len() * 48) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(skin_compute_wgsl().into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: PALETTE_BINDING, resource: bones.as_entire_binding() },
            wgpu::BindGroupEntry { binding: SKIN_BINDING, resource: skin.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: inputs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(mesh.vertices.len().div_ceil(64) as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let gpu = read_back(&device, &queue, &output);

    let now = mesh.skin_cpu(palette.current());
    let before = mesh.skin_cpu(palette.previous());
    let mut moved = 0;
    for (v, ((p, n), (q, _))) in now.iter().zip(&before).enumerate() {
        let at = |k: usize| Vec3::from_slice(&gpu[(3 * v + k) * 4..]);
        // weights are unorm16 on the GPU: agree to a fraction of a millimetre
        assert!(at(0).abs_diff_eq(*p, 2e-4), "vertex {v}: {} vs {p}", at(0));
        assert!(at(1).abs_diff_eq(*n, 2e-4), "normal {v}: {} vs {n}", at(1));
        assert!(at(2).abs_diff_eq(*q, 2e-4), "previous {v}: {} vs {q}", at(2));
        moved += (p.distance(*q) > 0.01) as usize;
    }
    assert!(moved > 150, "the two frames' palettes differ: {moved} vertices moved");
}

/// A skinned strip: white, its motion written to the velocity target.
fn strip_wgsl() -> String {
    format!(
        "{SKINNING_WGSL}\n{MOTION_VECTORS_WGSL}\n{}",
        r#"
struct Params { color: vec4f };
@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;
struct VOut { @builtin(position) @invariant clip: vec4f, @location(0) curr: vec4f, @location(1) prev: vec4f };
struct FOut { @location(0) color: vec4f, @location(4) velocity: vec2f };
@vertex
fn vertex_main(@builtin(vertex_index) vertex: u32, @location(0) position: vec4f, @location(1) normal: vec3f) -> VOut {
    let s = kansei_skin(vertex, position.xyz, normal);
    let world = mesh.world * vec4f(s.position, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(s.prev_position, 1.0));
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    return FOut(params.color, kansei_motion_vector(in.curr, in.prev));
}
"#
    )
}

/// A strip 0.2 m wide from y = 0 to 2 in the z = 0 plane, drawn through an index buffer: the
/// lower quad on the hip joint, the upper quad on the knee joint at y = 1.
fn strip() -> (Skeleton, SkinnedMesh) {
    let skeleton = Skeleton::new(
        vec!["hip".into(), "knee".into()],
        vec![None, Some(0)],
        vec![Transform::IDENTITY, Transform::from_translation_rotation(Vec3::Y, Quat::IDENTITY)],
    );
    let bind = skeleton.rest_model();
    let mut vertices = Vec::new();
    let mut joints = Vec::new();
    for (joint, rows) in [(0u16, [0.0f32, 1.0]), (1, [1.0, 2.0])] {
        for y in rows {
            for x in [-0.1f32, 0.1] {
                vertices.push(Vertex { position: [x, y, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [0.0; 2] });
                joints.push([joint, 0, 0, 0]);
            }
        }
    }
    let weights = vec![[1.0, 0.0, 0.0, 0.0]; vertices.len()];
    let indices = vec![0, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6];
    let mesh = SkinnedMesh { name: "strip".into(), vertices, indices, joints, weights, skin_joints: vec![0, 1], inverse_bind: bind.iter().map(|t| t.to_mat4().inverse()).collect(), material: None };
    (skeleton, mesh)
}

const SIZE: u32 = 64;

/// Colour coverage and velocity of a GBuffer after the renderer drew into it.
fn read_targets(renderer: &Renderer, gbuffer: &GBuffer) -> (Vec<bool>, Vec<[f32; 2]>) {
    let (device, queue) = (renderer.device(), renderer.queue());
    let read = |texture: &wgpu::Texture, bytes_per_texel: u32| {
        let row = (SIZE * bytes_per_texel).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * SIZE) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let bytes = buffer.slice(..).get_mapped_range().to_vec();
        (0..SIZE * SIZE).map(move |i| bytes[((i / SIZE) * row + (i % SIZE) * bytes_per_texel) as usize..][..bytes_per_texel as usize].to_vec()).collect::<Vec<_>>()
    };
    let half = |b: &[u8]| half_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let covered = read(&gbuffer.color_texture, 8).iter().map(|t| half(&t[6..8]) > 0.5).collect();
    let velocity = read(&gbuffer.velocity_texture, 4).iter().map(|t| [half(&t[0..2]), half(&t[2..4])]).collect();
    (covered, velocity)
}

fn half_to_f32(bits: u16) -> f32 {
    let (sign, exponent, mantissa) = ((bits >> 15) as u32, ((bits >> 10) & 0x1f) as i32, (bits & 0x3ff) as f32);
    let magnitude = match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// Pixel of a world point under a view-projection matrix.
fn pixel(view_proj: glam::Mat4, p: Vec3) -> (u32, u32) {
    let clip = view_proj * p.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    (((ndc.x * 0.5 + 0.5) * SIZE as f32) as u32, ((0.5 - ndc.y * 0.5) * SIZE as f32) as u32)
}

#[test]
fn the_renderer_draws_the_skinned_pose_and_its_motion() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut renderer = Renderer::new(RendererConfig { width: SIZE, height: SIZE, clear_color: crate::math::Vec4::new(0.0, 0.0, 0.0, 0.0), ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));

    let (skeleton, mesh) = strip();
    let mut palette = BonePalette::new(2);
    palette.update(&mesh, &skeleton.rest_model());
    let material = skinned_material("Strip", &strip_wgsl(), &[[1.0f32; 4]], &mesh, &palette, MaterialOptions { outputs_velocity: true, cull_mode: crate::materials::CullMode::None, ..Default::default() });
    let mut renderable = Renderable::new(mesh.geometry(), material);
    renderable.dynamic = true;
    let mut scene = Scene::new();
    let index = scene.add(SceneNode::Renderable(renderable));
    let mut camera = Camera::new(60.0, 0.1, 50.0, 1.0);
    camera.set_position(0.0, 1.0, 3.0);
    camera.look_at(&crate::math::Vec3::new(0.0, 1.0, 0.0));
    camera.update_projection_matrix();

    let gbuffer = GBuffer::new(renderer.device(), SIZE, SIZE, 1);
    renderer.render_scene_to_gbuffer(&mut scene, &mut camera, &gbuffer);
    let (rest, _) = read_targets(&renderer, &gbuffer);
    let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
    let at = |covered: &[bool], p: Vec3| {
        let (x, y) = pixel(view_proj, p);
        covered[(y * SIZE + x) as usize]
    };
    assert!(at(&rest, Vec3::new(0.0, 0.5, 0.0)) && at(&rest, Vec3::new(0.0, 1.5, 0.0)), "the strip stands upright");
    assert!(!at(&rest, Vec3::new(-0.5, 1.0, 0.0)));

    // bend the knee 90 degrees about z: the upper half now reaches along -x
    let mut pose = Pose::rest(&skeleton);
    pose.local[1].rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    palette.update(&mesh, &pose.model(&skeleton));
    let r = scene.get_renderable_mut(index).unwrap();
    palette.upload(renderer.queue(), &r.material.bindable_buffer(PALETTE_BINDING).expect("the palette is on the GPU after the first render"));
    camera.end_frame();
    renderer.render_scene_to_gbuffer(&mut scene, &mut camera, &gbuffer);
    let (bent, velocity) = read_targets(&renderer, &gbuffer);
    assert!(at(&bent, Vec3::new(0.0, 0.5, 0.0)), "the lower half stays");
    assert!(!at(&bent, Vec3::new(0.0, 1.6, 0.0)), "the upper half left");
    assert!(at(&bent, Vec3::new(-0.6, 1.0, 0.0)), "the upper half reaches along -x");
    // the upper half moved left and down since last frame, the lower half did not move
    let (x, y) = pixel(view_proj, Vec3::new(-0.6, 1.0, 0.0));
    let moved = velocity[(y * SIZE + x) as usize];
    assert!(moved[0] < -0.05 && moved[1] > 0.02, "{moved:?}");
    let (x, y) = pixel(view_proj, Vec3::new(0.0, 0.4, 0.0));
    let still = velocity[(y * SIZE + x) as usize];
    assert!(still[0].abs() < 1e-3 && still[1].abs() < 1e-3, "{still:?}");
}
