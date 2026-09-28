use super::gpu::*;
use super::tests::rock;
use super::*;

pub(super) fn validate(name: &str, code: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    module
}

pub(super) fn struct_size(module: &naga::Module, name: &str) -> usize {
    module
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no struct {name}"))
}

/// A device, or None without an adapter.
pub(super) fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    pollster::block_on(adapter.request_device(&Default::default(), None)).ok()
}

pub(super) fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    words
}

#[test]
fn the_gpu_words_hold_the_clusters_and_levels() {
    let mesh = ClusterMesh::build(&rock(3, false), &ClusterOptions::default());
    let words = mesh.gpu_words();
    let section = |k: usize| words[k] as usize;
    let levels = mesh.levels();
    assert_eq!((section(5), section(6), words[7]), (mesh.clusters.len(), levels.len(), mesh.max_triangles()));
    assert_eq!(words.len(), section(4) + levels.len() * LEVEL_WORDS);
    assert_eq!(f32::from_bits(words[section(0) + 5 * VERTEX_WORDS + 4]), mesh.vertices[5].normal[0]);
    for (i, c) in mesh.clusters.iter().enumerate() {
        let record = &words[section(3) + i * CLUSTER_WORDS..][..CLUSTER_WORDS];
        let decoded: Vec<[u32; 3]> = (0..record[2])
            .map(|t| {
                let packed = words[section(2) + (record[1] + t) as usize];
                [0, 1, 2].map(|k| words[section(1) + record[0] as usize + ((packed >> (8 * k)) & 0xff) as usize])
            })
            .collect();
        assert_eq!(decoded, mesh.triangles(i).collect::<Vec<_>>(), "cluster {i}");
        assert_eq!(f32::from_bits(record[15]), c.error);
        assert_eq!(f32::from_bits(record[24]), if c.parent_error.is_finite() { c.parent_error } else { NO_PARENT });
    }
    let root = &words[section(4) + (levels.len() - 1) * LEVEL_WORDS..];
    assert_eq!((root[0], root[1], f32::from_bits(root[3])), (levels.last().unwrap().first, levels.last().unwrap().count, NO_PARENT));
    // no infinity or NaN among the records' floats
    let floats = &words[section(3)..];
    assert!(floats.iter().all(|&w| (w >> 23) & 0xff != 0xff), "an infinity or NaN reaches the GPU");
}

const FETCH_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> kansei_cluster_mesh: array<u32>;
@group(0) @binding(1) var<storage, read_write> fetched: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let per_cluster = 3u * kansei_cluster_mesh[7];
    if (id.x >= per_cluster || id.y >= kansei_cluster_mesh[5]) {
        return;
    }
    let vertex = kansei_cluster_vertex(id.y, id.x);
    fetched[id.y * per_cluster + id.x] = vec2<u32>(vertex, bitcast<u32>(kansei_vertex_f32(vertex, 1u)));
}
"#;

#[test]
fn the_gpu_fetches_every_clusters_vertices() {
    let code = format!("{FETCH_WGSL}\n{CLUSTER_MESH_WGSL}");
    validate("fetch", &code);
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    use wgpu::util::DeviceExt;
    let mesh = ClusterMesh::build(&rock(3, true), &ClusterOptions::default());
    let per_cluster = 3 * mesh.max_triangles() as usize;
    let words = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&mesh.gpu_words()), usage: wgpu::BufferUsages::STORAGE });
    let fetched = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (mesh.clusters.len() * per_cluster * 8) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: words.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: fetched.as_entire_binding() }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(per_cluster.div_ceil(64) as u32, mesh.clusters.len() as u32, 1);
    }
    queue.submit(Some(encoder.finish()));
    let fetched = read_words(&device, &queue, &fetched);
    for (i, c) in mesh.clusters.iter().enumerate() {
        let triangles: Vec<[u32; 3]> = mesh.triangles(i).collect();
        for k in 0..per_cluster {
            // past its triangles, the first corner: triangles of no area
            let expected = if k / 3 < c.triangle_count as usize { triangles[k / 3][k % 3] } else { triangles[0][0] };
            let at = 2 * (i * per_cluster + k);
            assert_eq!(fetched[at], expected, "cluster {i}, vertex {k}");
            assert_eq!(f32::from_bits(fetched[at + 1]), mesh.vertices[expected as usize].position[1]);
        }
    }
}

use std::collections::BTreeSet;

#[test]
fn the_cull_shader_validates_and_its_uniforms_match() {
    let module = validate("cluster_cull", &format!("{CLUSTER_CULL_WGSL}\n{CLUSTER_MESH_WGSL}"));
    assert_eq!(struct_size(&module, "ClusterCull"), std::mem::size_of::<ClusterCullGpu>());
    assert_eq!(struct_size(&module, "ClusterView"), std::mem::size_of::<ClusterViewGpu>());
    assert_eq!(struct_size(&module, "ClusterDraw") as u64, DRAW_ARGS_BYTES);
}

/// A view for the cull tests: a camera at `eye` looking at `target` (60°, square), 512 pixels
/// high, and the budget.
struct TestView {
    view_proj: glam::Mat4,
    eye: glam::Vec3,
    ppr: f32,
    near: f32,
    threshold: f32,
    orthographic: bool,
}

impl TestView {
    fn looking(eye: glam::Vec3, target: glam::Vec3, threshold: f32) -> Self {
        let fov = 60f32.to_radians();
        let view_proj = glam::Mat4::perspective_rh(fov, 1.0, 0.1, 1000.0) * glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        Self { view_proj, eye, ppr: 512.0 / (2.0 * (fov / 2.0).tan()), near: 0.1, threshold, orthographic: false }
    }

    /// Straight down from `eye` onto a square `width` metres across, 512 pixels (a shadow
    /// cascade's or the sky occlusion's kind of view).
    fn top_down(eye: glam::Vec3, width: f32, threshold: f32) -> Self {
        let h = width / 2.0;
        let view_proj = glam::Mat4::orthographic_rh(-h, h, -h, h, 0.1, 1000.0) * glam::Mat4::look_at_rh(eye, eye - glam::Vec3::Y, glam::Vec3::NEG_Z);
        Self { view_proj, eye, ppr: 512.0 / width, near: 0.1, threshold, orthographic: true }
    }

    fn gpu(&self) -> ClusterViewGpu {
        ClusterViewGpu::new(self.view_proj, self.eye, self.ppr, self.near, self.threshold, self.orthographic)
    }
}

/// What the cull should draw of `mesh` placed by `model` (a uniform scale, maybe mirrored):
/// M1's `select` in the mesh's space, less the clusters outside the frustum and (with `cone`)
/// those facing away. Also returns the clusters within rounding of any of those decisions.
fn expected(mesh: &ClusterMesh, model: glam::Mat4, v: &TestView, cone: bool) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let m = glam::Mat3::from_mat4(model);
    let scale = m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length());
    let eye = model.inverse().transform_point3(v.eye);
    let lod = LodView { eye, pixels_per_radian: v.ppr, near: v.near / scale, threshold: v.threshold, orthographic: v.orthographic };
    let cone = cone && m.determinant() > 0.0 && !v.orthographic;
    let planes = crate::culling::frustum_planes(v.view_proj);
    let selected: BTreeSet<usize> = mesh.select(&lod).into_iter().collect();
    let close = |p: f32| (p - v.threshold).abs() <= 2e-3 * v.threshold.max(1e-3);
    let (mut drawn, mut ambiguous) = (BTreeSet::new(), BTreeSet::new());
    for (i, c) in mesh.clusters.iter().enumerate() {
        let center = model.transform_point3(c.bounds.center);
        let radius = c.bounds.radius * scale;
        let outside: Vec<f32> = planes.iter().map(|p| p.truncate().dot(center) + p.w + radius).collect();
        let facing = (c.cone_apex - eye).normalize_or_zero().dot(c.cone_axis) - c.cone_cutoff;
        if close(projected_error(c.error, c.lod_bounds, &lod))
            || close(projected_error(c.parent_error, c.parent_bounds, &lod))
            || outside.iter().any(|d| d.abs() < 1e-4 * radius.max(1.0))
            || (cone && facing.abs() < 1e-4)
        {
            ambiguous.insert(i as u32);
        } else if selected.contains(&i) && outside.iter().all(|&d| d >= 0.0) && !(cone && facing >= 0.0) {
            drawn.insert(i as u32);
        }
    }
    (drawn, ambiguous)
}

/// Cull once and read back the draw's words and the drawn (record, cluster) pairs.
fn cull(device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, gpu: &mut ClusterGpu, source: InstanceSource, params: ClusterCullGpu, view: &TestView) -> (Vec<u32>, Vec<(u32, u32)>) {
    gpu.bind(device, queue, culling, 0, source, params);
    culling.set_views(queue, &[view.gpu()]);
    let mut encoder = device.create_command_encoder(&Default::default());
    culling.encode(&mut encoder, &[(&*gpu, 0)]);
    queue.submit(Some(encoder.finish()));
    let args = read_words(device, queue, gpu.args(0));
    let list = read_words(device, queue, gpu.draws(0));
    let pairs = list.chunks(2).take(args[1] as usize).map(|p| (p[0], p[1])).collect();
    (args, pairs)
}

/// Asserts the GPU drew, of record `record`, the expected clusters (give or take the ambiguous).
fn assert_cut(label: &str, pairs: &[(u32, u32)], record: u32, (drawn, ambiguous): &(BTreeSet<u32>, BTreeSet<u32>)) {
    let gpu: BTreeSet<u32> = pairs.iter().filter(|p| p.0 == record).map(|p| p.1).collect();
    assert_eq!(gpu.len(), pairs.iter().filter(|p| p.0 == record).count(), "{label}: a cluster drawn twice");
    let missing: Vec<_> = drawn.difference(&gpu).collect();
    let extra: Vec<_> = gpu.difference(drawn).filter(|c| !ambiguous.contains(c)).collect();
    assert!(missing.is_empty() && extra.is_empty(), "{label}: missing {missing:?}, extra {extra:?} ({} expected)", drawn.len());
}

fn buffer(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(words), usage: wgpu::BufferUsages::STORAGE })
}

/// Records of 12 floats: position, scale, yaw, pad, rotation (x y z w), pad.
const PLACEMENT: InstanceTransform = InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: Some(24) };

fn placement_record(position: glam::Vec3, scale: f32, yaw: f32, rotation: glam::Quat) -> [f32; 12] {
    [position.x, position.y, position.z, scale, yaw, 0.0, rotation.x, rotation.y, rotation.z, rotation.w, 0.0, 0.0]
}

fn placement_matrix(r: &[f32; 12]) -> glam::Mat4 {
    glam::Mat4::from_translation(glam::Vec3::new(r[0], r[1], r[2])) * glam::Mat4::from_rotation_y(r[4]) * glam::Mat4::from_quat(glam::Quat::from_xyzw(r[6], r[7], r[8], r[9])) * glam::Mat4::from_scale(glam::Vec3::splat(r[3]))
}

#[test]
fn the_gpu_cut_of_placed_instances_is_the_cpu_cut() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let records = [
        placement_record(glam::Vec3::ZERO, 1.0, 0.0, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(6.0, 0.0, -4.0), 2.5, 1.2, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(-5.0, 1.5, -9.0), 0.5, -2.0, glam::Quat::from_rotation_x(0.7)),
        placement_record(glam::Vec3::new(30.0, 0.0, -60.0), 3.0, 0.3, glam::Quat::IDENTITY),
    ];
    let world = glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(1.5), glam::Quat::from_rotation_z(0.2), glam::Vec3::new(1.0, 0.0, 0.0));
    let record_buffer = buffer(&device, bytemuck::cast_slice(&records.concat()));
    let culling = ClusterCulling::new(&device);
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let source = InstanceSource::All { records: &record_buffer, count: records.len() as u32 };
    let params = ClusterCullGpu::new(world, Some(PLACEMENT), 48, &source, 4 * mesh.clusters.len() as u32, gpu.vertex_count(), true, 1.0);
    let mut totals = Vec::new();
    for (eye, target) in [(glam::Vec3::new(0.0, 2.0, 8.0), glam::Vec3::ZERO), (glam::Vec3::new(3.0, 0.5, 1.5), glam::Vec3::new(1.0, 0.0, 0.0)), (glam::Vec3::new(-20.0, 10.0, 30.0), glam::Vec3::new(10.0, 0.0, -30.0))] {
        for threshold in [0.0, 0.5, 1.0, 4.0] {
            let view = TestView::looking(eye, target, threshold);
            let (args, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
            assert_eq!((args[0], args[4]), (gpu.vertex_count(), records.len() as u32));
            for (k, r) in records.iter().enumerate() {
                let label = format!("eye {eye}, {threshold} px, instance {k}");
                assert_cut(&label, &pairs, k as u32, &expected(&mesh, world * placement_matrix(r), &view, true));
            }
            let triangles: u32 = pairs.iter().map(|p| mesh.clusters[p.1 as usize].triangle_count).sum();
            assert_eq!(args[6], triangles, "the triangles drawn");
            totals.push(triangles);
        }
    }
    // a coarser budget draws fewer triangles
    assert!(totals[3] < totals[0] && totals[0] > 0, "{totals:?}");
}

#[test]
fn matrix_culled_and_single_instances_cut_as_the_cpu_does() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let culling = ClusterCulling::new(&device);
    let view = TestView::looking(glam::Vec3::new(1.0, 1.5, 6.0), glam::Vec3::new(0.0, 0.0, -2.0), 1.0);
    let world = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.5, 0.0));

    // matrices, one of them mirrored (no cone test there: its winding is turned over), behind
    // two records the culled view doesn't count, with its count in word 5 of the instance draws
    let matrices = [
        glam::Mat4::IDENTITY,
        glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(2.0), glam::Quat::from_rotation_y(0.4), glam::Vec3::new(3.0, 0.0, -4.0)),
        glam::Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, -2.0)) * glam::Mat4::from_scale(glam::Vec3::new(-1.2, 1.2, 1.2)),
    ];
    let mut words: Vec<f32> = vec![9.0; 32];
    for m in &matrices {
        words.extend(m.to_cols_array());
    }
    let records = buffer(&device, bytemuck::cast_slice(&words));
    let instance_args = buffer(&device, &[0, 0, 0, 0, 0, 3, 0, 0]);
    let source = InstanceSource::Culled { records: &records, first_record: 2, capacity: 3, args: &instance_args, count_word: 5 };
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let params = ClusterCullGpu::new(world, Some(InstanceTransform::Matrix { offset: 0 }), 64, &source, 3 * mesh.clusters.len() as u32, gpu.vertex_count(), true, 1.0);
    let (args, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
    assert_eq!(args[4], 3);
    assert!(pairs.iter().all(|p| (2..5).contains(&p.0)), "records from the view's first");
    for (k, m) in matrices.iter().enumerate() {
        assert_cut(&format!("matrix {k}"), &pairs, 2 + k as u32, &expected(&mesh, world * *m, &view, true));
    }

    // no instances: the mesh once, where the renderable is
    let mut single = ClusterGpu::new(&device, &mesh);
    let params = ClusterCullGpu::new(world, None, 0, &InstanceSource::None, mesh.clusters.len() as u32, single.vertex_count(), true, 1.0);
    let (args, pairs) = cull(&device, &queue, &culling, &mut single, InstanceSource::None, params, &view);
    assert_eq!(args[4], 1);
    assert_cut("single", &pairs, 0, &expected(&mesh, world, &view, true));
}

#[test]
fn nothing_visible_draws_nothing_and_the_capacity_holds() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let culling = ClusterCulling::new(&device);
    let view = TestView::looking(glam::Vec3::new(0.0, 0.0, 1.5), glam::Vec3::ZERO, 0.0);
    let records = buffer(&device, bytemuck::cast_slice(&placement_record(glam::Vec3::ZERO, 1.0, 0.0, glam::Quat::IDENTITY)));
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let counted = |count: u32| buffer(&device, &[0, count, 0, 0, 0, 0, 0, 0]);

    // a full list: only the capacity drawn, the rest counted
    let one = counted(1);
    let source = InstanceSource::Culled { records: &records, first_record: 0, capacity: 1, args: &one, count_word: 1 };
    let params = ClusterCullGpu::new(glam::Mat4::IDENTITY, Some(PLACEMENT), 48, &source, 5, gpu.vertex_count(), true, 1.0);
    let (args, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
    assert_eq!(args[1], 5, "drawn: the capacity");
    assert!(args[5] > 5, "claimed: {}", args[5]);
    assert!(pairs.iter().all(|p| p.0 == 0 && (p.1 as usize) < mesh.clusters.len()));

    // then no instance visible: nothing drawn, nothing left from before
    let none = counted(0);
    let source = InstanceSource::Culled { records: &records, first_record: 0, capacity: 1, args: &none, count_word: 1 };
    let (args, _) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
    assert_eq!((args[1], args[4], args[5], args[6]), (0, 0, 0, 0));
}

use crate::buffers::InstanceBufferLayout;
use wgpu::VertexFormat::*;

fn layout(stride: u64, attributes: &[(u32, u64, wgpu::VertexFormat)]) -> InstanceBufferLayout {
    InstanceBufferLayout { stride, attributes: attributes.iter().map(|&(shader_location, offset, format)| wgpu::VertexAttribute { format, offset, shader_location }).collect() }
}

fn validate_stage(name: &str, code: &str, instances: Option<&InstanceBufferLayout>) -> naga::Module {
    let stage = cluster_vertex_stage(code, instances).unwrap_or_else(|e| panic!("{name}: {e}"));
    let module = validate(name, &stage);
    assert!(module.entry_points.iter().any(|e| e.name == CLUSTER_VERTEX_ENTRY && e.stage == naga::ShaderStage::Vertex), "{name}: no cluster entry point");
    assert!(!module.entry_points.iter().any(|e| e.name == "vertex_main"), "{name}: vertex_main is still an entry point");
    module
}

#[test]
fn the_engine_materials_get_a_cluster_stage() {
    validate_stage("basic", include_str!("../shaders/basic.wgsl"), None);
    validate_stage("basic_lit", include_str!("../shaders/basic_lit.wgsl"), None);
    let mat4 = layout(64, &[(3, 0, Float32x4), (4, 16, Float32x4), (5, 32, Float32x4), (6, 48, Float32x4)]);
    validate_stage("basic_instanced", include_str!("../shaders/basic_instanced.wgsl"), Some(&mat4));
    validate_stage("particle_billboard", include_str!("../shaders/particle_billboard.wgsl"), Some(&layout(16, &[(3, 0, Float32x4)])));
}

/// The forms the repository's materials take (the examples, the film): located parameters with
/// comments and a trailing comma, one returning a builtin; and a struct of located members with
/// instance attributes of each kind.
#[test]
fn located_and_struct_forms_get_a_cluster_stage() {
    let located = r#"
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) n: vec3<f32> };
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
@vertex
fn vertex_main(
    // per vertex
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>, /* unused */
) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.n = normal + vec3<f32>(uv, 0.0);
    return out;
}
@fragment fn fragment_main(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(in.n, 1.0); }
"#;
    validate_stage("located", located, None);
    let builtin_out = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex fn vertex_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> { return view_matrix * position; }
@fragment fn fragment_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
"#;
    validate_stage("builtin return", builtin_out, None);
    let instanced = r#"
struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) place: vec4<f32>,
    @location(4) yaw: f32,
    @location(5) ids: vec2<u32>,
    @location(6) offset: vec3i,
};
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) @interpolate(flat) id: u32 };
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = view_matrix * vec4<f32>(v.position.xyz * v.place.w + v.place.xyz + vec3<f32>(v.offset) + v.normal * v.yaw, 1.0);
    out.id = v.ids.x + v.ids.y;
    return out;
}
@fragment fn fragment_main(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(f32(in.id)); }
"#;
    let records = layout(48, &[(3, 0, Float32x4), (4, 16, Float32), (5, 20, Uint32x2), (6, 28, Sint32x3)]);
    validate_stage("instanced struct", instanced, Some(&records));
}

#[test]
fn inputs_the_cluster_path_cannot_feed_are_errors() {
    let error = |code: &str, instances: Option<&InstanceBufferLayout>| cluster_vertex_stage(code, instances).expect_err(code);
    assert!(error("@vertex fn vertex_main(@location(0) p: vec4<f32>, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> { return p; }", None).contains("instance_index"));
    assert!(error("struct VIn { @location(0) p: vec4<f32>, @builtin(vertex_index) i: u32 };\n@vertex fn vertex_main(v: VIn) -> @builtin(position) vec4<f32> { return v.p; }", None).contains("vertex_index"));
    assert!(error("@vertex fn vertex_main(@location(3) q: vec4<f32>) -> @builtin(position) vec4<f32> { return q; }", None).contains("location(3)"));
    let floats = layout(16, &[(3, 0, Float32x4)]);
    assert!(error("@vertex fn vertex_main(@location(3) q: vec4<u32>) -> @builtin(position) vec4<f32> { return vec4<f32>(q); }", Some(&floats)).contains("location(3)"));
    assert!(error("@fragment fn fragment_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }", None).contains("vertex_main"));
}

use crate::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use crate::cameras::Camera;
use crate::culling::InstanceCulling;
use crate::geometries::InstancedGeometry;
use crate::materials::{Binding, Material, MaterialOptions, ShaderStages};
use crate::objects::{Renderable, Scene, SceneNode};
use crate::renderers::{GBuffer, Renderer, RendererConfig};

/// Rocks placed by records of position + scale, then yaw (8 floats), coloured by their normals
/// in every GBuffer target.
const ROCKS_WGSL: &str = r#"
struct Tint { color: vec4<f32> };
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) yaw: f32 };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32> };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let local = turn(v.position.xyz * v.place.w, v.yaw) + v.place.xyz;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(local, 1.0);
    out.normal = (world_matrix * vec4<f32>(turn(v.normal, v.yaw), 0.0)).xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0) * tint.color;
    return FOut(n, vec4<f32>(0.0), n, n);
}
"#;

const SIZE: u32 = 192;

fn headless() -> Option<Renderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: SIZE, height: SIZE, clear_color: crate::math::Vec4::new(0.0, 0.0, 0.0, 0.0), ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// A scene of rocks at `placements` (x, y, z, scale, yaw), culled per instance (the first
/// `visible` records counted), with cluster LOD or without, and a camera looking at them.
fn rocks(renderer: &Renderer, placements: &[[f32; 5]], visible: u32, clusters: bool) -> (Scene, Camera, usize) {
    let mut scene = Scene::new();
    let index = scene.add(SceneNode::Renderable(rock_renderable(renderer, placements, visible, clusters)));
    let mut camera = Camera::new(50.0, 0.1, 200.0, 1.0);
    camera.set_position(0.5, 1.5, 6.0);
    camera.look_at(&crate::math::Vec3::new(0.0, 0.0, -1.5));
    camera.update_projection_matrix();
    (scene, camera, index)
}

/// `rocks`' renderable.
fn rock_renderable(renderer: &Renderer, placements: &[[f32; 5]], visible: u32, clusters: bool) -> Renderable {
    use wgpu::util::DeviceExt;
    let data: Vec<f32> = placements.iter().flat_map(|p| [p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, 0.0]).collect();
    let source = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE });
    let instances = ComputeBuffer::from_external("Rocks", source.clone(), BufferType::Storage).with_vertex_layout(
        32,
        vec![InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 }, InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32 }],
    );
    let mut material = Material::new("Rocks", ROCKS_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    let mut r = Renderable::new(InstancedGeometry::new(rock(4, false), visible, vec![instances]), material);
    r.instance_culling = Some(InstanceCulling::new(source, visible, 32, 0, 1.2).with_radius_scale(12));
    if clusters {
        let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
        r.clusters = Some(ClusterLod::new(mesh).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None }));
    }
    r
}

/// A half float's value.
fn half(bits: u16) -> f32 {
    let (sign, exponent, mantissa) = ((bits >> 15) as u32, ((bits >> 10) & 0x1f) as i32, (bits & 0x3ff) as f32);
    let magnitude = match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

/// Draw the scene into a GBuffer and read its colour target back (rgba16float; the test
/// material writes its normal-coded colour there, as into the albedo).
fn draw(renderer: &mut Renderer, scene: &mut Scene, camera: &mut Camera) -> Vec<[f32; 4]> {
    let gbuffer = GBuffer::new(renderer.device(), SIZE, SIZE, 1);
    renderer.render_scene_to_gbuffer(scene, camera, &gbuffer);
    let (device, queue) = (renderer.device(), renderer.queue());
    let row = (SIZE * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * SIZE) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        gbuffer.color_texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    (0..SIZE * SIZE)
        .map(|i| {
            let at = ((i / SIZE) * row + (i % SIZE) * 8) as usize;
            [0, 1, 2, 3].map(|c| half(u16::from_le_bytes([bytes[at + 2 * c], bytes[at + 2 * c + 1]])))
        })
        .collect()
}

/// (texels covered, texels differing by more than 1/255 in a channel).
fn compare(a: &[[f32; 4]], b: &[[f32; 4]]) -> (usize, usize) {
    let covered = a.iter().filter(|t| t[3] > 0.5).count();
    let differing = a.iter().zip(b).filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| (p - q).abs() > 1.0 / 255.0)).count();
    (covered, differing)
}

/// The cluster draw's words, read back.
fn cluster_args(renderer: &Renderer, scene: &Scene, index: usize) -> Vec<u32> {
    let gpu = scene.get_renderable(index).unwrap().clusters.as_ref().expect("still on the cluster path").gpu.as_ref().unwrap();
    read_words(renderer.device(), renderer.queue(), gpu.args(0))
}

const PLACEMENTS: [[f32; 5]; 4] = [[0.0, 0.0, 0.0, 1.0, 0.0], [2.6, 0.3, -1.5, 0.8, 1.1], [-2.4, -0.2, -2.0, 1.2, -0.6], [0.5, 1.8, -4.0, 1.5, 2.2]];

#[test]
fn at_zero_error_the_clusters_draw_what_the_mesh_does() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    renderer.set_cluster_error_threshold(0.0);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, false);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, true);
    let clusters = draw(&mut renderer, &mut scene, &mut camera);
    let args = cluster_args(&renderer, &scene, index);
    assert!(args[1] > 0 && args[4] == 4, "clusters drawn: {args:?}");
    let (covered, differing) = compare(&mesh, &clusters);
    assert!(covered > (SIZE * SIZE / 10) as usize, "the rocks cover {covered} texels");
    assert!(differing * 200 < covered, "{differing} of {covered} texels differ");
}

#[test]
fn a_pixel_of_error_draws_far_fewer_triangles_and_nearly_the_same_image() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let far: Vec<[f32; 5]> = (0..12).map(|k| [(k % 4) as f32 * 3.0 - 4.5, 0.0, -10.0 - (k / 4) as f32 * 6.0, 1.3, k as f32]).collect();
    let mut runs = Vec::new();
    for threshold in [0.0, 1.0] {
        renderer.set_cluster_error_threshold(threshold);
        let (mut scene, mut camera, index) = rocks(&renderer, &far, far.len() as u32, true);
        let image = draw(&mut renderer, &mut scene, &mut camera);
        runs.push((image, cluster_args(&renderer, &scene, index)[6]));
    }
    assert!(runs[1].1 * 3 < runs[0].1, "triangles at 0 and 1 px: {} and {}", runs[0].1, runs[1].1);
    // the same silhouettes to a pixel, and the shading of coarser triangles' interpolated normals
    let covered = |t: &[f32; 4]| t[3] > 0.5;
    let both: Vec<f32> = runs[0].0.iter().zip(&runs[1].0).filter(|(a, b)| covered(a) && covered(b)).map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)).collect();
    let silhouette = runs[0].0.iter().zip(&runs[1].0).filter(|(a, b)| covered(a) != covered(b)).count();
    let mean = both.iter().sum::<f32>() / both.len().max(1) as f32;
    assert!(both.len() > 500 && silhouette * 25 < both.len(), "{silhouette} silhouette texels differ, of {}", both.len());
    assert!(mean < 0.02, "shading differs by {mean} on average");
}

#[test]
fn grown_instances_rebind_the_clusters() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    renderer.set_cluster_error_threshold(0.0);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 2, true);
    draw(&mut renderer, &mut scene, &mut camera);
    // the other two records join: instance culling remakes its buffers, the clusters rebind
    let r = scene.get_renderable_mut(index).unwrap();
    r.instance_culling.as_mut().unwrap().count = 4;
    r.geometry.instance_count = 4;
    let clusters = draw(&mut renderer, &mut scene, &mut camera);
    assert_eq!(cluster_args(&renderer, &scene, index)[4], 4);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, false);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (covered, differing) = compare(&mesh, &clusters);
    assert!(differing * 200 < covered, "{differing} of {covered} texels differ");
}

#[test]
fn a_material_the_cluster_path_cannot_feed_keeps_the_mesh() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, true);
    // reads the instance index: no cluster stage, drawn as before
    let r = scene.get_renderable_mut(index).unwrap();
    r.material = Material::new(
        "Rocks",
        &ROCKS_WGSL.replace("@location(4) yaw: f32", "@location(4) yaw: f32, @builtin(instance_index) instance: u32"),
        vec![Binding::uniform(0, ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    r.material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    let image = draw(&mut renderer, &mut scene, &mut camera);
    assert!(scene.get_renderable(index).unwrap().clusters.is_none(), "cluster LOD dropped");
    assert!(image.iter().filter(|t| t[3] > 0.5).count() > (SIZE * SIZE / 10) as usize, "still drawn");
}

/// Replace renderable `index`'s material with the rocks' under `options` (4 targets).
fn rocks_material(scene: &mut Scene, index: usize, options: MaterialOptions) {
    let r = scene.get_renderable_mut(index).unwrap();
    r.material = Material::new("Rocks", ROCKS_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..options });
    r.material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
}

#[test]
fn front_culled_and_double_sided_materials_keep_the_faces_they_draw() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    renderer.set_cluster_error_threshold(0.0);
    for cull_mode in [crate::materials::CullMode::Front, crate::materials::CullMode::None] {
        let options = || MaterialOptions { cull_mode, ..Default::default() };
        let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, false);
        rocks_material(&mut scene, index, options());
        let mesh = draw(&mut renderer, &mut scene, &mut camera);
        let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, true);
        rocks_material(&mut scene, index, options());
        let clusters = draw(&mut renderer, &mut scene, &mut camera);
        let (covered, differing) = compare(&mesh, &clusters);
        assert!(covered > (SIZE * SIZE / 20) as usize && differing * 200 < covered, "{cull_mode:?}: {differing} of {covered} texels differ");
    }
}

#[test]
fn removing_cluster_lod_draws_the_mesh_again() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    renderer.set_cluster_error_threshold(0.0);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, true);
    draw(&mut renderer, &mut scene, &mut camera);
    // off, and seen from behind: the cut culled for the first view must not be drawn
    scene.get_renderable_mut(index).unwrap().clusters = None;
    let behind = |camera: &mut Camera| {
        camera.set_position(-0.5, 1.5, -10.0);
        camera.look_at(&crate::math::Vec3::new(0.0, 0.0, -1.5));
    };
    behind(&mut camera);
    let after = draw(&mut renderer, &mut scene, &mut camera);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, false);
    behind(&mut camera);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (covered, differing) = compare(&mesh, &after);
    assert!(covered > (SIZE * SIZE / 20) as usize && differing * 200 < covered, "{differing} of {covered} texels differ");
}

/// A bound on the largest scale `m` applies to any direction: the square root of the largest
/// absolute row sum of mᵀm (exact for orthogonal columns), as cluster_cull.wgsl takes it.
fn scale_bound(m: glam::Mat3) -> f32 {
    let g = m.transpose() * m;
    [g.x_axis, g.y_axis, g.z_axis].iter().map(|c| c.abs().element_sum()).fold(0.0, f32::max).sqrt()
}

/// What the cull should draw of `mesh` placed by any `model` (shear included): the cut rule in
/// world space with `scale_bound`, the frustum, and (with `cone`, unmirrored) the cone in the
/// mesh's space. Also returns the clusters within rounding of a decision.
fn expected_world(mesh: &ClusterMesh, model: glam::Mat4, v: &TestView, cone: bool, stretch: f32) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let m = glam::Mat3::from_mat4(model);
    let scale = scale_bound(m) * stretch;
    let eye = model.inverse().transform_point3(v.eye);
    let cone = cone && m.determinant() > 0.0 && !v.orthographic;
    let planes = crate::culling::frustum_planes(v.view_proj);
    // a stretch about the mesh's origin also moves a sphere's centre (cluster_cull.wgsl)
    let placed = |s: Sphere| (s.radius + (1.0 - 1.0 / stretch) * s.center.length()) * scale;
    let projected = |error: f32, s: Sphere| {
        if error == 0.0 {
            0.0
        } else if !error.is_finite() {
            f32::INFINITY
        } else if v.orthographic {
            error * scale * v.ppr
        } else {
            error * scale / (v.eye.distance(model.transform_point3(s.center)) - placed(s)).max(v.near) * v.ppr
        }
    };
    let close = |p: f32| (p - v.threshold).abs() <= 2e-3 * v.threshold.max(1e-3);
    let (mut drawn, mut ambiguous) = (BTreeSet::new(), BTreeSet::new());
    for (i, c) in mesh.clusters.iter().enumerate() {
        let (own, parent) = (projected(c.error, c.lod_bounds), projected(c.parent_error, c.parent_bounds));
        let center = model.transform_point3(c.bounds.center);
        let radius = placed(c.bounds);
        let outside: Vec<f32> = planes.iter().map(|p| p.truncate().dot(center) + p.w + radius).collect();
        let facing = (c.cone_apex - eye).normalize_or_zero().dot(c.cone_axis) - c.cone_cutoff;
        if close(own) || close(parent) || outside.iter().any(|d| d.abs() < 1e-4 * radius.max(1.0)) || (cone && facing.abs() < 1e-4) {
            ambiguous.insert(i as u32);
        } else if own <= v.threshold && parent > v.threshold && outside.iter().all(|&d| d >= 0.0) && !(cone && facing >= 0.0) {
            drawn.insert(i as u32);
        }
    }
    (drawn, ambiguous)
}

#[test]
fn sheared_transforms_keep_every_level_the_cut_needs() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    // stretched along x, instances turned by 45° and more: columns no longer orthogonal
    let world = glam::Mat4::from_scale(glam::Vec3::new(3.0, 1.0, 1.0));
    let records = [
        placement_record(glam::Vec3::ZERO, 1.0, 0.785, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(0.0, 0.0, -6.0), 1.2, 2.3, glam::Quat::from_rotation_x(0.5)),
    ];
    let record_buffer = buffer(&device, bytemuck::cast_slice(&records.concat()));
    let culling = ClusterCulling::new(&device);
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let source = InstanceSource::All { records: &record_buffer, count: records.len() as u32 };
    let params = ClusterCullGpu::new(world, Some(PLACEMENT), 48, &source, 2 * mesh.clusters.len() as u32, gpu.vertex_count(), true, 1.0);
    let mut checked = 0;
    for distance in [5.0f32, 8.0, 12.0, 20.0] {
        for k in 0..6 {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            let eye = glam::Vec3::new(distance * a.cos(), 1.0, -3.0 + distance * a.sin());
            for threshold in [0.5, 1.0, 2.0, 4.0] {
                let view = TestView::looking(eye, glam::Vec3::new(0.0, 0.0, -3.0), threshold);
                let (_, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
                for (i, r) in records.iter().enumerate() {
                    let label = format!("eye {eye}, {threshold} px, instance {i}");
                    assert_cut(&label, &pairs, i as u32, &expected_world(&mesh, world * placement_matrix(r), &view, true, 1.0));
                }
                checked += pairs.len();
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}

/// Cards (quads with a disc in their uv) placed like the rocks, alpha-tested and drawn from both
/// sides.
const CARDS_WGSL: &str = r#"
struct Tint { color: vec4<f32> };
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) place: vec4<f32>, @location(4) yaw: f32 };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) uv: vec2<f32> };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let local = turn(v.position.xyz * v.place.w, v.yaw) + v.place.xyz;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(local, 1.0);
    out.normal = (world_matrix * vec4<f32>(turn(v.normal, v.yaw), 0.0)).xyz;
    out.uv = v.uv;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    if (length(in.uv - vec2<f32>(0.5)) > 0.5) {
        discard;
    }
    let n = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0) * tint.color;
    return FOut(n, vec4<f32>(0.0), n, n);
}
"#;

/// Three crowns of cards at `placements` (x, y, z, scale, yaw), with card clusters or without,
/// and a camera at `eye` looking at `target`.
fn crowns(renderer: &Renderer, placements: &[[f32; 5]], clusters: bool, eye: crate::math::Vec3, target: crate::math::Vec3) -> (Scene, Camera, usize) {
    use wgpu::util::DeviceExt;
    let geometry = super::card_tests::crown(800);
    let data: Vec<f32> = placements.iter().flat_map(|p| [p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, 0.0]).collect();
    let source = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE });
    let instances = ComputeBuffer::from_external("Crowns", source.clone(), BufferType::Storage).with_vertex_layout(
        32,
        vec![InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 }, InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32 }],
    );
    let mut material = Material::new("Cards", CARDS_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), cull_mode: crate::materials::CullMode::None, ..Default::default() });
    material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    let count = placements.len() as u32;
    let mut r = Renderable::new(InstancedGeometry::new(geometry, count, vec![instances]), material);
    r.instance_culling = Some(InstanceCulling::new(source, count, 32, 0, 12.0).with_radius_scale(12));
    if clusters {
        let mesh = ClusterMesh::build(&super::card_tests::crown(800), &ClusterOptions { cards: true, ..Default::default() });
        r.clusters = Some(ClusterLod::new(mesh).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None }));
    }
    let mut scene = Scene::new();
    let index = scene.add(SceneNode::Renderable(r));
    let mut camera = Camera::new(50.0, 0.1, 500.0, 1.0);
    camera.set_position(eye.x, eye.y, eye.z);
    camera.look_at(&target);
    camera.update_projection_matrix();
    (scene, camera, index)
}

#[test]
fn card_clusters_draw_through_the_camera_path() {
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let placements = [[-4.0, 0.0, 0.0, 1.0, 0.3], [4.0, 0.0, -2.0, 0.9, 2.0], [0.0, 0.0, -6.0, 1.1, -1.0]];
    let near = (crate::math::Vec3::new(0.0, 6.0, 16.0), crate::math::Vec3::new(0.0, 5.0, -2.0));
    // at zero error, what the mesh draws, both faces of every card
    renderer.set_cluster_error_threshold(0.0);
    let (mut scene, mut camera, _) = crowns(&renderer, &placements, false, near.0, near.1);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (mut scene, mut camera, index) = crowns(&renderer, &placements, true, near.0, near.1);
    let clusters = draw(&mut renderer, &mut scene, &mut camera);
    assert!(cluster_args(&renderer, &scene, index)[1] > 0);
    let (covered, differing) = compare(&mesh, &clusters);
    assert!(covered > (SIZE * SIZE / 10) as usize && differing * 200 < covered, "{differing} of {covered} texels differ");
    // from 60 m at 2 px: far fewer triangles, about as many texels covered
    renderer.set_cluster_error_threshold(2.0);
    let far = (crate::math::Vec3::new(0.0, 6.0, 60.0), crate::math::Vec3::new(0.0, 5.0, -2.0));
    let (mut scene, mut camera, _) = crowns(&renderer, &placements, false, far.0, far.1);
    let mesh = draw(&mut renderer, &mut scene, &mut camera);
    let (mut scene, mut camera, index) = crowns(&renderer, &placements, true, far.0, far.1);
    let pruned = draw(&mut renderer, &mut scene, &mut camera);
    let triangles = cluster_args(&renderer, &scene, index)[6];
    assert!(triangles * 2 <= 3 * 1600, "{triangles} triangles for three crowns of 1600");
    let coverage = |image: &[[f32; 4]]| image.iter().filter(|t| t[3] > 0.5).count() as f32;
    let (full, kept) = (coverage(&mesh), coverage(&pruned));
    assert!(full > 200.0 && (kept / full - 1.0).abs() < 0.15, "coverage {kept} of {full}");
}

#[test]
fn a_negated_yaw_and_a_stretch_bound_the_film_s_trees() {
    // records of the film's trees: a bearing b turns the mesh by -b; the material stretches a
    // tree's width up to 1.1x its height and sways it: a 1.2 margin
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let records = [
        placement_record(glam::Vec3::new(0.0, 0.0, -3.0), 1.5, 0.9, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(4.0, 0.0, -6.0), 2.0, -2.2, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(-4.0, 0.0, -8.0), 1.0, 2.8, glam::Quat::IDENTITY),
    ];
    let record_buffer = buffer(&device, bytemuck::cast_slice(&records.concat()));
    let culling = ClusterCulling::new(&device);
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let source = InstanceSource::All { records: &record_buffer, count: records.len() as u32 };
    let transform = InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: -1.0, rotation: None };
    let params = ClusterCullGpu::new(glam::Mat4::IDENTITY, Some(transform), 48, &source, 3 * mesh.clusters.len() as u32, gpu.vertex_count(), true, 1.2);
    let turned = |r: &[f32; 12]| glam::Mat4::from_translation(glam::Vec3::new(r[0], r[1], r[2])) * glam::Mat4::from_rotation_y(-r[4]) * glam::Mat4::from_scale(glam::Vec3::splat(r[3]));
    for (eye, threshold) in [(glam::Vec3::new(0.0, 2.0, 6.0), 0.5), (glam::Vec3::new(6.0, 1.0, 2.0), 1.0), (glam::Vec3::new(-8.0, 4.0, 10.0), 2.0)] {
        let view = TestView::looking(eye, glam::Vec3::new(0.0, 0.0, -5.0), threshold);
        let (_, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
        // (the stretch loosens bounds and errors; cones are off where a material stretches)
        for (k, r) in records.iter().enumerate() {
            assert_cut(&format!("eye {eye}, instance {k}"), &pairs, k as u32, &expected_world(&mesh, turned(r), &view, false, 1.2));
        }
    }
}

#[test]
fn a_stretch_covers_an_instance_widened_about_its_origin() {
    // a material widening the film's trees 1.1x about their origin moves an off-axis cluster
    // outwards as well as growing it: every level-0 cluster with a widened vertex inside the
    // frustum must still be drawn, wherever the frustum's edges cut the instances
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&spokes(), &ClusterOptions::default());
    let records = [
        placement_record(glam::Vec3::new(0.0, 0.0, -3.0), 1.5, 0.9, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(4.0, 0.0, -6.0), 2.0, -2.2, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(-4.0, 0.0, -8.0), 1.0, 2.8, glam::Quat::IDENTITY),
    ];
    let record_buffer = buffer(&device, bytemuck::cast_slice(&records.concat()));
    let culling = ClusterCulling::new(&device);
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let source = InstanceSource::All { records: &record_buffer, count: records.len() as u32 };
    let transform = InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: -1.0, rotation: None };
    let params = ClusterCullGpu::new(glam::Mat4::IDENTITY, Some(transform), 48, &source, 3 * mesh.clusters.len() as u32, gpu.vertex_count(), true, 1.1);
    let widened = |r: &[f32; 12]| {
        glam::Mat4::from_translation(glam::Vec3::new(r[0], r[1], r[2])) * glam::Mat4::from_rotation_y(-r[4]) * glam::Mat4::from_scale(glam::Vec3::new(1.1 * r[3], r[3], 1.1 * r[3]))
    };
    let mut checked = 0;
    for step in 0..96 {
        // swing the view across the instances so its edges sweep through them
        let a = step as f32 * std::f32::consts::TAU / 96.0;
        let eye = glam::Vec3::new(0.0, 1.0, 4.0);
        let view = TestView::looking(eye, eye + glam::Vec3::new(a.sin(), -0.1, -a.cos()), 0.0);
        let planes = crate::culling::frustum_planes(view.view_proj);
        let (_, pairs) = cull(&device, &queue, &culling, &mut gpu, source, params, &view);
        for (k, r) in records.iter().enumerate() {
            let model = widened(r);
            for (i, c) in mesh.clusters.iter().enumerate().filter(|(_, c)| c.level == 0) {
                let inside = mesh.triangles(i).flatten().any(|v| {
                    let p = model.transform_point3(glam::Vec4::from(mesh.vertices[v as usize].position).truncate());
                    planes.iter().all(|q| q.truncate().dot(p) + q.w > 1e-3)
                });
                if inside {
                    checked += 1;
                    assert!(pairs.contains(&(k as u32, i as u32)), "view {step}, instance {k}: cluster {i} (level {}) has widened geometry in view but was culled", c.level);
                }
            }
        }
    }
    assert!(checked > 100, "only {checked} clusters in view");
}

/// Branches: 8 thin plates reaching from 1 m to 5 m out from the origin, so a cluster reaches
/// along the direction a widening moves it.
fn spokes() -> crate::geometries::Geometry {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for k in 0..8 {
        let a = k as f32 / 8.0 * std::f32::consts::TAU;
        let out = glam::Vec3::new(a.cos(), 0.0, a.sin());
        for j in 0..40 {
            super::card_tests::quad(&mut vertices, &mut indices, out * (1.05 + j as f32 * 0.1), out * 0.05, glam::Vec3::Y * 0.05);
        }
    }
    crate::geometries::Geometry::new("spokes", vertices, indices)
}

#[test]
fn every_view_gets_its_own_cut_in_one_pass() {
    // a frame's views (the camera, a spot light, a cascade) cut in one pass: each cut is its own
    // view's, not the last view written
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mesh = ClusterMesh::build(&rock(4, false), &ClusterOptions::default());
    let records = [
        placement_record(glam::Vec3::new(0.0, 0.0, -3.0), 1.5, 0.9, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(4.0, 0.0, -6.0), 2.0, -2.2, glam::Quat::IDENTITY),
        placement_record(glam::Vec3::new(-4.0, 0.0, -8.0), 1.0, 2.8, glam::Quat::IDENTITY),
    ];
    let record_buffer = buffer(&device, bytemuck::cast_slice(&records.concat()));
    let culling = ClusterCulling::new(&device);
    let mut gpu = ClusterGpu::new(&device, &mesh);
    let source = InstanceSource::All { records: &record_buffer, count: records.len() as u32 };
    let views = [
        TestView::looking(glam::Vec3::new(0.0, 1.0, 2.0), glam::Vec3::new(0.0, 0.0, -5.0), 0.5),
        TestView::looking(glam::Vec3::new(30.0, 10.0, 40.0), glam::Vec3::new(0.0, 0.0, -5.0), 2.0),
        TestView::top_down(glam::Vec3::new(0.0, 50.0, -5.0), 8.0, 1.0),
    ];
    for view in 0..views.len() as u32 {
        let params = ClusterCullGpu::new(glam::Mat4::IDENTITY, Some(PLACEMENT), 48, &source, 3 * mesh.clusters.len() as u32, gpu.vertex_count(), true, 1.0);
        gpu.bind(&device, &queue, &culling, view, source, params);
    }
    culling.set_views(&queue, &views.iter().map(TestView::gpu).collect::<Vec<_>>());
    let mut encoder = device.create_command_encoder(&Default::default());
    culling.encode(&mut encoder, &[(&gpu, 0), (&gpu, 1), (&gpu, 2)]);
    queue.submit(Some(encoder.finish()));
    let mut sizes = Vec::new();
    for (index, view) in views.iter().enumerate() {
        let args = read_words(&device, &queue, gpu.args(index as u32));
        let list = read_words(&device, &queue, gpu.draws(index as u32));
        let pairs: Vec<(u32, u32)> = list.chunks(2).take(args[1] as usize).map(|p| (p[0], p[1])).collect();
        for (k, r) in records.iter().enumerate() {
            assert_cut(&format!("view {index}, instance {k}"), &pairs, k as u32, &expected_world(&mesh, placement_matrix(r), view, true, 1.0));
        }
        sizes.push(pairs.len());
    }
    // the views differ enough that sharing one view's data can't pass
    assert!(sizes[0] != sizes[1] && sizes[1] != sizes[2], "{sizes:?}");
}

/// `rocks` lit by a shadowed spot light 4 m above them and a shadowed sun (two cascades), with
/// the renderer's culling stats on.
fn lit_rocks(renderer: &mut Renderer, clusters: bool) -> (Scene, Camera, usize) {
    use crate::lights::{DirectionalLight, Light, SpotLight};
    use crate::math::Vec3;
    renderer.enable_spot_shadows(256, 1);
    renderer.enable_cascaded_shadows(crate::shadows::CascadedShadowOptions { cascades: 2, resolution: 256, max_distance: 30.0, caster_distance: 30.0, ..Default::default() });
    renderer.set_culling_stats(true);
    let (mut scene, camera, index) = rocks(renderer, &PLACEMENTS, 4, clusters);
    scene.get_renderable_mut(index).unwrap().cast_shadow = true;
    let mut spot = SpotLight::new(Vec3::new(0.5, 4.0, -1.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 10.0, 20.0, 0.6, 0.8);
    spot.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Spot(spot)));
    let mut sun = DirectionalLight::new(Vec3::new(-0.3, -1.0, -0.2), Vec3::new(1.0, 1.0, 1.0), 1.0);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));
    (scene, camera, index)
}

/// Draw a few frames, until the stats of one are read back.
fn stats_after_frames(renderer: &mut Renderer, scene: &mut Scene, camera: &mut Camera) -> crate::culling::CullingStats {
    for _ in 0..6 {
        draw(renderer, scene, camera);
    }
    renderer.culling_stats().cloned().expect("stats read back")
}

#[test]
fn shadow_views_cut_clustered_casters() {
    use crate::culling::CullViewKind;
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (mut scene, mut camera, index) = lit_rocks(&mut renderer, true);
    let stats = stats_after_frames(&mut renderer, &mut scene, &mut camera);
    for kind in [CullViewKind::Camera, CullViewKind::SpotShadow(0), CullViewKind::Cascade(0)] {
        let s = stats.view(kind).unwrap_or_default();
        assert!(s.clusters > 0 && s.triangles > 0, "{kind:?}: {s:?}");
    }
    // every view with a cut of its own
    let gpu = scene.get_renderable(index).unwrap().clusters.as_ref().unwrap().gpu.as_ref().unwrap();
    assert!(gpu.cut(0).is_some() && gpu.cut(1).is_some(), "the camera's and the spot light's cuts");
}

#[test]
fn shadow_triangles_fall_with_the_scale() {
    use crate::culling::CullViewKind;
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (mut scene, mut camera, _) = lit_rocks(&mut renderer, true);
    let mut seen = Vec::new();
    for scale in [1.0, 8.0] {
        renderer.set_shadow_cluster_error_scale(scale);
        let stats = stats_after_frames(&mut renderer, &mut scene, &mut camera);
        seen.push([CullViewKind::Camera, CullViewKind::SpotShadow(0), CullViewKind::Cascade(0)].map(|k| stats.view(k).unwrap_or_default().triangles));
    }
    let ([camera0, spot0, cascade0], [camera1, spot1, cascade1]) = (seen[0], seen[1]);
    assert!(spot1 < spot0 && cascade1 < cascade0, "shadow triangles at scales 1 and 8: {seen:?}");
    assert_eq!(camera0, camera1, "the camera's cut moved with the shadows' scale: {seen:?}");
}

#[test]
fn renderables_off_a_view_get_no_cut_there() {
    use crate::culling::CullViewKind;
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (mut scene, mut camera, index) = lit_rocks(&mut renderer, true);
    scene.get_renderable_mut(index).unwrap().cast_shadow = false;
    let stats = stats_after_frames(&mut renderer, &mut scene, &mut camera);
    let gpu = scene.get_renderable(index).unwrap().clusters.as_ref().unwrap().gpu.as_ref().unwrap();
    // view 1: the spot shadow atlas's layer 0
    assert!(gpu.cut(1).is_none(), "a cut for a view the renderable isn't drawn in");
    assert_eq!(stats.view(CullViewKind::SpotShadow(0)).unwrap_or_default().clusters, 0);
    assert!(stats.camera().clusters > 0);
}

/// Layer `layer` of a Depth32Float array texture, read back.
fn read_depth(renderer: &Renderer, texture: &wgpu::Texture, layer: u32) -> Vec<f32> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::DepthOnly },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    (0..w * h).map(|i| f32::from_le_bytes(bytes[((i / w) * row + (i % w) * 4) as usize..][..4].try_into().unwrap())).collect()
}

/// (texels either map covers, texels whose depths differ by more than 1e-4).
fn compare_depths(a: &[f32], b: &[f32]) -> (usize, usize) {
    let covered = a.iter().zip(b).filter(|(x, y)| **x < 1.0 || **y < 1.0).count();
    let differing = a.iter().zip(b).filter(|(x, y)| (**x - **y).abs() > 1e-4).count();
    (covered, differing)
}

/// The shadow maps a frame of `lit_rocks` leaves (the spot light's layer, then the two
/// cascades), with cluster LOD or without, at `threshold` pixels and the shadows' `scale`.
fn shadow_maps(clusters: bool, threshold: f32, scale: f32) -> Option<Vec<Vec<f32>>> {
    let mut renderer = headless()?;
    renderer.set_cluster_error_threshold(threshold);
    renderer.set_shadow_cluster_error_scale(scale);
    let (mut scene, mut camera, _) = lit_rocks(&mut renderer, clusters);
    for _ in 0..2 {
        draw(&mut renderer, &mut scene, &mut camera);
    }
    let spot = &renderer.spot_shadow_atlas().unwrap().texture;
    let cascades = &renderer.cascaded_shadow_map().unwrap().texture;
    Some(vec![read_depth(&renderer, spot, 0), read_depth(&renderer, cascades, 0), read_depth(&renderer, cascades, 1)])
}

#[test]
fn shadow_maps_draw_the_shadow_cut() {
    // at no error the cut is the mesh, so the maps match; at a coarse shadow budget they are
    // the coarse cut's, not the mesh's
    let Some(mesh) = shadow_maps(false, 0.0, 1.0) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let exact = shadow_maps(true, 0.0, 1.0).unwrap();
    let coarse = shadow_maps(true, 1.0, 1e4).unwrap();
    for (k, name) in ["spot", "cascade 0", "cascade 1"].iter().enumerate() {
        let (covered, differing) = compare_depths(&mesh[k], &exact[k]);
        assert!(covered > 100 && differing * 200 < covered, "{name} at no error: {differing} of {covered} texels differ");
        let (covered, differing) = compare_depths(&mesh[k], &coarse[k]);
        assert!(differing * 20 > covered, "{name} at a coarse shadow budget: only {differing} of {covered} texels differ from the mesh's");
    }
}

/// The sky occlusion's visibility volume once built over `rocks`, with cluster LOD or without,
/// at `threshold` pixels and the top-down view's `scale`.
fn sky_volume(clusters: bool, threshold: f32, scale: f32) -> Option<Vec<u8>> {
    let mut renderer = headless()?;
    renderer.set_cluster_error_threshold(threshold);
    let options = crate::shadows::SkyOcclusionOptions { extent_m: 16.0, resolution: 256, volume_size: (32, 8), min_height_m: -4.0, max_height_m: 8.0, frames: 1, depth_tiles: 1, lod_error_scale: scale, ..Default::default() };
    renderer.enable_sky_occlusion(options);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, clusters);
    scene.get_renderable_mut(index).unwrap().cast_shadow = true;
    for _ in 0..12 {
        draw(&mut renderer, &mut scene, &mut camera);
    }
    let sky = renderer.sky_occlusion().unwrap();
    let texture = sky.volume_texture();
    let (side, levels) = (texture.width(), texture.height());
    let row = (side * 4).div_ceil(256) * 256;
    let (device, queue) = (renderer.device(), renderer.queue());
    let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * levels * side) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(levels) } },
        wgpu::Extent3d { width: side, height: levels, depth_or_array_layers: texture.depth_or_array_layers() },
    );
    queue.submit(Some(encoder.finish()));
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = read.slice(..).get_mapped_range().to_vec();
    Some(data)
}

#[test]
fn sky_occlusion_draws_the_top_down_cut() {
    let Some(mesh) = sky_volume(false, 0.0, 1.0) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let exact = sky_volume(true, 0.0, 1.0).unwrap();
    let coarse = sky_volume(true, 1.0, 1e4).unwrap();
    let occluded = mesh.chunks(4).filter(|v| v[0] < 250).count();
    let differing = |a: &[u8], b: &[u8]| a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x[0].abs_diff(y[0]) > 1).count();
    assert!(occluded > 20, "the rocks occlude only {occluded} voxels");
    assert!(differing(&mesh, &exact) * 50 < occluded, "at no error: {} voxels differ", differing(&mesh, &exact));
    assert!(differing(&mesh, &coarse) > 0, "at a coarse budget the volume is still the mesh's");
}

/// The rocks' material, writing motion vectors (`outputs_velocity`) from last frame's world
/// matrix and view.
const VELOCITY_ROCKS_WGSL: &str = r#"
struct Tint { color: vec4<f32> };
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
struct Temporal { view_proj: mat4x4<f32>, prev_view_proj: mat4x4<f32>, jitter: vec2<f32>, prev_jitter: vec2<f32>, frame: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(1) @binding(3) var<uniform> temporal: Temporal;
struct Transforms { world: mat4x4<f32>, prev_world: mat4x4<f32> };
@group(2) @binding(1) var<uniform> mesh: Transforms;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) yaw: f32 };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) curr: vec4<f32>, @location(2) prev: vec4<f32> };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32>, @location(4) velocity: vec2<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let local = vec4<f32>(turn(v.position.xyz * v.place.w, v.yaw) + v.place.xyz, 1.0);
    out.clip = projection_matrix * view_matrix * mesh.world * local;
    out.normal = (mesh.world * vec4<f32>(turn(v.normal, v.yaw), 0.0)).xyz;
    out.curr = temporal.view_proj * mesh.world * local;
    out.prev = temporal.prev_view_proj * mesh.prev_world * local;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0) * tint.color;
    let velocity = (in.curr.xy / in.curr.w - in.prev.xy / in.prev.w) * vec2<f32>(0.5, -0.5);
    return FOut(n, vec4<f32>(0.0), n, n, velocity);
}
"#;

/// Channels `channels` of a 16-bit float texture, read back.
fn read_half(renderer: &Renderer, texture: &wgpu::Texture, channels: u32) -> Vec<Vec<f32>> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 2 * channels).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    (0..w * h)
        .map(|i| {
            let at = ((i / w) * row + (i % w) * 2 * channels) as usize;
            (0..channels as usize).map(|c| half(u16::from_le_bytes([bytes[at + 2 * c], bytes[at + 2 * c + 1]]))).collect()
        })
        .collect()
}

/// A frame of `rocks` with the velocity material, after one where the rocks stood 0.3 m to the
/// left: the GBuffer's colour (alpha: covered) and velocity.
fn velocity_frame(clusters: bool, threshold: f32) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut renderer = headless()?;
    renderer.set_cluster_error_threshold(threshold);
    let (mut scene, mut camera, index) = rocks(&renderer, &PLACEMENTS, 4, clusters);
    let r = scene.get_renderable_mut(index).unwrap();
    r.material = Material::new("Rocks", VELOCITY_ROCKS_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), outputs_velocity: true, ..Default::default() });
    r.material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    r.object.set_position(-0.3, 0.0, 0.0);
    draw(&mut renderer, &mut scene, &mut camera);
    scene.get_renderable_mut(index).unwrap().object.set_position(0.0, 0.0, 0.0);
    let gbuffer = GBuffer::new(renderer.device(), SIZE, SIZE, 1);
    renderer.render_scene_to_gbuffer(&mut scene, &mut camera, &gbuffer);
    Some((read_half(&renderer, &gbuffer.color_texture, 4), read_half(&renderer, &gbuffer.velocity_texture, 2)))
}

#[test]
fn velocity_follows_the_cut_the_gbuffer_drew() {
    // the velocity pass depth-tests against the GBuffer: drawing the mesh where the GBuffer drew
    // a coarser cut leaves holes; drawing the camera's cut fills every texel it covers
    let Some((color, mesh)) = velocity_frame(false, 0.0) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let covered: Vec<usize> = (0..color.len()).filter(|&i| color[i][3] > 0.5).collect();
    let moving = covered.iter().filter(|&&i| mesh[i][0].abs() > 1e-3 && mesh[i][0] < GBuffer::NO_VELOCITY / 2.0).count();
    assert!(moving * 2 > covered.len() && covered.len() > 500, "the rocks' velocity: {moving} of {} texels", covered.len());
    let (_, exact) = velocity_frame(true, 0.0).unwrap();
    let differing = covered.iter().filter(|&&i| (0..2).any(|c| (mesh[i][c] - exact[i][c]).abs() > 1e-3)).count();
    assert!(differing * 200 < covered.len(), "at no error: {differing} of {} texels' velocities differ", covered.len());
    let (color, coarse) = velocity_frame(true, 4.0).unwrap();
    let covered: Vec<usize> = (0..color.len()).filter(|&i| color[i][3] > 0.5).collect();
    let holes = covered.iter().filter(|&&i| coarse[i][0] >= GBuffer::NO_VELOCITY / 2.0).count();
    assert!(holes * 50 < covered.len(), "at a 4-pixel budget: {holes} of {} covered texels have no velocity", covered.len());
}

/// A rendered planar reflection of `rocks` in the plane y = -1.5, with cluster LOD or without, at
/// `threshold` pixels and the reflection's `scale`: its texture.
fn reflection_image(clusters: bool, threshold: f32, scale: f32) -> Option<Vec<Vec<f32>>> {
    use crate::reflections::{PlanarReflection, PlanarReflectionOptions};
    let mut renderer = headless()?;
    renderer.set_cluster_error_threshold(threshold);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, clusters);
    let mut reflection = PlanarReflection::new(&renderer, crate::math::Vec3::new(0.0, -1.5, 0.0), crate::math::Vec3::new(0.0, 1.0, 0.0), PlanarReflectionOptions { width: 160, height: 160, mip_levels: 1, ..Default::default() });
    reflection.lod_error_scale = scale;
    renderer.add_planar_reflection(reflection);
    for _ in 0..2 {
        draw(&mut renderer, &mut scene, &mut camera);
    }
    Some(read_half(&renderer, renderer.planar_reflection(0).unwrap().texture(), 4))
}

#[test]
fn planar_reflections_draw_their_cut() {
    let Some(mesh) = reflection_image(false, 0.0, 1.0) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let lit = |t: &Vec<f32>| t[..3].iter().any(|c| *c > 0.02);
    let covered = mesh.iter().filter(|t| lit(t)).count();
    let differing = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).filter(|(x, y)| (0..3).any(|c| (x[c] - y[c]).abs() > 2.0 / 255.0)).count();
    let exact = reflection_image(true, 0.0, 1.0).unwrap();
    assert!(covered > 200, "the reflection shows {covered} texels of rock");
    assert!(differing(&mesh, &exact) * 200 < covered, "at no error: {} of {covered} texels differ", differing(&mesh, &exact));
    let coarse = reflection_image(true, 1.0, 1e4).unwrap();
    assert!(differing(&mesh, &coarse) * 20 > covered, "at a coarse budget only {} of {covered} texels differ from the mesh's", differing(&mesh, &coarse));
}

#[test]
fn every_clustered_renderable_finds_its_cuts_whatever_the_draw_order() {
    // transparent renderables come after the opaque ones, back to front: the passes still find
    // each renderable's cut for each view
    let Some(mut renderer) = headless() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (mut scene, mut camera, glass) = lit_rocks(&mut renderer, true);
    {
        let r = scene.get_renderable_mut(glass).unwrap();
        r.material = Material::new("Glass", ROCKS_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), transparent: true, ..Default::default() });
        r.material.set_uniform_bindable(0, "Tint", &[[1.0f32; 4]]);
    }
    let solid = scene.add(SceneNode::Renderable(rock_renderable(&renderer, &PLACEMENTS, 4, true)));
    scene.get_renderable_mut(solid).unwrap().cast_shadow = true;
    assert!(glass < solid);
    draw(&mut renderer, &mut scene, &mut camera);
    for index in [glass, solid] {
        for view in [0, 1] {
            assert!(renderer.has_cluster_cut(&scene, index, view), "renderable {index}: no cut found for view {view}");
        }
    }
}

/// `reflection_image`'s, seen from 2 m above the water, looking 1° down at the rocks: the usual
/// water shot.
fn level_reflection_image(clusters: bool, threshold: f32) -> Option<Vec<Vec<f32>>> {
    use crate::reflections::{PlanarReflection, PlanarReflectionOptions};
    let mut renderer = headless()?;
    renderer.set_cluster_error_threshold(threshold);
    let (mut scene, mut camera, _) = rocks(&renderer, &PLACEMENTS, 4, clusters);
    camera.set_position(0.0, 0.5, 8.0);
    camera.look_at(&crate::math::Vec3::new(0.0, 0.5 - 100.0 * 1f32.to_radians().tan(), -92.0));
    renderer.add_planar_reflection(PlanarReflection::new(&renderer, crate::math::Vec3::new(0.0, -1.5, 0.0), crate::math::Vec3::new(0.0, 1.0, 0.0), PlanarReflectionOptions { width: 192, height: 192, mip_levels: 1, ..Default::default() }));
    for _ in 0..2 {
        draw(&mut renderer, &mut scene, &mut camera);
    }
    Some(read_half(&renderer, renderer.planar_reflection(0).unwrap().texture(), 4))
}

#[test]
fn a_level_reflection_keeps_the_budget_near_the_water() {
    // the mirrored view's projection has its near plane on the water, far out along a level
    // view: errors must still be measured from the camera's near, or the reflection's cut is
    // far coarser than a pixel
    let Some(mesh) = level_reflection_image(false, 0.0) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let clusters = level_reflection_image(true, 1.0).unwrap();
    let covered = mesh.iter().filter(|t| t[..3].iter().any(|c| *c > 0.02)).count();
    let differing = mesh.iter().zip(&clusters).filter(|(x, y)| (0..3).any(|c| (x[c] - y[c]).abs() > 0.1)).count();
    assert!(covered > 200, "the reflection shows {covered} texels of rock");
    assert!(differing * 20 < covered, "at a 1-pixel budget {differing} of {covered} texels differ clearly from the mesh's");
}
