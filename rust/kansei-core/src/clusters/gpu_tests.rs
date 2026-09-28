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
