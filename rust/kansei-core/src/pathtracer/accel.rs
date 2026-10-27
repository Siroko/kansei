//! How the path tracer's passes trace its scene: the BVH4s in compute (`traversal.wgsl`, the web's
//! only one), or on native, where the device has hardware ray queries, an acceleration structure
//! of the same triangles (`traversal_query.wgsl`), picked by the RT grid's rules
//! ([`RtBackend::resolve`]: `Auto`, `KANSEI_RT_BACKEND`). Both give the same `HitInfo` for the
//! same triangle.
//!
//! The acceleration structure maps the scene as the BVHs do: a BLAS a mesh (`BLASEntry`), its
//! primitive i the mesh's triangle i in the builder's (reordered) order, and a TLAS instance a
//! scene instance (`PackedInstance`), in order, its custom data the instance's index. The BLASes
//! are built once; moving instances rebuilds the TLAS (`BVHBuilder::refresh_transforms`).
//!
//! Metal orders nothing through a TLAS's instances (wgpu 30 has no acceleration structure barrier
//! there, #9215; see `rt/query.rs`): the CPU waits for the BLASes' submit before building the TLAS.
//! The trace binds the TLAS itself, which Metal does order after its build.

use wgpu::util::DeviceExt;

use super::buffers::{PackedInstance, PackedTriangle};
use super::bvh_builder::BLASEntry;
use crate::rt::RtBackend;

const INTERSECTION_WGSL: &str = include_str!("shaders/intersection.wgsl");
const INSTANCES_WGSL: &str = include_str!("shaders/instances.wgsl");
const TRAVERSAL_WGSL: &str = include_str!("shaders/traversal.wgsl");
const TRAVERSAL_QUERY_WGSL: &str = include_str!("shaders/traversal_query.wgsl");

/// The scene in hardware acceleration structures.
pub(crate) struct SceneAccel {
    /// the BLASes' vertices: each triangle's three positions, in the builder's order
    _vertices: wgpu::Buffer,
    /// an entry's BLAS (none for an empty one)
    blas: Vec<Option<wgpu::Blas>>,
    tlas: wgpu::Tlas,
}

impl SceneAccel {
    /// Whether `device`'s acceleration structure limits take a scene of BLASes of up to `largest`
    /// triangles and `instances` instances.
    pub(crate) fn fits(device: &wgpu::Device, largest: usize, instances: usize) -> bool {
        let limits = device.limits();
        largest as u64 <= limits.max_blas_primitive_count as u64 && instances as u64 <= limits.max_tlas_instance_count as u64
    }

    /// Build the BLASes of `entries` (their triangles in `triangles`) and the TLAS of `instances`
    /// (instance i placing entry i), each in a submit of its own.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, triangles: &[PackedTriangle], entries: &[BLASEntry], instances: &[PackedInstance]) -> Self {
        let positions: Vec<[f32; 3]> = triangles.iter().flat_map(|t| [t.v0, t.v1, t.v2]).collect();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PathTracer/BlasVertices"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let sizes: Vec<wgpu::BlasTriangleGeometrySizeDescriptor> = entries
            .iter()
            .map(|e| wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: e.tri_count as u32 * 3,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            })
            .collect();
        let blas: Vec<Option<wgpu::Blas>> = sizes
            .iter()
            .map(|size| {
                (size.vertex_count > 0).then(|| device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("PathTracer/Blas"),
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
                ))
            })
            .collect();
        let builds: Vec<wgpu::BlasBuildEntry> = blas
            .iter()
            .zip(entries.iter().zip(&sizes))
            .filter_map(|(blas, (entry, size))| Some((blas.as_ref()?, (entry, size))))
            .map(|(blas, (entry, size))| wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: &vertices,
                    first_vertex: entry.tri_offset as u32 * 3,
                    vertex_stride: 12,
                    index_buffer: None,
                    first_index: None,
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            })
            .collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("PathTracer/BlasBuild") });
        encoder.build_acceleration_structures(builds.iter(), std::iter::empty());
        let submit = queue.submit(Some(encoder.finish()));
        if device.adapter_info().backend == wgpu::Backend::Metal {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(submit), timeout: None });
        }
        let mut accel = Self { _vertices: vertices, blas, tlas: Self::create_tlas(device, instances.len()) };
        accel.build_tlas(device, queue, instances);
        accel
    }

    fn create_tlas(device: &wgpu::Device, instances: usize) -> wgpu::Tlas {
        device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("PathTracer/Tlas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: instances.max(1) as u32,
        })
    }

    /// Rebuild the TLAS with `instances`' transforms (the same instances, moved).
    pub(crate) fn set_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, instances: &[PackedInstance]) {
        if instances.len() > self.tlas.get().len() {
            self.tlas = Self::create_tlas(device, instances.len());
        }
        self.build_tlas(device, queue, instances);
    }

    fn build_tlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, instances: &[PackedInstance]) {
        for (i, inst) in instances.iter().enumerate() {
            let t = inst.transform;
            let transform = [t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1], t[1][2], t[1][3], t[2][0], t[2][1], t[2][2], t[2][3]];
            self.tlas[i] = self.blas.get(i).and_then(Option::as_ref).map(|blas| wgpu::TlasInstance::new(blas, transform, i as u32, 0xff));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("PathTracer/TlasBuild") });
        encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&self.tlas));
        queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn tlas(&self) -> &wgpu::Tlas {
        &self.tlas
    }
}

/// WGSL tracing the path tracer's scene through `backend` (`Software` or `Hardware`): `Ray`,
/// `HitInfo`, `Instance`, `trace_bvh(ray)`, `trace_bvh_shadow(ray, max_dist)`,
/// `trace_bvh_internal(ray, any_hit, max_dist)` and `get_world_tri_area(tri, instance)`, with its
/// bindings in group 0 from `first` (four: `triangles`; the BLAS nodes, unused by the hardware;
/// the TLAS nodes or the acceleration structure; `instances`). With `count_rays`, binding
/// `count_rays` holds an `atomic<u32>` that each traced ray increments. A module with the
/// hardware's must pass through `rt::hoist_wgsl_enables`.
pub(crate) fn traversal_wgsl(backend: RtBackend, first: u32, count_rays: Option<u32>) -> String {
    let (traversal, accel) = match backend {
        RtBackend::Hardware => (TRAVERSAL_QUERY_WGSL, format!("@group(0) @binding({}) var scene_tlas : acceleration_structure;\n", first + 2)),
        _ => (
            TRAVERSAL_WGSL,
            format!(
                "@group(0) @binding({}) var<storage, read> bvh4_nodes      : array<vec4f>;\n@group(0) @binding({}) var<storage, read> tlas_bvh4_nodes : array<vec4f>;\n",
                first + 1,
                first + 2
            ),
        ),
    };
    let count = match count_rays {
        Some(binding) => format!("@group(0) @binding({binding}) var<storage, read_write> pt_ray_count : atomic<u32>;\nfn pt_count_ray() {{ atomicAdd(&pt_ray_count, 1u); }}\n"),
        None => "fn pt_count_ray() {}\n".to_string(),
    };
    format!(
        "{INTERSECTION_WGSL}\n{INSTANCES_WGSL}\n{traversal}\n\
         // ── the scene's bindings ──\n\
         @group(0) @binding({first}) var<storage, read> triangles : array<f32>;\n\
         {accel}\
         @group(0) @binding({}) var<storage, read> instances : array<Instance>;\n{count}",
        first + 3
    )
}

/// The layout entries for `traversal_wgsl(backend, first, _)`'s bindings (without the counter).
pub(crate) fn traversal_layout_entries(backend: RtBackend, first: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    match backend {
        RtBackend::Hardware => vec![
            storage(first),
            wgpu::BindGroupLayoutEntry { binding: first + 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::AccelerationStructure { vertex_return: false }, count: None },
            storage(first + 3),
        ],
        _ => vec![storage(first), storage(first + 1), storage(first + 2), storage(first + 3)],
    }
}

/// A counter's layout entry (`traversal_wgsl`'s `count_rays`).
pub(crate) fn counter_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// A module with `traversal_wgsl` in it, ready to compile (its `enable` first).
pub(crate) fn assemble(code: String) -> String {
    crate::rt::hoist_wgsl_enables(&code)
}

/// The compute module of `code` (assembled, with `traversal_wgsl(backend, ..)` its only ray
/// queries). On the hardware backend it is made without wgpu's ray query initialization
/// tracking, which cost ten times the path tracer's trace on NVIDIA's Vulkan driver (1080p, 1 spp:
/// 9.3 against 0.95 ms on an RTX 5090); `traversal_query.wgsl` keeps the rules it checks.
pub(crate) fn shader_module(device: &wgpu::Device, backend: RtBackend, label: &str, code: &str) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) };
    match backend {
        // SAFETY: the module's one ray query (traversal_query.wgsl's `trace_bvh_internal`) is
        // initialized before it proceeds, proceeds until it returns false before its committed
        // intersection is read, reads no candidate, and is given no ray with a NaN or an infinity
        // in its origin or direction or a t_max not above its t_min (those return a miss first).
        RtBackend::Hardware => unsafe {
            device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks { ray_query_initialization_tracking: false, ..wgpu::ShaderRuntimeChecks::checked() })
        },
        _ => device.create_shader_module(desc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validated with naga for `backend`: ray queries for the hardware one.
    fn validate(backend: RtBackend, name: &str, code: &str) -> naga::Module {
        let name = format!("{name} ({backend:?})");
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        let mut capabilities = naga::valid::Capabilities::empty();
        if backend == RtBackend::Hardware {
            capabilities |= naga::valid::Capabilities::RAY_QUERY;
        }
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        module
    }

    fn struct_span(module: &naga::Module, name: &str) -> usize {
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
    fn shaders_validate_pathtracer() {
        for backend in [RtBackend::Software, RtBackend::Hardware] {
            for count_rays in [false, true] {
                let trace = validate(backend, &format!("trace (count rays {count_rays})"), &super::super::path_tracer::trace_wgsl(backend, count_rays));
                assert_eq!(struct_span(&trace, "TraceParams"), 192);
                assert_eq!(struct_span(&trace, "Instance"), std::mem::size_of::<PackedInstance>());
            }
            validate(backend, "probe trace", &super::super::probes::trace_wgsl(backend));
            validate(backend, "restir", &super::super::restir::restir_wgsl(backend));
        }
        // the web's stays core WebGPU
        assert!(!super::super::path_tracer::trace_wgsl(RtBackend::Software, false).contains("enable "));
    }

    /// The layout entries name the bindings the WGSL declares.
    #[test]
    fn traversal_bindings_match_their_layout() {
        for backend in [RtBackend::Software, RtBackend::Hardware] {
            let code = traversal_wgsl(backend, 5, None);
            for entry in traversal_layout_entries(backend, 5) {
                assert!(code.contains(&format!("@binding({})", entry.binding)), "{backend:?}: binding {} undeclared", entry.binding);
            }
            assert_eq!(code.matches("@binding(").count(), traversal_layout_entries(backend, 5).len(), "{backend:?}");
        }
    }
}
