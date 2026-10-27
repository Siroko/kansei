//! The hardware backend of `RtGrid`: its gathered triangles in an acceleration structure for
//! hardware ray queries (wgpu's `EXPERIMENTAL_RAY_QUERY`, native only), traced by `RT_QUERY_WGSL`'s
//! `kansei_rt_trace` in place of the grid's cells.
//!
//! One BLAS holds the triangles by id (the gather's own ids, so every `kansei_rt_*(id)` accessor
//! reads them as it does the grid's): an opaque geometry, and when alpha-tested or glass sources
//! were gathered a second, non-opaque one (only its candidates come back to the shader). Sizes are
//! fixed on the CPU, so both hold the grid's triangle capacity, the slots nothing fills holding
//! filler triangles outside the box (Metal's builds lost the real triangles among many inactive
//! ones). One TLAS instance holds the BLAS. A rebuild builds the BLAS in the build's encoder, and
//! the TLAS in a submit of its own before the first trace (`RtGridHandle::prepare_trace`).
//!
//! Metal needs more: wgpu 30's Metal backend has no acceleration structure barrier (#9215, fix open
//! as #9645), and Metal tracks no hazard through a TLAS's instances, so nothing orders a TLAS build
//! after the BLAS build it reads, even in a later submit (rays then missed most of the scene). The
//! CPU waits for the BLAS's submit before the TLAS build. And a rebuild must not overwrite the BLAS
//! the frames before are still tracing through their TLAS: Metal alternates two BLASes, the wait
//! for one's build also seeing the other's traces done. (The trace binds the TLAS itself, which
//! Metal does order after its build.)

use std::sync::{Arc, Mutex};

use super::RtBackend;

pub(crate) const BUILD_WGSL: &str = concat!(include_str!("shaders/rt_types.wgsl"), include_str!("shaders/rt_query_build.wgsl"));
/// Bytes of a triangle's three vertices in a BLAS vertex buffer.
const TRIANGLE_VERTEX_BYTES: u64 = 36;
/// Workgroups along x before the positions pass wraps to y.
const GROUPS_X: u32 = 32768;

/// The WGSL `KanseiRtQueryParams` (rt_query_build.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct QueryParamsGpu {
    filler: [f32; 3],
    capacity: u32,
    split: u32,
    _pad: [u32; 3],
}

/// The BLAS's shape: what it was made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    capacity: u32,
    split: bool,
    fast_build: bool,
}

pub(crate) struct RtQuery {
    params: wgpu::Buffer,
    /// the opaque and the non-opaque geometry's vertices (a placeholder without the split)
    positions: [wgpu::Buffer; 2],
    /// one BLAS, or on Metal two built in turn (`next` the next), with what each was made for
    blas: Vec<Option<(Shape, wgpu::Blas)>>,
    next: usize,
    tlas: Arc<Mutex<TlasState>>,
    /// a BLAS built since the TLAS
    built: bool,
    layout: wgpu::BindGroupLayout,
    positions_pipeline: wgpu::ComputePipeline,
    mark: wgpu::ComputePipeline,
    group: Option<wgpu::BindGroup>,
    written: Option<QueryParamsGpu>,
}

/// The TLAS, shared with the grid's handle, which builds it before the first trace after a BLAS
/// build (`build_tlas`).
pub(crate) struct TlasState {
    tlas: wgpu::Tlas,
    /// the BLAS's submit (or a later one) when the TLAS is to be built over it
    pending: Option<wgpu::SubmissionIndex>,
    /// wait for that submit first (Metal, #9215)
    wait: bool,
}

impl RtQuery {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RtQuery/Positions"),
            entries: &[buffer(0, ro), buffer(1, ro), buffer(2, rw), buffer(3, wgpu::BufferBindingType::Uniform), buffer(4, rw)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("RtQuery/Positions"), source: wgpu::ShaderSource::Wgsl(BUILD_WGSL.into()) });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("RtQuery/Positions"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |entry_point: &str, layout: Option<&wgpu::PipelineLayout>| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("RtQuery/{entry_point}")),
                layout,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("RtQuery/Mark"), bind_group_layouts: &[], immediate_size: 0 });
        let metal = device.adapter_info().backend == wgpu::Backend::Metal;
        Self {
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtQuery/Params"),
                size: std::mem::size_of::<QueryParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            positions: [0, 1].map(|_| device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtQuery/Positions"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })),
            blas: vec![None; if metal { 2 } else { 1 }],
            next: 0,
            tlas: Arc::new(Mutex::new(TlasState {
                tlas: device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("RtQuery/Tlas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    max_instances: 1,
                }),
                pending: None,
                wait: metal,
            })),
            built: false,
            positions_pipeline: pipeline("positions_main", Some(&pipeline_layout)),
            mark: pipeline("mark_main", Some(&empty_layout)),
            layout,
            group: None,
            written: None,
        }
    }

    /// The TLAS (the trace binds it), shared with the grid's handle.
    pub(crate) fn tlas(&self) -> Arc<Mutex<TlasState>> {
        self.tlas.clone()
    }

    /// Bytes on the GPU (the vertex buffers; the acceleration structures' own sizes aren't exposed).
    pub(crate) fn memory_bytes(&self) -> u64 {
        self.positions.iter().map(wgpu::Buffer::size).sum()
    }

    /// Ready the BLAS for a build of the grid's `triangles` (`capacity` of them; `counters`' first
    /// word how many were gathered), between the grid's `begin` and `finish`: made anew when its
    /// shape changed, its vertex buffers with it. `split`: alpha-tested or glass sources are
    /// gathered; `fast_build`: the grid rebuilds every frame; `below`: a point under the grid's
    /// box, where the filler goes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, triangles: &wgpu::Buffer, counters: &wgpu::Buffer, capacity: u32, split: bool, fast_build: bool, below: [f32; 3]) {
        let shape = Shape { capacity, split, fast_build };
        let slot = &mut self.blas[self.next];
        if slot.as_ref().is_none_or(|(s, _)| *s != shape) {
            let flags = if fast_build { wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD } else { wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE };
            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor { label: Some("RtQuery/Blas"), flags, update_mode: wgpu::AccelerationStructureUpdateMode::Build },
                wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: Self::sizes(shape) },
            );
            *slot = Some((shape, blas));
        }
        let blas = &slot.as_ref().unwrap().1;
        self.tlas.lock().unwrap().tlas[0] = Some(wgpu::TlasInstance::new(blas, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0, 0xff));
        if self.written.is_none_or(|w| w.capacity != capacity || (w.split != 0) != split) {
            for k in 0..2 {
                let bytes = if k == 0 || split { capacity as u64 * TRIANGLE_VERTEX_BYTES } else { 16 };
                if self.positions[k].size() != bytes {
                    self.positions[k] = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("RtQuery/Positions"),
                        size: bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
                        mapped_at_creation: false,
                    });
                }
            }
            self.group = None;
        }
        let params = QueryParamsGpu { filler: below, capacity, split: split as u32, _pad: [0; 3] };
        if self.written != Some(params) {
            queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
            self.written = Some(params);
        }
        // (the grid makes its triangles and counters anew only when its capacity changes, and the
        // vertex buffers with them)
        if self.group.is_none() {
            fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
            }
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtQuery/Positions"),
                layout: &self.layout,
                entries: &[entry(0, triangles), entry(1, counters), entry(2, &self.positions[0]), entry(3, &self.params), entry(4, &self.positions[1])],
            }));
        }
    }

    /// Record the BLAS's build after the gather, in its encoder: the vertices, then the BLAS. The
    /// TLAS waits for this encoder's submit (`submitted`), then for `build_tlas`.
    pub(crate) fn record_blas(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let (Some((shape, blas)), Some(group)) = (self.blas[self.next].as_ref(), self.group.as_ref()) else { panic!("RtQuery::prepare first") };
        let stamp = crate::profiling::gpu_pass("Rt/BlasInput");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Rt/BlasInput"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&self.positions_pipeline);
            pass.set_bind_group(0, group, &[]);
            let groups = shape.capacity.div_ceil(64).max(1);
            pass.dispatch_workgroups(groups.min(GROUPS_X), groups.div_ceil(GROUPS_X), 1);
        }
        let sizes = Self::sizes(*shape);
        let geometries = sizes
            .iter()
            .zip(&self.positions)
            .map(|(size, positions)| wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer: positions,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            })
            .collect();
        encoder.build_acceleration_structures(std::iter::once(&wgpu::BlasBuildEntry { blas, geometry: wgpu::BlasGeometries::TriangleGeometries(geometries) }), std::iter::empty());
        self.built = true;
        self.next = (self.next + 1) % self.blas.len();
    }

    /// After the submit holding the BLAS's build: the TLAS is to be built over it.
    pub(crate) fn submitted(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if std::mem::take(&mut self.built) {
            // a submit after the BLAS's to wait for (not an empty one: waiting on that returned
            // before the BLAS was built)
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RtQuery/Built") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtQuery/Built"), timestamp_writes: None });
                pass.set_pipeline(&self.mark);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.tlas.lock().unwrap().pending = Some(queue.submit(Some(encoder.finish())));
        }
    }

    /// The BLAS's geometries: the opaque one, and the non-opaque one with the split.
    fn sizes(shape: Shape) -> Vec<wgpu::BlasTriangleGeometrySizeDescriptor> {
        let size = |flags| wgpu::BlasTriangleGeometrySizeDescriptor { vertex_format: wgpu::VertexFormat::Float32x3, vertex_count: shape.capacity * 3, index_format: None, index_count: None, flags };
        let mut sizes = vec![size(wgpu::AccelerationStructureGeometryFlags::OPAQUE)];
        if shape.split {
            sizes.push(size(wgpu::AccelerationStructureGeometryFlags::empty()));
        }
        sizes
    }
}

impl TlasState {
    pub(crate) fn tlas(&self) -> &wgpu::Tlas {
        &self.tlas
    }

    /// Build the TLAS over the BLAS built last, if one was, in a submit of its own after the
    /// BLAS's: on Metal once that has finished (#9215).
    pub(crate) fn build(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(blas_submit) = self.pending.take() else { return };
        if self.wait {
            let _t = crate::profiling::cpu_scope("rt/tlas wait");
            let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(blas_submit), timeout: None });
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Rt/TlasBuild") });
        encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&self.tlas));
        queue.submit(Some(encoder.finish()));
    }
}

/// Whether `backend` traces with hardware ray queries on `device`, for `RtBackend::Auto` (which
/// `KANSEI_RT_BACKEND` overrides on native: `software`, `hardware` or `auto`): where the device has
/// them, on Vulkan, and on Metal on Apple GPUs with ray tracing hardware (M3, A17 and later; M1 and
/// M2 report ray queries without it). Not DX12: NVIDIA's driver 617.42 miscompiles a ray query
/// there (the scout's coherent primary rays traced with another branch's flags); ask for
/// `Hardware` to use it anyway.
pub(crate) fn resolve(backend: RtBackend, device: &wgpu::Device) -> RtBackend {
    let backend = match backend {
        RtBackend::Auto => env_backend().unwrap_or(RtBackend::Auto),
        b => b,
    };
    let has_queries = device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    match backend {
        RtBackend::Software => RtBackend::Software,
        RtBackend::Hardware if has_queries => RtBackend::Hardware,
        RtBackend::Hardware => {
            log::warn!("rt grid: hardware ray queries asked for, but the device has none (EXPERIMENTAL_RAY_QUERY): tracing in software");
            RtBackend::Software
        }
        RtBackend::Auto => {
            let info = device.adapter_info();
            let trusted = match info.backend {
                wgpu::Backend::Vulkan => true,
                wgpu::Backend::Metal => apple_ray_tracing_hardware(&info.name),
                _ => false,
            };
            if has_queries && trusted {
                RtBackend::Hardware
            } else {
                RtBackend::Software
            }
        }
    }
}

/// `KANSEI_RT_BACKEND`, on native.
fn env_backend() -> Option<RtBackend> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let value = std::env::var("KANSEI_RT_BACKEND").ok()?;
        match value.to_ascii_lowercase().as_str() {
            "software" | "sw" => Some(RtBackend::Software),
            "hardware" | "hw" => Some(RtBackend::Hardware),
            "auto" | "" => None,
            other => {
                log::warn!("KANSEI_RT_BACKEND={other}: expected software, hardware or auto");
                None
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    None
}

/// Whether an Apple GPU, by its Metal name ("Apple M4 Pro", "Apple A17 Pro GPU"), has ray
/// tracing hardware: Apple family 9 and later, M3 and A17 on.
pub(crate) fn apple_ray_tracing_hardware(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("Apple ") else { return false };
    let mut chars = rest.chars();
    let (first, generation): (u32, u32) = match chars.next() {
        Some('M') => (3, chars.take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)),
        Some('A') => (17, chars.take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)),
        _ => return false,
    };
    generation >= first
}

#[cfg(test)]
mod tests {
    use super::apple_ray_tracing_hardware;

    #[test]
    fn apple_gpus_from_m3_and_a17_have_ray_tracing_hardware() {
        for (name, has) in [
            ("Apple M1", false),
            ("Apple M1 Max", false),
            ("Apple M2 Ultra", false),
            ("Apple M3", true),
            ("Apple M4 Pro", true),
            ("Apple M5", true),
            ("Apple A16 GPU", false),
            ("Apple A17 Pro GPU", true),
            ("Apple A18", true),
            ("Apple Paravirtual device", false),
            ("AMD Radeon Pro 5500M", false),
        ] {
            assert_eq!(apple_ray_tracing_hardware(name), has, "{name}");
        }
    }
}
