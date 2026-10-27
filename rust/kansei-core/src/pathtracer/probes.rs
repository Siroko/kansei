use crate::renderers::Renderer;
use bytemuck::{Pod, Zeroable};

use super::bvh_builder::GPUBVHData;
use crate::rt::RtBackend;

/// Uniform parameters for the probe trace shader.
///
/// Layout matches the WGSL `ProbeTraceParams` struct:
///   vec3f grid_min       (12 bytes)
///   f32   grid_step_x    (4 bytes)
///   vec3u grid_dims      (12 bytes)
///   f32   grid_step_y    (4 bytes)
///   f32   grid_step_z    (4 bytes)
///   u32   rays_per_probe (4 bytes)
///   u32   frame_index    (4 bytes)
///   u32   light_count    (4 bytes)
///   f32   max_distance   (4 bytes)
///   f32   _pad0          (4 bytes)
///   f32   _pad1          (4 bytes)
///   f32   _pad2          (4 bytes)
/// Total: 64 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ProbeTraceParams {
    grid_min: [f32; 3],
    grid_step_x: f32,
    grid_dims: [u32; 3],
    grid_step_y: f32,
    grid_step_z: f32,
    rays_per_probe: u32,
    frame_index: u32,
    light_count: u32,
    max_distance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const TRACE_PARAMS_SIZE: u64 = std::mem::size_of::<ProbeTraceParams>() as u64;

/// Uniform parameters for the probe SH update shader.
///
/// Layout matches the WGSL `ProbeUpdateParams` struct:
///   u32 rays_per_probe (4 bytes)
///   u32 total_probes   (4 bytes)
///   f32 hysteresis     (4 bytes)
///   u32 frame_index    (4 bytes)
/// Total: 16 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ProbeUpdateParams {
    rays_per_probe: u32,
    total_probes: u32,
    hysteresis: f32,
    frame_index: u32,
}

const UPDATE_PARAMS_SIZE: u64 = std::mem::size_of::<ProbeUpdateParams>() as u64;

/// SH data per probe: 9 coefficients * vec4f (16 bytes) = 144 bytes.
const SH_BYTES_PER_PROBE: u64 = 9 * 16;

/// Ray result data per ray: 2 vec4f (32 bytes) = [radiance+dist, dir+pad].
const RAY_RESULT_BYTES: u64 = 2 * 16;

/// The probe trace shader for the scene's `backend`.
pub(crate) fn trace_wgsl(backend: RtBackend) -> String {
    super::accel::assemble(format!("{}\n{}", super::accel::traversal_wgsl(backend, 1, None), include_str!("shaders/probe-trace.wgsl")))
}

/// The probe trace's layout and pipeline for the scene's `backend`. Bindings:
///  0: params (uniform)
///  1-4: the scene (`GPUBVHData::layout_entries`: triangles, BLAS nodes, TLAS nodes or the
///       acceleration structure, instances)
///  5: materials (storage, read)
///  6: scene_lights (storage, read)
///  7: ray_results (storage, read_write)
///  8: probe_sh_prev (storage, read)
fn trace_pipeline(device: &wgpu::Device, backend: RtBackend) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let trace_shader = super::accel::shader_module(device, backend, "ProbeGrid/TraceShader", &trace_wgsl(backend));
    let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(TRACE_PARAMS_SIZE),
            },
            count: None,
        },
        buffer(5, read),
        buffer(6, read),
        buffer(7, wgpu::BufferBindingType::Storage { read_only: false }),
        buffer(8, read),
    ];
    entries.extend(super::accel::traversal_layout_entries(backend, 1));
    let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ProbeGrid/TraceBGL"), entries: &entries });

    let trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ProbeGrid/TracePL"),
        bind_group_layouts: &[Some(&trace_bgl)],
        immediate_size: 0,
    });

    let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ProbeGrid/TracePipeline"),
        layout: Some(&trace_pl),
        module: &trace_shader,
        entry_point: Some("probe_trace_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (trace_bgl, trace_pipeline)
}

/// Irradiance probe grid for multi-bounce global illumination.
///
/// A uniform 3D grid of probes that each store L2 spherical harmonics
/// (9 RGB coefficients). Each frame:
///
///   1. **Trace** — fires `rays_per_frame` rays per probe using spherical
///      Fibonacci sampling (rotated per frame). Hits evaluate direct
///      lighting + indirect from previous-frame SH (multi-bounce).
///   2. **Update** — projects ray results onto SH basis and blends with
///      history via temporal hysteresis.
///
/// SH buffers are double-buffered (ping-pong) so the trace shader can
/// read the previous frame's SH while the update shader writes the new.
pub struct ProbeGrid {
    /// the trace's layout and pipeline for each backend traced so far
    traces: Vec<(RtBackend, wgpu::BindGroupLayout, wgpu::ComputePipeline)>,
    update_pipeline: wgpu::ComputePipeline,
    update_bgl: wgpu::BindGroupLayout,
    sh_buf_a: Option<wgpu::Buffer>,
    sh_buf_b: Option<wgpu::Buffer>,
    ray_results_buf: Option<wgpu::Buffer>,
    trace_params_buf: wgpu::Buffer,
    update_params_buf: wgpu::Buffer,
    /// Grid dimensions (number of probes along each axis).
    pub grid_dims: [u32; 3],
    /// World-space minimum corner of the probe grid.
    pub grid_min: [f32; 3],
    /// World-space step size along each axis.
    pub grid_step: [f32; 3],
    /// Total number of probes (product of grid_dims).
    pub probe_count: u32,
    /// Number of rays traced per probe per frame (default 64).
    pub rays_per_frame: u32,
    /// Temporal hysteresis blend factor (default 0.95). Higher values
    /// give smoother but slower-converging results.
    pub hysteresis: f32,
    ping: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl ProbeGrid {
    /// Create the probe grid pipelines and parameter buffers.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        // ── Update pipeline ──────────────────────────────────────────
        let update_src = include_str!("shaders/probe-update.wgsl");
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ProbeGrid/UpdateShader"),
            source: wgpu::ShaderSource::Wgsl(update_src.into()),
        });

        // Update BGL: bindings 0-3
        //  0: params (uniform)
        //  1: ray_results (storage, read)
        //  2: sh_history (storage, read)
        //  3: sh_output (storage, read_write)
        let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ProbeGrid/UpdateBGL"),
            entries: &[
                // 0: params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UPDATE_PARAMS_SIZE),
                    },
                    count: None,
                },
                // 1: ray_results
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: sh_history
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: sh_output
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ProbeGrid/UpdatePL"),
            bind_group_layouts: &[Some(&update_bgl)],
            immediate_size: 0,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ProbeGrid/UpdatePipeline"),
            layout: Some(&update_pl),
            module: &update_shader,
            entry_point: Some("probe_update_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Parameter buffers ────────────────────────────────────────
        let trace_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProbeGrid/TraceParams"),
            size: TRACE_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProbeGrid/UpdateParams"),
            size: UPDATE_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            traces: Vec::new(),
            update_pipeline,
            update_bgl,
            sh_buf_a: None,
            sh_buf_b: None,
            ray_results_buf: None,
            trace_params_buf,
            update_params_buf,
            grid_dims: [0, 0, 0],
            grid_min: [0.0, 0.0, 0.0],
            grid_step: [1.0, 1.0, 1.0],
            probe_count: 0,
            rays_per_frame: 64,
            hysteresis: 0.95,
            ping: false,
            device,
            queue,
        }
    }

    /// Compute grid dimensions from scene bounds and allocate SH + ray buffers.
    ///
    /// # Arguments
    ///
    /// * `bounds_min` - World-space minimum corner of the scene AABB
    /// * `bounds_max` - World-space maximum corner of the scene AABB
    /// * `spacing`    - Desired distance between probes (uniform on all axes)
    pub fn configure(&mut self, bounds_min: [f32; 3], bounds_max: [f32; 3], spacing: f32) {
        let spacing = spacing.max(0.01);

        // Compute grid dimensions (at least 2 probes per axis for interpolation)
        let dims = [
            ((bounds_max[0] - bounds_min[0]) / spacing).ceil().max(2.0) as u32,
            ((bounds_max[1] - bounds_min[1]) / spacing).ceil().max(2.0) as u32,
            ((bounds_max[2] - bounds_min[2]) / spacing).ceil().max(2.0) as u32,
        ];

        let step = [
            (bounds_max[0] - bounds_min[0]) / (dims[0] - 1).max(1) as f32,
            (bounds_max[1] - bounds_min[1]) / (dims[1] - 1).max(1) as f32,
            (bounds_max[2] - bounds_min[2]) / (dims[2] - 1).max(1) as f32,
        ];

        self.grid_dims = dims;
        self.grid_min = bounds_min;
        self.grid_step = step;
        self.probe_count = dims[0] * dims[1] * dims[2];

        // SH buffers: 9 vec4f per probe
        let sh_size = self.probe_count as u64 * SH_BYTES_PER_PROBE;

        let make_sh = |label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: sh_size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        self.sh_buf_a = Some(make_sh("ProbeGrid/SH_A"));
        self.sh_buf_b = Some(make_sh("ProbeGrid/SH_B"));

        // Ray results buffer: 2 vec4f per ray
        let total_rays = self.probe_count as u64 * self.rays_per_frame as u64;
        let ray_buf_size = total_rays * RAY_RESULT_BYTES;

        self.ray_results_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProbeGrid/RayResults"),
            size: ray_buf_size.max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        self.ping = false;
    }

    /// Dispatch probe trace + SH update passes.
    ///
    /// # Arguments
    ///
    /// * `encoder`      - Active command encoder
    /// * `bvh_data`     - GPU BVH data from `BVHBuilder`
    /// * `tlas_buf`     - TLAS BVH4 node buffer (unused on the hardware backend)
    /// * `materials_buf`- Materials storage buffer
    /// * `lights_buf`   - Lights storage buffer
    /// * `light_count`  - Number of lights
    /// * `frame_index`  - Current frame number
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bvh_data: &GPUBVHData,
        tlas_buf: &wgpu::Buffer,
        materials_buf: &wgpu::Buffer,
        lights_buf: &wgpu::Buffer,
        light_count: u32,
        frame_index: u32,
    ) {
        if self.probe_count == 0 {
            return;
        }

        let (Some(sh_a), Some(sh_b), Some(ray_buf)) = (
            self.sh_buf_a.as_ref(),
            self.sh_buf_b.as_ref(),
            self.ray_results_buf.as_ref(),
        ) else {
            return;
        };

        // Determine SH ping-pong: prev reads, cur writes
        let (sh_prev, sh_cur) = if self.ping {
            (sh_b, sh_a)
        } else {
            (sh_a, sh_b)
        };

        // ── Pass 1: Trace ────────────────────────────────────────────
        let trace_params = ProbeTraceParams {
            grid_min: self.grid_min,
            grid_step_x: self.grid_step[0],
            grid_dims: self.grid_dims,
            grid_step_y: self.grid_step[1],
            grid_step_z: self.grid_step[2],
            rays_per_probe: self.rays_per_frame,
            frame_index,
            light_count,
            max_distance: 100.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        self.queue
            .write_buffer(&self.trace_params_buf, 0, bytemuck::bytes_of(&trace_params));

        let backend = bvh_data.backend();
        if !self.traces.iter().any(|(b, ..)| *b == backend) {
            let (layout, pipeline) = trace_pipeline(&self.device, backend);
            self.traces.push((backend, layout, pipeline));
        }
        let (_, trace_bgl, trace_pipeline) = self.traces.iter().find(|(b, ..)| *b == backend).unwrap();
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.trace_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: materials_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: lights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: ray_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: sh_prev.as_entire_binding(),
            },
        ];
        entries.extend(bvh_data.bind_group_entries(tlas_buf, 1));
        let trace_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ProbeGrid/TraceBG"),
            layout: trace_bgl,
            entries: &entries,
        });

        let total_rays = self.probe_count * self.rays_per_frame;
        let trace_wg = (total_rays + 63) / 64;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ProbeGrid/Trace"),
                timestamp_writes: crate::profiling::gpu_pass("ProbeGrid/Trace").as_ref().map(crate::profiling::PassStamp::compute),
            });
            pass.set_pipeline(trace_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(trace_wg, 1, 1);
        }

        // ── Pass 2: SH Update ────────────────────────────────────────
        let update_params = ProbeUpdateParams {
            rays_per_probe: self.rays_per_frame,
            total_probes: self.probe_count,
            hysteresis: self.hysteresis,
            frame_index,
        };
        self.queue.write_buffer(
            &self.update_params_buf,
            0,
            bytemuck::bytes_of(&update_params),
        );

        let update_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ProbeGrid/UpdateBG"),
            layout: &self.update_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ray_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sh_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sh_cur.as_entire_binding(),
                },
            ],
        });

        let update_wg = (self.probe_count + 63) / 64;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ProbeGrid/Update"),
                timestamp_writes: crate::profiling::gpu_pass("ProbeGrid/Update").as_ref().map(crate::profiling::PassStamp::compute),
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &update_bg, &[]);
            pass.dispatch_workgroups(update_wg, 1, 1);
        }

        // Flip ping-pong
        self.ping = !self.ping;
    }

    /// Returns the current SH data buffer for the trace shader to read
    /// (the buffer that was most recently written to by `update()`).
    pub fn sh_buffer(&self) -> Option<&wgpu::Buffer> {
        // update() wrote sh_b when ping was false, then flipped it, so a
        // true ping means sh_b holds the newest coefficients.
        if self.ping {
            self.sh_buf_b.as_ref()
        } else {
            self.sh_buf_a.as_ref()
        }
    }
}
