use super::params::*;

const MAX_GRID_CELLS: u32 = 262144;
const PREFIX_SUM_BLOCK_SIZE: u32 = 512;

const SIM_PARAMS_WGSL: &str = include_str!("shaders/sim-params.wgsl");
const GRID_CLEAR_WGSL: &str = include_str!("shaders/grid-clear.wgsl");
const GRID_ASSIGN_WGSL: &str = include_str!("shaders/grid-assign.wgsl");
const PREFIX_SUM_LOCAL_WGSL: &str = include_str!("shaders/prefix-sum-local.wgsl");
const PREFIX_SUM_TOP_WGSL: &str = include_str!("shaders/prefix-sum-top.wgsl");
const PREFIX_SUM_DISTRIBUTE_WGSL: &str = include_str!("shaders/prefix-sum-distribute.wgsl");
const SCATTER_WGSL: &str = include_str!("shaders/scatter.wgsl");
const DENSITY_WGSL: &str = include_str!("shaders/density.wgsl");
const FORCES_WGSL: &str = include_str!("shaders/forces.wgsl");
const INTEGRATE_WGSL: &str = include_str!("shaders/integrate.wgsl");

/// Prepend SimParams struct to a shader that needs it.
fn with_params(shader: &str) -> String {
    format!("{}\n{}", SIM_PARAMS_WGSL, shader)
}

/// A single compute pass — pipeline + bind group.
struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl Pass {
    fn dispatch<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, wx: u32, wy: u32, wz: u32) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wx, wy, wz);
    }
}

/// SPH fluid simulation — 10 compute passes per substep.
pub struct FluidSimulation {
    pub params: FluidSimulationOptions,
    pub world_bounds_min: [f32; 3],
    pub world_bounds_max: [f32; 3],
    particle_count: u32,
    grid_dims: [u32; 3],
    grid_origin: [f32; 3],
    total_cells: u32,

    // Stored GPU handles (cheap Arc clones)
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,

    params_data: Vec<f32>,
    params_buffer: Option<wgpu::Buffer>,
    positions_buffer: Option<wgpu::Buffer>,
    original_positions_buffer: Option<wgpu::Buffer>,
    velocities_buffer: Option<wgpu::Buffer>,
    densities_buffer: Option<wgpu::Buffer>,
    cell_indices_buffer: Option<wgpu::Buffer>,
    cell_counts_buffer: Option<wgpu::Buffer>,
    cell_offsets_buffer: Option<wgpu::Buffer>,
    scatter_counters_buffer: Option<wgpu::Buffer>,
    sorted_indices_buffer: Option<wgpu::Buffer>,
    block_sums_buffer: Option<wgpu::Buffer>,
    // Camera matrices for mouse interaction in forces shader
    view_matrix_buffer: Option<wgpu::Buffer>,
    projection_matrix_buffer: Option<wgpu::Buffer>,
    inverse_view_matrix_buffer: Option<wgpu::Buffer>,
    world_matrix_buffer: Option<wgpu::Buffer>,

    // 10 compute passes
    grid_clear_counts: Option<Pass>,
    grid_clear_scatter: Option<Pass>,
    grid_assign: Option<Pass>,
    prefix_sum_local: Option<Pass>,
    prefix_sum_top: Option<Pass>,
    prefix_sum_distribute: Option<Pass>,
    scatter: Option<Pass>,
    density: Option<Pass>,
    forces: Option<Pass>,
    integrate: Option<Pass>,
    // Per-substep param buffers for batched update (avoids writeBuffer overwrite)
    substep_param_buffers: Vec<wgpu::Buffer>,
}

impl FluidSimulation {
    pub fn new(renderer: &crate::renderers::Renderer, params: FluidSimulationOptions, positions: &[f32]) -> Self {
        let device = renderer.device();
        let queue = renderer.queue();
        let particle_count = (positions.len() / 4) as u32;
        let mut sim = Self {
            params,
            world_bounds_min: [0.0; 3],
            world_bounds_max: [0.0; 3],
            particle_count,
            grid_dims: [1, 1, 1],
            grid_origin: [0.0; 3],
            total_cells: 1,
            device: Some(device.clone()),
            queue: Some(queue.clone()),
            params_data: vec![0.0; ParamOffsets::BUFFER_SIZE],
            params_buffer: None,
            positions_buffer: None,
            original_positions_buffer: None,
            velocities_buffer: None,
            densities_buffer: None,
            cell_indices_buffer: None,
            cell_counts_buffer: None,
            cell_offsets_buffer: None,
            scatter_counters_buffer: None,
            sorted_indices_buffer: None,
            block_sums_buffer: None,
            view_matrix_buffer: None,
            projection_matrix_buffer: None,
            inverse_view_matrix_buffer: None,
            world_matrix_buffer: None,
            grid_clear_counts: None,
            grid_clear_scatter: None,
            grid_assign: None,
            prefix_sum_local: None,
            prefix_sum_top: None,
            prefix_sum_distribute: None,
            scatter: None,
            density: None,
            forces: None,
            integrate: None,
            substep_param_buffers: Vec::new(),
        };
        sim.compute_grid_from_positions(positions);
        sim.create_buffers(positions, device);
        sim.create_passes(device);
        sim
    }

    fn compute_grid_from_positions(&mut self, positions: &[f32]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for i in 0..self.particle_count as usize {
            for d in 0..3 {
                let v = positions[i * 4 + d];
                min[d] = min[d].min(v);
                max[d] = max[d].max(v);
            }
        }
        let pad = self.params.world_bounds_padding;
        for d in 0..3 {
            let range = if d == 2 && self.params.dimensions == 2 { 0.01 } else { (max[d] - min[d]).max(1.0) };
            self.world_bounds_min[d] = min[d] - range * pad;
            self.world_bounds_max[d] = max[d] + range * pad;
        }
        let cs = self.params.smoothing_radius;
        let mpa = if self.params.dimensions == 3 { (MAX_GRID_CELLS as f32).cbrt() as u32 } else { (MAX_GRID_CELLS as f32).sqrt() as u32 };
        for d in 0..3 {
            let g = ((self.world_bounds_max[d] - self.world_bounds_min[d]) / cs).ceil() as u32;
            let lim = if d == 2 && self.params.dimensions == 2 { 1 } else { mpa };
            self.grid_dims[d] = g.max(1).min(lim);
        }
        self.total_cells = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2];
        self.grid_origin = self.world_bounds_min;
    }

    fn create_buffers(&mut self, positions: &[f32], device: &wgpu::Device) {
        let n = self.particle_count as usize;
        let tc = self.total_cells as usize;

        let mk_storage = |label: &str, data: &[f32]| -> wgpu::Buffer {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        self.params_buffer = Some({
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SimParams"),
                contents: bytemuck::cast_slice(&self.params_data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        self.positions_buffer = Some({
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Positions"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            })
        });

        self.original_positions_buffer = Some(mk_storage("OrigPositions", positions));
        self.velocities_buffer = Some(mk_storage("Velocities", &vec![0.0f32; n * 4]));
        self.densities_buffer = Some(mk_storage("Densities", &vec![0.0f32; n * 2]));
        self.cell_indices_buffer = Some(mk_storage("CellIndices", &vec![0.0f32; n]));
        self.sorted_indices_buffer = Some(mk_storage("SortedIndices", &vec![0.0f32; n]));
        self.cell_counts_buffer = Some(mk_storage("CellCounts", &vec![0.0f32; tc]));
        self.cell_offsets_buffer = Some(mk_storage("CellOffsets", &vec![0.0f32; tc]));
        self.scatter_counters_buffer = Some(mk_storage("ScatterCounters", &vec![0.0f32; tc]));
        let nb = ((tc + PREFIX_SUM_BLOCK_SIZE as usize - 1) / PREFIX_SUM_BLOCK_SIZE as usize).max(1);
        self.block_sums_buffer = Some(mk_storage("BlockSums", &vec![0.0f32; nb]));

        // Identity matrices for camera (mouse interaction in forces shader)
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        let mk_mat = |label: &str| -> wgpu::Buffer {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&identity),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        self.view_matrix_buffer = Some(mk_mat("ViewMatrix"));
        self.projection_matrix_buffer = Some(mk_mat("ProjectionMatrix"));
        self.inverse_view_matrix_buffer = Some(mk_mat("InverseViewMatrix"));
        self.world_matrix_buffer = Some(mk_mat("WorldMatrix"));
    }

    fn make_pass(device: &wgpu::Device, label: &str, code: &str, entries: &[wgpu::BindGroupLayoutEntry], bg_entries: Vec<wgpu::BindGroupEntry>) -> Pass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}/Shader", label)),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}/BGL", label)),
            entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/Layout", label)),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}/Pipeline", label)),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}/BG", label)),
            layout: &bgl,
            entries: &bg_entries,
        });
        Pass { pipeline, bind_group }
    }

    fn create_passes(&mut self, device: &wgpu::Device) {
        let c = wgpu::ShaderStages::COMPUTE;
        let storage = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }
        };
        let uniform = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry { binding, visibility: c, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
        };
        macro_rules! buf {
            ($binding:expr, $buffer:expr) => {
                wgpu::BindGroupEntry { binding: $binding, resource: $buffer.as_entire_binding() }
            };
        }

        let pos = self.positions_buffer.as_ref().unwrap();
        let orig = self.original_positions_buffer.as_ref().unwrap();
        let vel = self.velocities_buffer.as_ref().unwrap();
        let dens = self.densities_buffer.as_ref().unwrap();
        let ci = self.cell_indices_buffer.as_ref().unwrap();
        let cc = self.cell_counts_buffer.as_ref().unwrap();
        let co = self.cell_offsets_buffer.as_ref().unwrap();
        let sc = self.scatter_counters_buffer.as_ref().unwrap();
        let si = self.sorted_indices_buffer.as_ref().unwrap();
        let bs = self.block_sums_buffer.as_ref().unwrap();
        let par = self.params_buffer.as_ref().unwrap();

        // 1. Grid clear counts — clears cell_counts
        self.grid_clear_counts = Some(Self::make_pass(device, "GridClearCounts", GRID_CLEAR_WGSL,
            &[storage(0)], vec![buf!(0, cc)]));

        // 2. Grid clear scatter — clears scatter_counters
        self.grid_clear_scatter = Some(Self::make_pass(device, "GridClearScatter", GRID_CLEAR_WGSL,
            &[storage(0)], vec![buf!(0, sc)]));

        // 3. Grid assign
        self.grid_assign = Some(Self::make_pass(device, "GridAssign", &with_params(GRID_ASSIGN_WGSL),
            &[storage(0), storage(1), storage(2), uniform(3)],
            vec![buf!(0, pos), buf!(1, ci), buf!(2, cc), buf!(3, par)]));

        // 4. Prefix sum local
        self.prefix_sum_local = Some(Self::make_pass(device, "PrefixSumLocal", PREFIX_SUM_LOCAL_WGSL,
            &[storage(0), storage(1), storage(2)],
            vec![buf!(0, cc), buf!(1, co), buf!(2, bs)]));

        // 5. Prefix sum top
        self.prefix_sum_top = Some(Self::make_pass(device, "PrefixSumTop", PREFIX_SUM_TOP_WGSL,
            &[storage(0)], vec![buf!(0, bs)]));

        // 6. Prefix sum distribute
        self.prefix_sum_distribute = Some(Self::make_pass(device, "PrefixSumDistribute", PREFIX_SUM_DISTRIBUTE_WGSL,
            &[storage(0), storage(1)],
            vec![buf!(0, bs), buf!(1, co)]));

        // 7. Scatter
        self.scatter = Some(Self::make_pass(device, "Scatter", &with_params(SCATTER_WGSL),
            &[storage(0), storage(1), storage(2), storage(3), uniform(4)],
            vec![buf!(0, ci), buf!(1, co), buf!(2, sc), buf!(3, si), buf!(4, par)]));

        // 8. Density
        self.density = Some(Self::make_pass(device, "Density", &with_params(DENSITY_WGSL),
            &[storage(0), storage(1), storage(2), storage(3), uniform(4)],
            vec![buf!(0, pos), buf!(1, co), buf!(2, si), buf!(3, dens), buf!(4, par)]));

        // 9. Forces (bindings 7-10 are camera matrices for mouse interaction)
        let vm = self.view_matrix_buffer.as_ref().unwrap();
        let pm = self.projection_matrix_buffer.as_ref().unwrap();
        let ivm = self.inverse_view_matrix_buffer.as_ref().unwrap();
        let wm = self.world_matrix_buffer.as_ref().unwrap();
        self.forces = Some(Self::make_pass(device, "Forces", &with_params(FORCES_WGSL),
            &[storage(0), storage(1), storage(2), storage(3), storage(4), storage(5), uniform(6), uniform(7), uniform(8), uniform(9), uniform(10)],
            vec![buf!(0, pos), buf!(1, vel), buf!(2, dens), buf!(3, orig), buf!(4, co), buf!(5, si), buf!(6, par), buf!(7, vm), buf!(8, pm), buf!(9, ivm), buf!(10, wm)]));

        // 10. Integrate
        self.integrate = Some(Self::make_pass(device, "Integrate", &with_params(INTEGRATE_WGSL),
            &[storage(0), storage(1), uniform(2)],
            vec![buf!(0, pos), buf!(1, vel), buf!(2, par)]));
    }

    /// Upload camera matrices for mouse interaction in the forces shader.
    pub fn set_camera_matrices(&self, view: &[f32; 16], proj: &[f32; 16], inv_view: &[f32; 16], world: &[f32; 16]) {
        let queue = self.queue.as_ref().expect("FluidSimulation not initialized");
        if let Some(ref b) = self.view_matrix_buffer { queue.write_buffer(b, 0, bytemuck::cast_slice(view)); }
        if let Some(ref b) = self.projection_matrix_buffer { queue.write_buffer(b, 0, bytemuck::cast_slice(proj)); }
        if let Some(ref b) = self.inverse_view_matrix_buffer { queue.write_buffer(b, 0, bytemuck::cast_slice(inv_view)); }
        if let Some(ref b) = self.world_matrix_buffer { queue.write_buffer(b, 0, bytemuck::cast_slice(world)); }
    }

    /// Pack params and dispatch all substeps — one submit per substep.
    pub fn update(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2]) {
        let device = self.device.clone().expect("FluidSimulation not initialized");
        let queue = self.queue.clone().expect("FluidSimulation not initialized");
        for _s in 0..self.params.substeps {
            self.update_substep(&device, &queue, dt, mouse_strength, mouse_pos, mouse_dir);
        }
    }

    /// Pack params and dispatch all substeps in a SINGLE queue.submit().
    /// Uses per-substep param buffers to avoid writeBuffer overwrite (see MEMORY.md).
    pub fn update_batched(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2]) {
        let device = self.device.clone().expect("FluidSimulation not initialized");
        let queue = self.queue.clone().expect("FluidSimulation not initialized");
        let n = self.particle_count;
        let particle_wg = ((n + 63) / 64) as u32;
        let grid_wg = ((self.total_cells + 255) / 256) as u32;
        let prefix_wg = ((self.total_cells + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE).max(1) as u32;
        let substeps = self.params.substeps;

        // Ensure we have per-substep param buffers
        while self.substep_param_buffers.len() < substeps as usize {
            self.substep_param_buffers.push(
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("FluidSim/SubstepParams"),
                    size: (self.params_data.len() * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            );
        }

        // Upload all substep params BEFORE encoding (writeBuffer is immediate)
        for s in 0..substeps as usize {
            self.pack_params(dt, mouse_strength, mouse_pos, mouse_dir);
            queue.write_buffer(&self.substep_param_buffers[s], 0, bytemuck::cast_slice(&self.params_data));
        }

        // Encode all substeps into one command buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FluidSim/Batched") });

        for s in 0..substeps as usize {
            // Copy substep params to the main params buffer used by compute passes
            if let Some(ref pb) = self.params_buffer {
                encoder.copy_buffer_to_buffer(
                    &self.substep_param_buffers[s], 0,
                    pb, 0,
                    (self.params_data.len() * 4) as u64,
                );
            }

            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            macro_rules! dispatch {
                ($pass:expr, $wx:expr) => {
                    if let Some(ref p) = $pass {
                        p.dispatch(&mut cp, $wx, 1, 1);
                    }
                };
            }

            dispatch!(self.grid_clear_counts, grid_wg);
            dispatch!(self.grid_clear_scatter, grid_wg);
            dispatch!(self.grid_assign, particle_wg);
            dispatch!(self.prefix_sum_local, prefix_wg.max(1));
            dispatch!(self.prefix_sum_top, 1);
            dispatch!(self.prefix_sum_distribute, prefix_wg.max(1));
            dispatch!(self.scatter, particle_wg);
            dispatch!(self.density, particle_wg);
            dispatch!(self.forces, particle_wg);
            dispatch!(self.integrate, particle_wg);
            drop(cp);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run a single substep — pack params, dispatch all 10 compute passes, submit.
    fn update_substep(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32,
                          mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2]) {
        let n = self.particle_count;
        let particle_wg = ((n + 63) / 64) as u32;
        let grid_wg = ((self.total_cells + 255) / 256) as u32;
        let prefix_wg = ((self.total_cells + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE).max(1) as u32;

        self.pack_params(dt, mouse_strength, mouse_pos, mouse_dir);
        if let Some(ref pb) = self.params_buffer {
            queue.write_buffer(pb, 0, bytemuck::cast_slice(&self.params_data));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FluidSim") });
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });

        macro_rules! dispatch {
            ($pass:expr, $wx:expr) => {
                if let Some(ref p) = $pass {
                    p.dispatch(&mut cp, $wx, 1, 1);
                }
            };
        }

        dispatch!(self.grid_clear_counts, grid_wg);
        dispatch!(self.grid_clear_scatter, grid_wg);
        dispatch!(self.grid_assign, particle_wg);
        dispatch!(self.prefix_sum_local, prefix_wg.max(1));
        dispatch!(self.prefix_sum_top, 1);
        dispatch!(self.prefix_sum_distribute, prefix_wg.max(1));
        dispatch!(self.scatter, particle_wg);
        dispatch!(self.density, particle_wg);
        dispatch!(self.forces, particle_wg);
        dispatch!(self.integrate, particle_wg);
        drop(cp);

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn pack_params(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2]) {
        let p = &self.params;
        let f = &mut self.params_data;
        let sub_dt = dt / p.substeps as f32;

        f[ParamOffsets::DT] = sub_dt;
        f[ParamOffsets::PARTICLE_COUNT] = f32::from_ne_bytes(self.particle_count.to_ne_bytes());
        f[ParamOffsets::DIMENSIONS] = f32::from_ne_bytes(p.dimensions.to_ne_bytes());
        f[ParamOffsets::SMOOTHING_RADIUS] = p.smoothing_radius;
        f[ParamOffsets::PRESSURE_MULTIPLIER] = p.pressure_multiplier;
        f[ParamOffsets::DENSITY_TARGET] = p.density_target;
        f[ParamOffsets::NEAR_PRESSURE_MULTIPLIER] = p.near_pressure_multiplier;
        f[ParamOffsets::VISCOSITY] = p.viscosity;
        f[ParamOffsets::DAMPING] = p.damping;
        f[ParamOffsets::RETURN_TO_ORIGIN_STRENGTH] = p.return_to_origin_strength;
        f[ParamOffsets::MOUSE_STRENGTH] = mouse_strength;
        f[ParamOffsets::MOUSE_RADIUS] = p.mouse_radius;
        f[ParamOffsets::GRAVITY_X] = p.gravity[0];
        f[ParamOffsets::GRAVITY_Y] = p.gravity[1];
        f[ParamOffsets::GRAVITY_Z] = p.gravity[2];
        f[ParamOffsets::MOUSE_FORCE] = p.mouse_force;
        f[ParamOffsets::MOUSE_POS_X] = mouse_pos[0];
        f[ParamOffsets::MOUSE_POS_Y] = mouse_pos[1];
        f[ParamOffsets::MOUSE_DIR_X] = mouse_dir[0];
        f[ParamOffsets::MOUSE_DIR_Y] = mouse_dir[1];
        f[ParamOffsets::GRID_DIMS_X] = f32::from_ne_bytes(self.grid_dims[0].to_ne_bytes());
        f[ParamOffsets::GRID_DIMS_Y] = f32::from_ne_bytes(self.grid_dims[1].to_ne_bytes());
        f[ParamOffsets::GRID_DIMS_Z] = f32::from_ne_bytes(self.grid_dims[2].to_ne_bytes());
        f[ParamOffsets::CELL_SIZE] = p.smoothing_radius;
        f[ParamOffsets::GRID_ORIGIN_X] = self.grid_origin[0];
        f[ParamOffsets::GRID_ORIGIN_Y] = self.grid_origin[1];
        f[ParamOffsets::GRID_ORIGIN_Z] = self.grid_origin[2];
        f[ParamOffsets::TOTAL_CELLS] = f32::from_ne_bytes(self.total_cells.to_ne_bytes());
        f[ParamOffsets::WORLD_BOUNDS_MIN_X] = self.world_bounds_min[0];
        f[ParamOffsets::WORLD_BOUNDS_MIN_Y] = self.world_bounds_min[1];
        f[ParamOffsets::WORLD_BOUNDS_MIN_Z] = self.world_bounds_min[2];
        f[ParamOffsets::WORLD_BOUNDS_MAX_X] = self.world_bounds_max[0];
        f[ParamOffsets::WORLD_BOUNDS_MAX_Y] = self.world_bounds_max[1];
        f[ParamOffsets::WORLD_BOUNDS_MAX_Z] = self.world_bounds_max[2];

        let k = if p.dimensions == 3 { compute_kernel_factors_3d(p.smoothing_radius) } else { compute_kernel_factors_2d(p.smoothing_radius) };
        f[ParamOffsets::POLY6_FACTOR] = k.poly6;
        f[ParamOffsets::SPIKY_POW2_FACTOR] = k.spiky_pow2;
        f[ParamOffsets::SPIKY_POW3_FACTOR] = k.spiky_pow3;
        f[ParamOffsets::SPIKY_POW2_DERIV_FACTOR] = k.spiky_pow2_deriv;
        f[ParamOffsets::SPIKY_POW3_DERIV_FACTOR] = k.spiky_pow3_deriv;
    }

    pub fn particle_count(&self) -> u32 { self.particle_count }
    pub fn grid_dims(&self) -> [u32; 3] { self.grid_dims }
    pub fn positions_buffer(&self) -> Option<&wgpu::Buffer> { self.positions_buffer.as_ref() }
    pub fn params_buffer(&self) -> Option<&wgpu::Buffer> { self.params_buffer.as_ref() }

    /// Rebuild spatial grid from current world_bounds_min/max.
    /// Call after changing bounds at runtime.
    pub fn rebuild_grid(&mut self) {
        let device = self.device.clone().expect("FluidSimulation not initialized");
        let cs = self.params.smoothing_radius;
        let mpa = if self.params.dimensions == 3 {
            (MAX_GRID_CELLS as f32).cbrt() as u32
        } else {
            (MAX_GRID_CELLS as f32).sqrt() as u32
        };
        for d in 0..3 {
            let g = ((self.world_bounds_max[d] - self.world_bounds_min[d]) / cs).ceil() as u32;
            let lim = if d == 2 && self.params.dimensions == 2 { 1 } else { mpa };
            self.grid_dims[d] = g.max(1).min(lim);
        }
        self.total_cells = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2];
        self.grid_origin = self.world_bounds_min;

        // Reallocate grid-sized buffers
        let tc = self.total_cells as usize;
        let mk = |label: &str, size: usize| -> wgpu::Buffer {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&vec![0.0f32; size]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        self.cell_counts_buffer = Some(mk("CellCounts", tc));
        self.cell_offsets_buffer = Some(mk("CellOffsets", tc));
        self.scatter_counters_buffer = Some(mk("ScatterCounters", tc));
        let nb = ((tc + PREFIX_SUM_BLOCK_SIZE as usize - 1) / PREFIX_SUM_BLOCK_SIZE as usize).max(1);
        self.block_sums_buffer = Some(mk("BlockSums", nb));

        // Recreate compute passes with new buffers
        self.create_passes(&device);
    }
}
