use super::params::*;
use super::pbf::{shader_sources as pbf_sources, GpuPbf};
use crate::simulations::grid::{GridLayout, NeighbourGrid, NeighbourGridOptions};

/// Cap on hash-grid cells (3 × u32 per cell → 24 MB at the cap). Large enough that the
/// cell stays equal to the smoothing radius for the tanks we use; `fit_grid` still
/// widens the cell if a scene would exceed it.
const MAX_GRID_CELLS: u32 = 2_097_152;
/// Workgroup size of the neighbor-search passes (density, forces). Substituted into the
/// WGSL as `__NEIGHBOR_WG__` so the dispatch count and the shader cannot disagree.
const NEIGHBOR_WG: u32 = 64;

pub(crate) const SIM_PARAMS_WGSL: &str = include_str!("shaders/sim-params.wgsl");
const DENSITY_WGSL: &str = include_str!("shaders/density.wgsl");
const FORCES_WGSL: &str = include_str!("shaders/forces.wgsl");
const INTEGRATE_WGSL: &str = include_str!("shaders/integrate.wgsl");

/// Prepend SimParams struct to a shader that needs it.
fn with_params(shader: &str) -> String {
    format!("{}\n{}", SIM_PARAMS_WGSL, shader).replace("__NEIGHBOR_WG__", &NEIGHBOR_WG.to_string())
}

/// A single compute pass — pipeline + bind group.
struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl Pass {
    fn dispatch(&self, cpass: &mut wgpu::ComputePass<'_>, wx: u32, wy: u32, wz: u32) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wx, wy, wz);
    }
}

/// A compute pass run after each substep's integration, in the same compute pass as the solver:
/// it corrects the particles' positions and velocities (a container's walls, moving colliders).
/// Implementors bind the simulation's positions, velocities and `SimParams` (which carry the
/// substep's `dt` and the particle count) with [`FluidSimulation::substep_pipeline`].
pub trait FluidSubstepPass {
    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, particle_count: u32);
}

/// SPH or Position Based Fluids simulation on a [`NeighbourGrid`].
///
/// The particle buffers hold [`capacity`](Self::capacity) particles, of which the first
/// [`particle_count`](Self::particle_count) are live: every pass (the solvers, the neighbour grid,
/// the [`FluidSubstepPass`]es, which get the live count in `SimParams`) runs on those only.
/// [`emit`](Self::emit) appends particles into the spare capacity at runtime and
/// [`reset_particles`](Self::reset_particles) puts back a set of them (an initial fill).
pub struct FluidSimulation {
    pub params: FluidSimulationOptions,
    pub world_bounds_min: [f32; 3],
    pub world_bounds_max: [f32; 3],
    /// Live particles: the first `particle_count` of `capacity`.
    particle_count: u32,
    capacity: u32,
    /// The neighbour grid's cells: the smoothing radius wide, coarsened only if the tank would
    /// otherwise need more than `MAX_GRID_CELLS` cells.
    layout: GridLayout,

    // Stored GPU handles (cheap Arc clones)
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,

    params_data: Vec<f32>,
    params_buffer: Option<wgpu::Buffer>,
    positions_buffer: Option<wgpu::Buffer>,
    original_positions_buffer: Option<wgpu::Buffer>,
    velocities_buffer: Option<wgpu::Buffer>,
    densities_buffer: Option<wgpu::Buffer>,
    /// The neighbour grid, also sorting the positions and velocities into cell order.
    grid: Option<NeighbourGrid>,
    // Camera matrices for mouse interaction in forces shader
    view_matrix_buffer: Option<wgpu::Buffer>,
    projection_matrix_buffer: Option<wgpu::Buffer>,
    inverse_view_matrix_buffer: Option<wgpu::Buffer>,
    world_matrix_buffer: Option<wgpu::Buffer>,

    // the solver's passes (the grid's are its own)
    density: Option<Pass>,
    forces: Option<Pass>,
    integrate: Option<Pass>,
    // Per-substep param buffers for batched update (avoids writeBuffer overwrite)
    substep_param_buffers: Vec<wgpu::Buffer>,
    /// Position Based Fluids' buffers and passes, made on its first step.
    pbf: Option<PbfPasses>,
}

impl FluidSimulation {
    /// A simulation of the particles at `positions` (4 floats each: x, y, z and 1), with no
    /// room for more.
    pub fn new(renderer: &crate::renderers::Renderer, params: FluidSimulationOptions, positions: &[f32]) -> Self {
        Self::with_capacity(renderer, params, positions, 0)
    }

    /// [`new`](Self::new), with buffers for `capacity` particles (at least those at `positions`):
    /// room for [`emit`](Self::emit) to add the rest at runtime.
    pub fn with_capacity(renderer: &crate::renderers::Renderer, params: FluidSimulationOptions, positions: &[f32], capacity: u32) -> Self {
        let device = renderer.device();
        let queue = renderer.queue();
        let particle_count = (positions.len() / 4) as u32;
        let mut sim = Self {
            params,
            world_bounds_min: [0.0; 3],
            world_bounds_max: [0.0; 3],
            particle_count,
            capacity: capacity.max(particle_count),
            layout: GridLayout { origin: [0.0; 3], cell_size: 1.0, dims: [1; 3] },
            device: Some(device.clone()),
            queue: Some(queue.clone()),
            params_data: vec![0.0; ParamOffsets::BUFFER_SIZE],
            params_buffer: None,
            positions_buffer: None,
            original_positions_buffer: None,
            velocities_buffer: None,
            densities_buffer: None,
            grid: None,
            view_matrix_buffer: None,
            projection_matrix_buffer: None,
            inverse_view_matrix_buffer: None,
            world_matrix_buffer: None,
            density: None,
            forces: None,
            integrate: None,
            substep_param_buffers: Vec::new(),
            pbf: None,
        };
        sim.compute_grid_from_positions(positions);
        sim.create_buffers(positions, device, queue);
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
        self.fit_grid();
    }

    /// Size the hash grid to the world bounds. Cells are `smoothing_radius`
    /// wide (the neighbor search only visits ±1 cell, so they must not be
    /// smaller) and are coarsened uniformly if the tank needs more than
    /// `MAX_GRID_CELLS`. Clamping each axis to the cube root of the cap
    /// instead (the old behavior) silently folded every particle beyond 64
    /// cells on an axis into the edge cell: tens of thousands of neighbors
    /// per particle, broken pressure, and a collapsed pool.
    fn fit_grid(&mut self) {
        let mut max = self.world_bounds_max;
        if self.params.dimensions == 2 {
            max[2] = self.world_bounds_min[2];
        }
        self.layout = GridLayout::covering(self.world_bounds_min, max, self.params.smoothing_radius, MAX_GRID_CELLS);
    }

    fn create_buffers(&mut self, positions: &[f32], device: &wgpu::Device, queue: &wgpu::Queue) {
        let n = self.capacity as usize;
        // the live particles, then the spare capacity (zeros: never read until emitted into)
        let mut positions = positions[..self.particle_count as usize * 4].to_vec();
        positions.resize(n * 4, 0.0);
        let positions = positions.as_slice();

        let mk_storage = |label: &str, data: &[f32]| -> wgpu::Buffer {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
        // Densities are stored in *sorted* (cell) order — see density.wgsl.
        self.densities_buffer = Some(mk_storage("Densities", &vec![0.0f32; n * 2]));
        let mut grid = NeighbourGrid::new(device, queue, &NeighbourGridOptions {
            label: "FluidSim/Grid",
            capacity: self.capacity,
            layout: self.layout,
            positions: self.positions_buffer.as_ref().unwrap(),
            sorted_copies: &[self.positions_buffer.as_ref().unwrap(), self.velocities_buffer.as_ref().unwrap()],
        });
        grid.set_count(self.particle_count);
        self.grid = Some(grid);

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
        let grid = self.grid.as_ref().unwrap();
        let co = grid.cell_offsets();
        let si = grid.sorted_indices();
        let par = self.params_buffer.as_ref().unwrap();
        let spos = grid.sorted(0);
        let svel = grid.sorted(1);

        // 8. Density
        self.density = Some(Self::make_pass(device, "Density", &with_params(DENSITY_WGSL),
            &[storage(0), storage(1), storage(2), uniform(3)],
            vec![buf!(0, spos), buf!(1, co), buf!(2, dens), buf!(3, par)]));

        // 9. Forces (bindings 7-10 are camera matrices for mouse interaction)
        let vm = self.view_matrix_buffer.as_ref().unwrap();
        let pm = self.projection_matrix_buffer.as_ref().unwrap();
        let ivm = self.inverse_view_matrix_buffer.as_ref().unwrap();
        let wm = self.world_matrix_buffer.as_ref().unwrap();
        self.forces = Some(Self::make_pass(device, "Forces", &with_params(FORCES_WGSL),
            &[storage(0), storage(1), storage(2), storage(3), storage(4), storage(5), uniform(6), uniform(7), uniform(8), uniform(9), uniform(10), storage(11)],
            vec![buf!(0, spos), buf!(1, svel), buf!(2, dens), buf!(3, orig), buf!(4, co), buf!(5, si), buf!(6, par), buf!(7, vm), buf!(8, pm), buf!(9, ivm), buf!(10, wm), buf!(11, vel)]));

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
        self.update_with(dt, mouse_strength, mouse_pos, mouse_dir, &[]);
    }

    /// [`update`](Self::update), running `extra` after each substep's integration, in order.
    pub fn update_with(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2], extra: &[&dyn FluidSubstepPass]) {
        let device = self.device.clone().expect("FluidSimulation not initialized");
        let queue = self.queue.clone().expect("FluidSimulation not initialized");
        for _s in 0..self.params.substeps {
            self.update_substep(&device, &queue, dt, mouse_strength, mouse_pos, mouse_dir, extra);
        }
    }

    /// Pack params and dispatch all substeps in a SINGLE queue.submit().
    /// Uses per-substep param buffers to avoid writeBuffer overwrite (see MEMORY.md).
    pub fn update_batched(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2]) {
        self.update_batched_with(dt, mouse_strength, mouse_pos, mouse_dir, &[]);
    }

    /// [`update_batched`](Self::update_batched), running `extra` after each substep's
    /// integration, in order.
    pub fn update_batched_with(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2], extra: &[&dyn FluidSubstepPass]) {
        let device = self.device.clone().expect("FluidSimulation not initialized");
        let queue = self.queue.clone().expect("FluidSimulation not initialized");
        let substeps = self.params.substeps;
        self.prepare_solver(&device, &queue);

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

            let stamp = crate::profiling::gpu_pass("FluidSim/Substep");
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FluidSim/Substep"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            self.encode_substep(&mut cp, extra);
            drop(cp);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run a single substep — pack params, dispatch the grid and solver passes, submit.
    fn update_substep(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32,
                          mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2], extra: &[&dyn FluidSubstepPass]) {
        self.prepare_solver(device, queue);
        self.pack_params(dt, mouse_strength, mouse_pos, mouse_dir);
        if let Some(ref pb) = self.params_buffer {
            queue.write_buffer(pb, 0, bytemuck::cast_slice(&self.params_data));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FluidSim") });
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        self.encode_substep(&mut cp, extra);
        drop(cp);

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// The solver's resources before a step: PBF's buffers and passes on first use, and its
    /// options.
    fn prepare_solver(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.params.solver != FluidSolver::Pbf {
            return;
        }
        if self.pbf.is_none() {
            self.pbf = Some(PbfPasses::new(self, device));
        }
        let gpu = GpuPbf::new(&self.params.pbf, self.params.smoothing_radius);
        queue.write_buffer(&self.pbf.as_ref().unwrap().params, 0, bytemuck::bytes_of(&gpu));
    }

    /// One substep's passes: the neighbour grid, then SPH (density, forces, integration) or PBF
    /// (predict, projections, velocities), with `extra` after the integration.
    fn encode_substep(&self, cp: &mut wgpu::ComputePass<'_>, extra: &[&dyn FluidSubstepPass]) {
        let n = self.particle_count;
        let particle_wg = (n + 63) / 64;
        let neighbor_wg = (n + NEIGHBOR_WG - 1) / NEIGHBOR_WG;
        macro_rules! dispatch {
            ($pass:expr, $wx:expr) => {
                if let Some(ref p) = $pass {
                    p.dispatch(cp, $wx, 1, 1);
                }
            };
        }
        let pbf = self.pbf.as_ref().filter(|_| self.params.solver == FluidSolver::Pbf);
        if let Some(pbf) = pbf {
            // predict, and keep the prediction in the container and out of the colliders
            pbf.predict.dispatch(cp, particle_wg, 1, 1);
            for pass in extra {
                pass.dispatch(cp, n);
            }
        }
        if let Some(grid) = &self.grid {
            grid.encode(cp);
        }
        match pbf {
            None => {
                dispatch!(self.density, neighbor_wg);
                dispatch!(self.forces, neighbor_wg);
                dispatch!(self.integrate, particle_wg);
                for pass in extra {
                    pass.dispatch(cp, n);
                }
            }
            Some(pbf) => {
                for _ in 0..self.params.pbf.iterations.max(1) {
                    pbf.lambda.dispatch(cp, neighbor_wg, 1, 1);
                    pbf.delta.dispatch(cp, neighbor_wg, 1, 1);
                    pbf.apply.dispatch(cp, particle_wg, 1, 1);
                }
                pbf.unsort.dispatch(cp, particle_wg, 1, 1);
                for pass in extra {
                    pass.dispatch(cp, n);
                }
                pbf.velocity.dispatch(cp, particle_wg, 1, 1);
                pbf.gather.dispatch(cp, particle_wg, 1, 1);
                if self.params.pbf.vorticity > 0.0 {
                    pbf.vorticity.dispatch(cp, neighbor_wg, 1, 1);
                }
                pbf.xsph.dispatch(cp, neighbor_wg, 1, 1);
            }
        }
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
        let grid = self.layout;
        f[ParamOffsets::GRID_DIMS_X] = f32::from_ne_bytes(grid.dims[0].to_ne_bytes());
        f[ParamOffsets::GRID_DIMS_Y] = f32::from_ne_bytes(grid.dims[1].to_ne_bytes());
        f[ParamOffsets::GRID_DIMS_Z] = f32::from_ne_bytes(grid.dims[2].to_ne_bytes());
        f[ParamOffsets::CELL_SIZE] = grid.cell_size;
        f[ParamOffsets::GRID_ORIGIN_X] = grid.origin[0];
        f[ParamOffsets::GRID_ORIGIN_Y] = grid.origin[1];
        f[ParamOffsets::GRID_ORIGIN_Z] = grid.origin[2];
        f[ParamOffsets::TOTAL_CELLS] = f32::from_ne_bytes(grid.total_cells().to_ne_bytes());
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
        f[ParamOffsets::GRAVITY_CENTER_X] = p.gravity_center[0];
        f[ParamOffsets::GRAVITY_CENTER_Y] = p.gravity_center[1];
        f[ParamOffsets::GRAVITY_CENTER_Z] = p.gravity_center[2];
        f[ParamOffsets::RADIAL_GRAVITY] = if p.radial_gravity { 1.0 } else { 0.0 };
        f[ParamOffsets::NEGATIVE_PRESSURE_SCALE] = p.negative_pressure_scale;
        f[ParamOffsets::SOLVER] = f32::from_ne_bytes(((p.solver == FluidSolver::Pbf) as u32).to_ne_bytes());
    }

    /// The live particles (the first `particle_count` of the buffers).
    pub fn particle_count(&self) -> u32 { self.particle_count }

    /// How many particles the buffers hold: the most there can be.
    pub fn capacity(&self) -> u32 { self.capacity }

    /// Append particles at `positions` with `velocities` (the simulation's space, per simulated
    /// second; one velocity for all, or one each) after the live ones, as many as the spare
    /// capacity takes: they join the next step. Returns how many were added.
    ///
    /// Each call writes three buffers (`queue.write_buffer`, which lands before the next
    /// submit): emit once per step, not per particle.
    pub fn emit(&mut self, positions: &[[f32; 3]], velocities: &[[f32; 3]]) -> u32 {
        assert!(velocities.len() == 1 || velocities.len() == positions.len(), "one velocity, or one per particle");
        let n = (positions.len() as u32).min(self.capacity - self.particle_count);
        if n == 0 {
            return 0;
        }
        let queue = self.queue.as_ref().expect("FluidSimulation not initialized");
        let p: Vec<f32> = positions[..n as usize].iter().flat_map(|q| [q[0], q[1], q[2], 1.0]).collect();
        let v: Vec<f32> = (0..n as usize).flat_map(|k| {
            let v = velocities[k.min(velocities.len() - 1)];
            [v[0], v[1], v[2], 0.0]
        }).collect();
        let offset = self.particle_count as u64 * 16;
        queue.write_buffer(self.positions_buffer.as_ref().unwrap(), offset, bytemuck::cast_slice(&p));
        queue.write_buffer(self.original_positions_buffer.as_ref().unwrap(), offset, bytemuck::cast_slice(&p));
        queue.write_buffer(self.velocities_buffer.as_ref().unwrap(), offset, bytemuck::cast_slice(&v));
        self.particle_count += n;
        self.grid.as_mut().unwrap().set_count(self.particle_count);
        n
    }

    /// Put the particles back to `positions` (4 floats each, as for [`new`](Self::new); at most
    /// `capacity` of them), at rest: e.g. the initial fill, dropping whatever was emitted since.
    pub fn reset_particles(&mut self, positions: &[f32]) {
        let n = ((positions.len() / 4) as u32).min(self.capacity);
        let queue = self.queue.as_ref().expect("FluidSimulation not initialized");
        let p = &positions[..n as usize * 4];
        queue.write_buffer(self.positions_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(p));
        queue.write_buffer(self.original_positions_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(p));
        queue.write_buffer(self.velocities_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&vec![0.0f32; n as usize * 4]));
        self.particle_count = n;
        self.grid.as_mut().unwrap().set_count(n);
    }

    /// The device and queue the simulation runs on.
    pub fn gpu(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (self.device.as_ref().expect("FluidSimulation not initialized"), self.queue.as_ref().expect("FluidSimulation not initialized"))
    }

    /// A compute pipeline for a [`FluidSubstepPass`] (entry point `main`) and its bind group:
    /// the positions (binding 0), velocities (1) and `SimParams` (2), then `extra` from binding
    /// 3 on, each a uniform (`false`) or a read-only storage buffer (`true`). `code` must declare
    /// them and include `SimParams` (the `sim-params.wgsl` source).
    pub fn substep_pipeline(&self, label: &str, code: &str, extra: &[(&wgpu::Buffer, bool)]) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let device = self.gpu().0;
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = wgpu::BufferBindingType::Storage { read_only: false };
        let mut entries = vec![entry(0, storage), entry(1, storage), entry(2, wgpu::BufferBindingType::Uniform)];
        let mut resources: Vec<&wgpu::Buffer> = vec![self.positions_buffer.as_ref().unwrap(), self.velocities_buffer.as_ref().unwrap(), self.params_buffer.as_ref().unwrap()];
        for (k, (buffer, is_storage)) in extra.iter().enumerate() {
            let ty = if *is_storage { wgpu::BufferBindingType::Storage { read_only: true } } else { wgpu::BufferBindingType::Uniform };
            entries.push(entry(3 + k as u32, ty));
            resources.push(buffer);
        }
        let bind_entries: Vec<wgpu::BindGroupEntry> = resources.iter().enumerate().map(|(k, b)| wgpu::BindGroupEntry { binding: k as u32, resource: b.as_entire_binding() }).collect();
        let pass = Self::make_pass(device, label, code, &entries, bind_entries);
        (pass.pipeline, pass.bind_group)
    }
    /// The box the particles are kept in (`world_bounds_min`/`max`, the simulation's space) grown
    /// by `margin` all round: for culling the fluid against a view
    /// ([`crate::culling::aabb_in_frustum`]), once mapped to the world.
    pub fn bounds(&self, margin: f32) -> (glam::Vec3, glam::Vec3) {
        (glam::Vec3::from(self.world_bounds_min) - margin, glam::Vec3::from(self.world_bounds_max) + margin)
    }
    pub fn grid_dims(&self) -> [u32; 3] { self.layout.dims }
    pub fn positions_buffer(&self) -> Option<&wgpu::Buffer> { self.positions_buffer.as_ref() }
    /// The per-particle velocity buffer (`array<vec4<f32>>`), for external
    /// additive passes such as the glyph attractor. `None` before initialization.
    pub fn velocities_buffer(&self) -> Option<&wgpu::Buffer> {
        self.velocities_buffer.as_ref()
    }
    pub fn params_buffer(&self) -> Option<&wgpu::Buffer> { self.params_buffer.as_ref() }

    /// The neighbour grid of the last substep, for passes that walk it (such as ray tracing the
    /// particles as spheres; bind its `params_buffer` with
    /// [`NEIGHBOUR_GRID_WGSL`](crate::simulations::grid::NEIGHBOUR_GRID_WGSL)). Its `sorted(0)`
    /// holds the positions in cell order (as they were before that substep's integration) and
    /// `sorted(1)` the velocities. [`rebuild_grid`](Self::rebuild_grid) replaces its per-cell
    /// buffers when the number of cells changes.
    pub fn grid(&self) -> Option<&NeighbourGrid> { self.grid.as_ref() }
    /// The grid's [`sorted`](NeighbourGrid::sorted)`(0)`, [`sorted_indices`](NeighbourGrid::sorted_indices)
    /// and [`cell_offsets`](NeighbourGrid::cell_offsets) (see [`grid`](Self::grid)). Cell
    /// `(x, y, z)` is `x + dims.x * (y + dims.y * z)` of `grid_dims`, `cell_size` wide from
    /// `grid_origin`.
    pub fn sorted_positions_buffer(&self) -> Option<&wgpu::Buffer> { self.grid.as_ref().map(|g| g.sorted(0)) }
    pub fn sorted_indices_buffer(&self) -> Option<&wgpu::Buffer> { self.grid.as_ref().map(NeighbourGrid::sorted_indices) }
    pub fn cell_offsets_buffer(&self) -> Option<&wgpu::Buffer> { self.grid.as_ref().map(NeighbourGrid::cell_offsets) }
    pub fn cell_size(&self) -> f32 { self.layout.cell_size }
    pub fn grid_origin(&self) -> [f32; 3] { self.layout.origin }

    /// Return the positions buffer wrapped as a `ComputeBuffer` with vec4 vertex
    /// layout at `shader_location`, ready for use with `InstancedGeometry`.
    pub fn positions_as_compute_buffer(&self, shader_location: u32) -> Option<crate::buffers::ComputeBuffer> {
        let buf = self.positions_buffer.as_ref()?.clone();
        Some(
            crate::buffers::ComputeBuffer::from_external("Positions", buf, crate::buffers::BufferType::Storage)
                .with_vertex_vec4(shader_location)
        )
    }

    /// Rebuild spatial grid from current world_bounds_min/max.
    /// Call after changing bounds at runtime.
    pub fn rebuild_grid(&mut self) {
        let device = self.device.clone().expect("FluidSimulation not initialized");
        self.fit_grid();
        // new per-cell buffers: the solver's passes again (PBF's on its next step)
        if self.grid.as_mut().unwrap().set_layout(self.layout) {
            self.create_passes(&device);
            self.pbf = None;
        }
    }
}

/// Position Based Fluids' buffers and passes on a simulation's buffers (see `pbf`).
pub(crate) struct PbfPasses {
    params: wgpu::Buffer,
    predict: Pass,
    lambda: Pass,
    delta: Pass,
    apply: Pass,
    unsort: Pass,
    velocity: Pass,
    gather: Pass,
    vorticity: Pass,
    xsph: Pass,
}

impl PbfPasses {
    fn new(sim: &FluidSimulation, device: &wgpu::Device) -> Self {
        let n = sim.capacity.max(1) as u64;
        let mk = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let previous = mk("FluidSim/PBF/Previous", n * 16);
        let lambdas = mk("FluidSim/PBF/Lambdas", n * 4);
        let deltas = mk("FluidSim/PBF/Deltas", n * 16);
        let omega = mk("FluidSim/PBF/Omega", n * 16);
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("FluidSim/PBF/Params"), size: std::mem::size_of::<GpuPbf>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sources: std::collections::HashMap<&str, String> = pbf_sources(SIM_PARAMS_WGSL, NEIGHBOR_WG).into_iter().collect();
        let pos = sim.positions_buffer.as_ref().unwrap();
        let vel = sim.velocities_buffer.as_ref().unwrap();
        let grid = sim.grid.as_ref().unwrap();
        let (spos, svel, co, si) = (grid.sorted(0), grid.sorted(1), grid.cell_offsets(), grid.sorted_indices());
        let sp = sim.params_buffer.as_ref().unwrap();
        // storage buffers (read-write) then uniforms, bound in order
        let pass = |name: &str, storage: &[&wgpu::Buffer], uniforms: &[&wgpu::Buffer]| -> Pass {
            let mut entries = Vec::new();
            let mut bind = Vec::new();
            for (k, b) in storage.iter().chain(uniforms).enumerate() {
                let ty = if k < storage.len() { wgpu::BufferBindingType::Storage { read_only: false } } else { wgpu::BufferBindingType::Uniform };
                entries.push(wgpu::BindGroupLayoutEntry { binding: k as u32, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None });
                bind.push(wgpu::BindGroupEntry { binding: k as u32, resource: b.as_entire_binding() });
            }
            FluidSimulation::make_pass(device, &format!("FluidSim/PBF/{name}"), &sources[name], &entries, bind)
        };
        Self {
            predict: pass("predict", &[pos, vel, &previous], &[sp]),
            lambda: pass("lambda", &[spos, co, &lambdas], &[sp, &params]),
            delta: pass("delta", &[spos, co, &lambdas, &deltas], &[sp, &params]),
            apply: pass("apply", &[spos, &deltas], &[sp]),
            unsort: pass("unsort", &[spos, si, pos], &[sp]),
            velocity: pass("velocity", &[pos, &previous, vel], &[sp, &params]),
            gather: pass("gather", &[pos, vel, si, spos, svel], &[sp]),
            vorticity: pass("vorticity", &[spos, svel, co, &omega], &[sp]),
            xsph: pass("xsph", &[spos, svel, co, &omega, si, vel], &[sp, &params]),
            params,
        }
    }
}
