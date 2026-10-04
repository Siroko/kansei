// SteeringSimulation -- compute-based 3D boids with spatial hash + verlet constraints.
// Engine-adjacent code: accesses renderer.device()/queue() internally.

use wgpu::util::DeviceExt;

use crate::text_data::ParticleData;

const PREFIX_SUM_BLOCK_SIZE: u32 = 512;
/// Cells per grid axis at most: 64³ cells is 512 prefix-sum blocks, all the top-level scan holds.
const MAX_GRID_DIM: u32 = 64;

const SIM_PARAMS_WGSL: &str = include_str!("shaders/sim_params.wgsl");
const GRID_CLEAR_WGSL: &str = include_str!("shaders/grid_clear.wgsl");
const GRID_ASSIGN_WGSL: &str = include_str!("shaders/grid_assign.wgsl");
const PREFIX_SUM_LOCAL_WGSL: &str = include_str!("shaders/prefix_sum_local.wgsl");
const PREFIX_SUM_TOP_WGSL: &str = include_str!("shaders/prefix_sum_top.wgsl");
const PREFIX_SUM_DISTRIBUTE_WGSL: &str = include_str!("shaders/prefix_sum_distribute.wgsl");
const SCATTER_WGSL: &str = include_str!("shaders/scatter.wgsl");
const STEERING_WGSL: &str = include_str!("shaders/steering.wgsl");
const VERLET_WGSL: &str = include_str!("shaders/verlet.wgsl");
const INTEGRATE_WGSL: &str = include_str!("shaders/integrate.wgsl");
const REPULSION_WGSL: &str = include_str!("shaders/repulsion.wgsl");

/// Prepend SimParams struct + helpers to a shader that needs them.
fn with_params(shader: &str) -> String {
    format!("{}\n{}", SIM_PARAMS_WGSL, shader)
}

// ── Tunable parameters (exposed to lib.rs) ──

pub struct SteeringParams {
    pub separation_strength: f32,
    pub separation_radius: f32,
    pub wander_strength: f32,
    pub wander_speed: f32,
    pub max_speed: f32,
    pub max_force: f32,
    pub damping: f32,
    pub bounds_size: f32,
    pub mouse_force: f32,
    pub cohesion_strength: f32,
    pub alignment_strength: f32,
    pub attractor_pos: [f32; 3],
    pub attractor_strength: f32,
    pub repulsion_strength: f32,
    pub repulsion_radius: f32,
    pub max_per_cell: u32,
    pub verlet_iterations: u32,
}

impl Default for SteeringParams {
    fn default() -> Self {
        Self {
            separation_strength: 8.0,
            separation_radius: 14.5,
            wander_strength: 0.0,
            wander_speed: 0.2,
            max_speed: 100.0,
            max_force: 32.0,
            damping: 1.0,
            bounds_size: 100.0,
            mouse_force: 1500.0,
            cohesion_strength: 8.3,
            alignment_strength: 7.9,
            attractor_pos: [0.0, 0.0, 0.0],
            attractor_strength: 2.0,
            repulsion_strength: 15.0,
            repulsion_radius: 6.0,
            max_per_cell: 64,
            verlet_iterations: 3,
        }
    }
}

/// The cursor in world space, from lib.rs each frame: the ray from the camera through it and its
/// motion turned into the camera's right/up plane.
pub struct MouseState {
    pub strength: f32,
    pub ray_origin: [f32; 3],
    pub ray_dir: [f32; 3],
    pub dir: [f32; 3],
}

// ── Internal pass wrapper (mirrors fluid sim pattern) ──

struct Pass {
    label: &'static str,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
}

impl Pass {
    fn new(device: &wgpu::Device, label: &'static str, code: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}/Shader", label)),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}/BGL", label)),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/Layout", label)),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}/Pipeline", label)),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Pass { label, pipeline, layout, bind_group: None }
    }

    /// Bind `buffers` to bindings 0, 1, 2, ... in order.
    fn bind(&mut self, device: &wgpu::Device, buffers: &[&wgpu::Buffer]) {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: binding as u32, resource: buffer.as_entire_binding() })
            .collect();
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}/BG", self.label)),
            layout: &self.layout,
            entries: &entries,
        }));
    }

    fn dispatch<'a>(&'a self, cpass: &mut wgpu::ComputePass<'a>, wx: u32, wy: u32, wz: u32) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, self.bind_group.as_ref(), &[]);
        cpass.dispatch_workgroups(wx, wy, wz);
    }
}

/// A zeroed `count`-element u32 storage buffer (at least one element).
fn zeroed_u32(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (count.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The hash grid for `params`: cells at least as wide as both search radii (the 3×3×3
/// neighbourhood must reach every neighbour) and wide enough that at most MAX_GRID_DIM of them
/// span the bounds; the grid starts at -bounds on every axis.
fn grid_layout(params: &SteeringParams) -> (f32, [u32; 3], [f32; 3]) {
    let bounds = params.bounds_size.max(1.0);
    let cell_size = params
        .separation_radius
        .max(params.repulsion_radius)
        .max(bounds * 2.0 / MAX_GRID_DIM as f32);
    let dim = ((bounds * 2.0 / cell_size).ceil() as u32).clamp(1, MAX_GRID_DIM);
    (cell_size, [dim; 3], [-bounds; 3])
}

// ── SimParams uniform layout (44 f32 = 176 bytes, 11 vec4 blocks) ──
// Offsets must match the WGSL struct exactly.

struct ParamOffsets;
impl ParamOffsets {
    const DT: usize = 0;
    const PARTICLE_COUNT: usize = 1;
    const VEHICLE_COUNT: usize = 2;
    const SEPARATION_STRENGTH: usize = 3;
    const SEPARATION_RADIUS: usize = 4;
    const WANDER_STRENGTH: usize = 5;
    const WANDER_SPEED: usize = 6;
    const MAX_SPEED: usize = 7;
    const MAX_FORCE: usize = 8;
    const DAMPING: usize = 9;
    const BOUNDS_SIZE: usize = 10;
    const TIME: usize = 11;
    const MOUSE_STRENGTH: usize = 12;
    const GRID_DIMS: usize = 13; // x, y, z
    const CELL_SIZE: usize = 16;
    const GRID_ORIGIN: usize = 17; // x, y, z
    const TOTAL_CELLS: usize = 20;
    const VERLET_ITERATIONS: usize = 21;
    const MOUSE_FORCE: usize = 22;
    const COHESION_STRENGTH: usize = 23;
    const ALIGNMENT_STRENGTH: usize = 24;
    const ATTRACTOR: usize = 25; // x, y, z
    const ATTRACTOR_STRENGTH: usize = 28;
    const REPULSION_STRENGTH: usize = 29;
    const REPULSION_RADIUS: usize = 30;
    const MAX_PER_CELL: usize = 31;
    const MOUSE_RAY_ORIGIN: usize = 32; // x, y, z
    const MOUSE_RAY_DIR: usize = 35; // x, y, z
    const MOUSE_DIR: usize = 38; // x, y, z; 41-43 pad
    const BUFFER_SIZE: usize = 44;
}

// ── Main simulation struct ──

pub struct SteeringSimulation {
    // Stored GPU handles (cheap Arc clones)
    device: wgpu::Device,
    queue: wgpu::Queue,

    // GPU buffers
    positions_buffer: wgpu::Buffer,
    velocities_buffer: wgpu::Buffer,
    word_meta_buffer: wgpu::Buffer,
    rest_lengths_buffer: wgpu::Buffer,
    vehicle_indices_buffer: wgpu::Buffer,
    cell_indices_buffer: wgpu::Buffer,
    sorted_indices_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    // sized by the grid: rebuilt (with every bind group) when its cell count changes
    cell_counts_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    scatter_counters_buffer: wgpu::Buffer,
    block_sums_buffer: wgpu::Buffer,

    // Compute passes
    grid_clear: Pass,
    grid_assign: Pass,
    prefix_sum_local: Pass,
    prefix_sum_top: Pass,
    prefix_sum_distribute: Pass,
    scatter: Pass,
    steering: Pass,
    verlet: Pass,
    integrate: Pass,
    repulsion: Pass,

    // Counts
    particle_count: u32,
    vehicle_count: u32,
    total_cells: u32,
    grid_dims: [u32; 3],
    grid_origin: [f32; 3],
    cell_size: f32,

    // Packed params for upload
    params_data: Vec<f32>,
}

impl SteeringSimulation {
    pub fn new(
        renderer: &kansei_core::renderers::Renderer,
        particle_data: &ParticleData,
        params: &SteeringParams,
    ) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let particle_count = particle_data.total_particles;

        // Build vehicle indices: global particle indices where letter_idx == 0
        let mut vehicle_indices: Vec<u32> = Vec::new();
        for i in 0..particle_count as usize {
            // word_meta is [word_id, letter_idx, word_len, particle_offset] per particle
            let letter_idx = particle_data.word_meta[i * 4 + 1];
            if letter_idx == 0 {
                vehicle_indices.push(i as u32);
            }
        }
        let vehicle_count = vehicle_indices.len() as u32;

        let (cell_size, grid_dims, grid_origin) = grid_layout(params);
        let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];

        // Create GPU buffers
        let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringSim/Positions"),
            contents: bytemuck::cast_slice(&particle_data.positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                 | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        });

        let velocities_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringSim/Velocities"),
            contents: bytemuck::cast_slice(&particle_data.velocities),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let word_meta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringSim/WordMeta"),
            contents: bytemuck::cast_slice(&particle_data.word_meta),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rest_lengths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringSim/RestLengths"),
            contents: bytemuck::cast_slice(&particle_data.rest_lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let vehicle_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringSim/VehicleIndices"),
            contents: bytemuck::cast_slice(&vehicle_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pc = particle_count as usize;
        let cell_indices_buffer = zeroed_u32(&device, "SteeringSim/CellIndices", pc);
        let sorted_indices_buffer = zeroed_u32(&device, "SteeringSim/SortedIndices", pc);
        let [cell_counts_buffer, cell_offsets_buffer, scatter_counters_buffer, block_sums_buffer] =
            Self::grid_buffers(&device, total_cells);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SteeringSim/Params"),
            size: (ParamOffsets::BUFFER_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create compute passes
        let c = wgpu::ShaderStages::COMPUTE;
        let buffer_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: c,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = |binding: u32| buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only: false });
        let storage_ro = |binding: u32| buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only: true });
        let uniform = |binding: u32| buffer_entry(binding, wgpu::BufferBindingType::Uniform);

        // Bindings in the order bind_passes gives the buffers
        let grid_clear = Pass::new(&device, "GridClear", GRID_CLEAR_WGSL, &[storage(0), storage(1)]);
        let grid_assign = Pass::new(&device, "GridAssign", &with_params(GRID_ASSIGN_WGSL),
            &[storage_ro(0), storage(1), storage(2), uniform(3)]);
        let prefix_sum_local = Pass::new(&device, "PrefixSumLocal", PREFIX_SUM_LOCAL_WGSL,
            &[storage(0), storage(1), storage(2)]);
        let prefix_sum_top = Pass::new(&device, "PrefixSumTop", PREFIX_SUM_TOP_WGSL, &[storage(0)]);
        let prefix_sum_distribute = Pass::new(&device, "PrefixSumDistribute", PREFIX_SUM_DISTRIBUTE_WGSL,
            &[storage(0), storage(1)]);
        let scatter = Pass::new(&device, "Scatter", &with_params(SCATTER_WGSL),
            &[storage_ro(0), storage_ro(1), storage(2), storage(3), uniform(4)]);
        let steering = Pass::new(&device, "Steering", &with_params(STEERING_WGSL),
            &[storage(0), storage(1), uniform(2), storage_ro(3), storage_ro(4), storage_ro(5), storage_ro(6), storage_ro(7)]);
        let verlet = Pass::new(&device, "Verlet", &with_params(VERLET_WGSL),
            &[storage(0), storage_ro(1), storage_ro(2), uniform(3)]);
        let integrate = Pass::new(&device, "Integrate", &with_params(INTEGRATE_WGSL),
            &[storage(0), storage(1), uniform(2)]);
        let repulsion = Pass::new(&device, "Repulsion", &with_params(REPULSION_WGSL),
            &[storage(0), uniform(1), storage_ro(2), storage_ro(3), storage_ro(4), storage_ro(5)]);

        let mut sim = Self {
            device,
            queue,
            positions_buffer,
            velocities_buffer,
            word_meta_buffer,
            rest_lengths_buffer,
            vehicle_indices_buffer,
            cell_indices_buffer,
            sorted_indices_buffer,
            params_buffer,
            cell_counts_buffer,
            cell_offsets_buffer,
            scatter_counters_buffer,
            block_sums_buffer,
            grid_clear,
            grid_assign,
            prefix_sum_local,
            prefix_sum_top,
            prefix_sum_distribute,
            scatter,
            steering,
            verlet,
            integrate,
            repulsion,
            particle_count,
            vehicle_count,
            total_cells,
            grid_dims,
            grid_origin,
            cell_size,
            params_data: vec![0.0f32; ParamOffsets::BUFFER_SIZE],
        };
        sim.bind_passes();
        sim
    }

    /// Cell counts, cell offsets, scatter counters and prefix-sum block sums for `total_cells`.
    fn grid_buffers(device: &wgpu::Device, total_cells: u32) -> [wgpu::Buffer; 4] {
        let tc = total_cells as usize;
        let blocks = tc.div_ceil(PREFIX_SUM_BLOCK_SIZE as usize);
        [
            zeroed_u32(device, "SteeringSim/CellCounts", tc),
            zeroed_u32(device, "SteeringSim/CellOffsets", tc),
            zeroed_u32(device, "SteeringSim/ScatterCounters", tc),
            zeroed_u32(device, "SteeringSim/BlockSums", blocks),
        ]
    }

    /// (Re)create every pass's bind group from the current buffers.
    fn bind_passes(&mut self) {
        let d = &self.device;
        // clears both cellCounts and scatterCounters
        self.grid_clear.bind(d, &[&self.cell_counts_buffer, &self.scatter_counters_buffer]);
        // hashes every particle's position into its cell
        self.grid_assign.bind(d, &[&self.positions_buffer, &self.cell_indices_buffer, &self.cell_counts_buffer, &self.params_buffer]);
        // exclusive scan of cellCounts into cellOffsets (cellCounts is left as it was)
        self.prefix_sum_local.bind(d, &[&self.cell_counts_buffer, &self.cell_offsets_buffer, &self.block_sums_buffer]);
        self.prefix_sum_top.bind(d, &[&self.block_sums_buffer]);
        self.prefix_sum_distribute.bind(d, &[&self.block_sums_buffer, &self.cell_offsets_buffer]);
        self.scatter.bind(d, &[
            &self.cell_indices_buffer, &self.cell_offsets_buffer, &self.scatter_counters_buffer,
            &self.sorted_indices_buffer, &self.params_buffer,
        ]);
        // boids: reads each cell's range (cellOffsets) and size (cellCounts)
        self.steering.bind(d, &[
            &self.positions_buffer, &self.velocities_buffer, &self.params_buffer, &self.sorted_indices_buffer,
            &self.cell_offsets_buffer, &self.cell_counts_buffer, &self.vehicle_indices_buffer, &self.word_meta_buffer,
        ]);
        self.verlet.bind(d, &[&self.positions_buffer, &self.word_meta_buffer, &self.rest_lengths_buffer, &self.params_buffer]);
        self.integrate.bind(d, &[&self.positions_buffer, &self.velocities_buffer, &self.params_buffer]);
        self.repulsion.bind(d, &[
            &self.positions_buffer, &self.params_buffer, &self.sorted_indices_buffer,
            &self.cell_offsets_buffer, &self.cell_counts_buffer, &self.word_meta_buffer,
        ]);
    }

    /// Fit the hash grid to the current radii and bounds; a new cell count reallocates the
    /// grid's buffers.
    fn fit_grid(&mut self, params: &SteeringParams) {
        let (cell_size, grid_dims, grid_origin) = grid_layout(params);
        self.cell_size = cell_size;
        self.grid_origin = grid_origin;
        if grid_dims == self.grid_dims {
            return;
        }
        self.grid_dims = grid_dims;
        self.total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
        [self.cell_counts_buffer, self.cell_offsets_buffer, self.scatter_counters_buffer, self.block_sums_buffer] =
            Self::grid_buffers(&self.device, self.total_cells);
        self.bind_passes();
    }

    /// Pack simulation parameters into the uniform buffer.
    fn pack_params(&mut self, params: &SteeringParams, dt: f32, time: f32, mouse: &MouseState) {
        let bits = |v: u32| f32::from_ne_bytes(v.to_ne_bytes());
        let f = &mut self.params_data;

        f[ParamOffsets::DT] = dt;
        f[ParamOffsets::PARTICLE_COUNT] = bits(self.particle_count);
        f[ParamOffsets::VEHICLE_COUNT] = bits(self.vehicle_count);
        f[ParamOffsets::SEPARATION_STRENGTH] = params.separation_strength;
        f[ParamOffsets::SEPARATION_RADIUS] = params.separation_radius;
        f[ParamOffsets::WANDER_STRENGTH] = params.wander_strength;
        f[ParamOffsets::WANDER_SPEED] = params.wander_speed;
        f[ParamOffsets::MAX_SPEED] = params.max_speed;
        f[ParamOffsets::MAX_FORCE] = params.max_force;
        f[ParamOffsets::DAMPING] = params.damping;
        f[ParamOffsets::BOUNDS_SIZE] = params.bounds_size;
        f[ParamOffsets::TIME] = time;
        f[ParamOffsets::MOUSE_STRENGTH] = mouse.strength;
        for axis in 0..3 {
            f[ParamOffsets::GRID_DIMS + axis] = bits(self.grid_dims[axis]);
            f[ParamOffsets::GRID_ORIGIN + axis] = self.grid_origin[axis];
            f[ParamOffsets::ATTRACTOR + axis] = params.attractor_pos[axis];
            f[ParamOffsets::MOUSE_RAY_ORIGIN + axis] = mouse.ray_origin[axis];
            f[ParamOffsets::MOUSE_RAY_DIR + axis] = mouse.ray_dir[axis];
            f[ParamOffsets::MOUSE_DIR + axis] = mouse.dir[axis];
        }
        f[ParamOffsets::CELL_SIZE] = self.cell_size;
        f[ParamOffsets::TOTAL_CELLS] = bits(self.total_cells);
        f[ParamOffsets::VERLET_ITERATIONS] = bits(params.verlet_iterations);
        f[ParamOffsets::MOUSE_FORCE] = params.mouse_force;
        f[ParamOffsets::COHESION_STRENGTH] = params.cohesion_strength;
        f[ParamOffsets::ALIGNMENT_STRENGTH] = params.alignment_strength;
        f[ParamOffsets::ATTRACTOR_STRENGTH] = params.attractor_strength;
        f[ParamOffsets::REPULSION_STRENGTH] = params.repulsion_strength;
        f[ParamOffsets::REPULSION_RADIUS] = params.repulsion_radius;
        f[ParamOffsets::MAX_PER_CELL] = bits(params.max_per_cell);
    }

    /// Run one simulation step: spatial hash build, steering, verlet, integrate.
    pub fn update(&mut self, params: &SteeringParams, dt: f32, time: f32, mouse: &MouseState) {
        self.fit_grid(params);
        self.pack_params(params, dt, time, mouse);
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&self.params_data));

        let vehicle_wg = ((self.vehicle_count + 63) / 64).max(1);
        let particle_wg = ((self.particle_count + 63) / 64).max(1);
        let grid_wg = ((self.total_cells + 255) / 256).max(1);
        let prefix_wg = ((self.total_cells + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE).max(1);

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("SteeringSim") },
        );

        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
            );

            // Spatial hash build
            self.grid_clear.dispatch(&mut cp, grid_wg, 1, 1);
            self.grid_assign.dispatch(&mut cp, particle_wg, 1, 1);
            self.prefix_sum_local.dispatch(&mut cp, prefix_wg, 1, 1);
            self.prefix_sum_top.dispatch(&mut cp, 1, 1, 1);
            self.prefix_sum_distribute.dispatch(&mut cp, prefix_wg, 1, 1);
            self.scatter.dispatch(&mut cp, particle_wg, 1, 1);

            // Steering forces (vehicles only)
            self.steering.dispatch(&mut cp, vehicle_wg, 1, 1);

            // Integration — move all particles by velocity × dt.
            self.integrate.dispatch(&mut cp, particle_wg, 1, 1);

            // Verlet distance constraints — snap trailing letters to
            // rest_length from their anchor.
            for _ in 0..params.verlet_iterations {
                self.verlet.dispatch(&mut cp, particle_wg, 1, 1);
            }

            // Repulsion AFTER verlet — pushes overlapping letters from
            // different words apart. Runs last so its changes aren't
            // overwritten by verlet snapping.
            self.repulsion.dispatch(&mut cp, particle_wg, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Public accessor so lib.rs can share this buffer with InstancedGeometry.
    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buffer
    }
}
