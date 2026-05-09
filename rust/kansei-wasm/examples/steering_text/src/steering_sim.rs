// SteeringSimulation -- compute-based 3D boids with spatial hash + verlet constraints.
// Engine-adjacent code: accesses renderer.device()/queue() internally.

use wgpu::util::DeviceExt;

use crate::text_data::ParticleData;

const PREFIX_SUM_BLOCK_SIZE: u32 = 512;

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

/// Mouse state passed from lib.rs each frame.
pub struct MouseState {
    pub strength: f32,
    pub pos: [f32; 2],
    pub dir: [f32; 2],
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            strength: 0.0,
            pos: [0.0; 2],
            dir: [0.0; 2],
        }
    }
}

// ── Internal pass wrapper (mirrors fluid sim pattern) ──

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

// ── SimParams uniform layout (32 f32 = 128 bytes, 8 vec4 blocks) ──
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
    const MOUSE_POS_X: usize = 13;
    const MOUSE_POS_Y: usize = 14;
    const MOUSE_DIR_X: usize = 15;
    const MOUSE_DIR_Y: usize = 16;
    const GRID_DIMS_X: usize = 17;
    const GRID_DIMS_Y: usize = 18;
    const GRID_DIMS_Z: usize = 19;
    const CELL_SIZE: usize = 20;
    const GRID_ORIGIN_X: usize = 21;
    const GRID_ORIGIN_Y: usize = 22;
    const GRID_ORIGIN_Z: usize = 23;
    const TOTAL_CELLS: usize = 24;
    const VERLET_ITERATIONS: usize = 25;
    const MOUSE_FORCE: usize = 26;
    const COHESION_STRENGTH: usize = 27;
    const ALIGNMENT_STRENGTH: usize = 28;
    const ATTRACTOR_X: usize = 29;
    const ATTRACTOR_Y: usize = 30;
    const ATTRACTOR_Z: usize = 31;
    const ATTRACTOR_STRENGTH: usize = 32;
    const REPULSION_STRENGTH: usize = 33;
    const REPULSION_RADIUS: usize = 34;
    const MAX_PER_CELL: usize = 35;
    const _PAD3: usize = 36;
    const BUFFER_SIZE: usize = 40; // round up to vec4 alignment
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
    cell_counts_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    scatter_counters_buffer: wgpu::Buffer,
    sorted_indices_buffer: wgpu::Buffer,
    block_sums_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

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

        // Compute grid dimensions from bounds and separation radius
        let cell_size = params.separation_radius;
        let bounds = params.bounds_size;
        let grid_dim = ((bounds * 2.0) / cell_size).ceil() as u32;
        let grid_dim = grid_dim.max(1).min(64); // cap per-axis to avoid huge grids
        let grid_dims = [grid_dim, grid_dim, grid_dim];
        let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
        let grid_origin = [-bounds, -bounds, -bounds];

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

        let tc = total_cells as usize;
        let pc = particle_count as usize;
        let zeroed_u32 = |label: &str, count: usize| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&vec![0u32; count]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        let cell_indices_buffer = zeroed_u32("SteeringSim/CellIndices", pc);
        let cell_counts_buffer = zeroed_u32("SteeringSim/CellCounts", tc);
        let cell_offsets_buffer = zeroed_u32("SteeringSim/CellOffsets", tc);
        let scatter_counters_buffer = zeroed_u32("SteeringSim/ScatterCounters", tc);
        let sorted_indices_buffer = zeroed_u32("SteeringSim/SortedIndices", pc);
        let num_blocks = ((tc + PREFIX_SUM_BLOCK_SIZE as usize - 1) / PREFIX_SUM_BLOCK_SIZE as usize).max(1);
        let block_sums_buffer = zeroed_u32("SteeringSim/BlockSums", num_blocks);

        let params_data = vec![0.0f32; ParamOffsets::BUFFER_SIZE];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringSim/Params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute passes
        let c = wgpu::ShaderStages::COMPUTE;
        let storage = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding, visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }
        };
        let storage_ro = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding, visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }
        };
        let uniform = |binding: u32| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding, visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }
        };

        macro_rules! buf {
            ($binding:expr, $buffer:expr) => {
                wgpu::BindGroupEntry { binding: $binding, resource: $buffer.as_entire_binding() }
            };
        }

        let make_pass = |label: &str, code: &str,
                         layout_entries: &[wgpu::BindGroupLayoutEntry],
                         bg_entries: Vec<wgpu::BindGroupEntry>| -> Pass
        {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}/Shader", label)),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{}/BGL", label)),
                entries: layout_entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}/Layout", label)),
                bind_group_layouts: &[&bgl],
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
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{}/BG", label)),
                layout: &bgl,
                entries: &bg_entries,
            });
            Pass { pipeline, bind_group }
        };

        // 1. Grid clear -- clears both cellCounts and scatterCounters
        let grid_clear = make_pass("GridClear", GRID_CLEAR_WGSL,
            &[storage(0), storage(1)],
            vec![buf!(0, cell_counts_buffer), buf!(1, scatter_counters_buffer)]);

        // 2. Grid assign -- hash particle positions into cells
        let grid_assign = make_pass("GridAssign", &with_params(GRID_ASSIGN_WGSL),
            &[storage_ro(0), storage(1), storage(2), uniform(3)],
            vec![
                buf!(0, positions_buffer),
                buf!(1, cell_indices_buffer),
                buf!(2, cell_counts_buffer),
                buf!(3, params_buffer),
            ]);

        // 3. Prefix sum local
        let prefix_sum_local = make_pass("PrefixSumLocal", PREFIX_SUM_LOCAL_WGSL,
            &[storage(0), storage(1), storage(2)],
            vec![
                buf!(0, cell_counts_buffer),
                buf!(1, cell_offsets_buffer),
                buf!(2, block_sums_buffer),
            ]);

        // 4. Prefix sum top
        let prefix_sum_top = make_pass("PrefixSumTop", PREFIX_SUM_TOP_WGSL,
            &[storage(0)],
            vec![buf!(0, block_sums_buffer)]);

        // 5. Prefix sum distribute
        let prefix_sum_distribute = make_pass("PrefixSumDistribute", PREFIX_SUM_DISTRIBUTE_WGSL,
            &[storage(0), storage(1)],
            vec![buf!(0, block_sums_buffer), buf!(1, cell_offsets_buffer)]);

        // 6. Scatter
        let scatter = make_pass("Scatter", &with_params(SCATTER_WGSL),
            &[storage_ro(0), storage_ro(1), storage(2), storage(3), uniform(4)],
            vec![
                buf!(0, cell_indices_buffer),
                buf!(1, cell_offsets_buffer),
                buf!(2, scatter_counters_buffer),
                buf!(3, sorted_indices_buffer),
                buf!(4, params_buffer),
            ]);

        // 7. Steering -- boids behavior
        // Note: cellCounts here is read as non-atomic u32 (prefix sum wrote it as plain u32)
        // but the buffer was created for atomic use. The prefix sum output (cellOffsets)
        // is read-only. cellCounts after prefix-sum still holds the original counts.
        // Actually, cellCounts is consumed by prefix-sum as input, and cellOffsets is the
        // exclusive scan output. The steering shader needs the original counts per cell
        // to know how many vehicles are in each cell. But prefix_sum_local reads cellCounts
        // as input and writes cellOffsets as output -- it does NOT modify cellCounts.
        // So cellCounts still holds the per-cell counts after the prefix sum. Good.
        let steering = make_pass("Steering", &with_params(STEERING_WGSL),
            &[storage(0), storage(1), uniform(2), storage_ro(3), storage_ro(4), storage_ro(5), storage_ro(6), storage_ro(7)],
            vec![
                buf!(0, positions_buffer),
                buf!(1, velocities_buffer),
                buf!(2, params_buffer),
                buf!(3, sorted_indices_buffer),
                buf!(4, cell_offsets_buffer),
                buf!(5, cell_counts_buffer),
                buf!(6, vehicle_indices_buffer),
                buf!(7, word_meta_buffer),
            ]);

        // 8. Verlet constraint
        let verlet = make_pass("Verlet", &with_params(VERLET_WGSL),
            &[storage(0), storage_ro(1), storage_ro(2), uniform(3)],
            vec![
                buf!(0, positions_buffer),
                buf!(1, word_meta_buffer),
                buf!(2, rest_lengths_buffer),
                buf!(3, params_buffer),
            ]);

        // 9. Integrate
        let integrate = make_pass("Integrate", &with_params(INTEGRATE_WGSL),
            &[storage(0), storage(1), uniform(2)],
            vec![
                buf!(0, positions_buffer),
                buf!(1, velocities_buffer),
                buf!(2, params_buffer),
            ]);

        // 10. Repulsion -- inter-particle repulsion using spatial hash
        let repulsion = make_pass("Repulsion", &with_params(REPULSION_WGSL),
            &[storage(0), uniform(1), storage_ro(2), storage_ro(3), storage_ro(4), storage_ro(5)],
            vec![
                buf!(0, positions_buffer),
                buf!(1, params_buffer),
                buf!(2, sorted_indices_buffer),
                buf!(3, cell_offsets_buffer),
                buf!(4, cell_counts_buffer),
                buf!(5, word_meta_buffer),
            ]);

        Self {
            device,
            queue,
            positions_buffer,
            velocities_buffer,
            word_meta_buffer,
            rest_lengths_buffer,
            vehicle_indices_buffer,
            cell_indices_buffer,
            cell_counts_buffer,
            cell_offsets_buffer,
            scatter_counters_buffer,
            sorted_indices_buffer,
            block_sums_buffer,
            params_buffer,
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
            params_data,
        }
    }

    /// Pack simulation parameters into the uniform buffer.
    fn pack_params(&mut self, params: &SteeringParams, dt: f32, time: f32, mouse: &MouseState) {
        let f = &mut self.params_data;

        f[ParamOffsets::DT] = dt;
        f[ParamOffsets::PARTICLE_COUNT] = f32::from_ne_bytes(self.particle_count.to_ne_bytes());
        f[ParamOffsets::VEHICLE_COUNT] = f32::from_ne_bytes(self.vehicle_count.to_ne_bytes());
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
        f[ParamOffsets::MOUSE_POS_X] = mouse.pos[0];
        f[ParamOffsets::MOUSE_POS_Y] = mouse.pos[1];
        f[ParamOffsets::MOUSE_DIR_X] = mouse.dir[0];
        f[ParamOffsets::MOUSE_DIR_Y] = mouse.dir[1];
        f[ParamOffsets::GRID_DIMS_X] = f32::from_ne_bytes(self.grid_dims[0].to_ne_bytes());
        f[ParamOffsets::GRID_DIMS_Y] = f32::from_ne_bytes(self.grid_dims[1].to_ne_bytes());
        f[ParamOffsets::GRID_DIMS_Z] = f32::from_ne_bytes(self.grid_dims[2].to_ne_bytes());
        f[ParamOffsets::CELL_SIZE] = params.separation_radius;
        f[ParamOffsets::GRID_ORIGIN_X] = self.grid_origin[0];
        f[ParamOffsets::GRID_ORIGIN_Y] = self.grid_origin[1];
        f[ParamOffsets::GRID_ORIGIN_Z] = self.grid_origin[2];
        f[ParamOffsets::TOTAL_CELLS] = f32::from_ne_bytes(self.total_cells.to_ne_bytes());
        f[ParamOffsets::VERLET_ITERATIONS] = f32::from_ne_bytes(params.verlet_iterations.to_ne_bytes());
        f[ParamOffsets::MOUSE_FORCE] = params.mouse_force;
        f[ParamOffsets::COHESION_STRENGTH] = params.cohesion_strength;
        f[ParamOffsets::ALIGNMENT_STRENGTH] = params.alignment_strength;
        f[ParamOffsets::ATTRACTOR_X] = params.attractor_pos[0];
        f[ParamOffsets::ATTRACTOR_Y] = params.attractor_pos[1];
        f[ParamOffsets::ATTRACTOR_Z] = params.attractor_pos[2];
        f[ParamOffsets::ATTRACTOR_STRENGTH] = params.attractor_strength;
        f[ParamOffsets::REPULSION_STRENGTH] = params.repulsion_strength;
        f[ParamOffsets::REPULSION_RADIUS] = params.repulsion_radius;
        f[ParamOffsets::MAX_PER_CELL] = f32::from_ne_bytes(params.max_per_cell.to_ne_bytes());
        f[ParamOffsets::_PAD3] = 0.0;
    }

    /// Run one simulation step: spatial hash build, steering, verlet, integrate.
    pub fn update(&mut self, params: &SteeringParams, dt: f32, time: f32, mouse: &MouseState) {
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

    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    pub fn vehicle_count(&self) -> u32 {
        self.vehicle_count
    }
}
