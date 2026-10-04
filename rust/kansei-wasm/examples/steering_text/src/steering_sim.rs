// SteeringSimulation -- compute-based 3D boids on kansei's neighbour grid + verlet constraints.
// Engine-adjacent code: accesses renderer.device()/queue() internally.

use kansei_core::simulations::grid::{GridLayout, NeighbourGrid, NeighbourGridOptions, NEIGHBOUR_GRID_WGSL};
use wgpu::util::DeviceExt;

use crate::text_data::ParticleData;

/// Cells in the neighbour grid at most (64³), which bounds its memory when the bounds grow.
const MAX_GRID_CELLS: u32 = 1 << 18;

const SIM_PARAMS_WGSL: &str = include_str!("shaders/sim_params.wgsl");
const STEERING_WGSL: &str = include_str!("shaders/steering.wgsl");
const VERLET_WGSL: &str = include_str!("shaders/verlet.wgsl");
const INTEGRATE_WGSL: &str = include_str!("shaders/integrate.wgsl");
const REPULSION_WGSL: &str = include_str!("shaders/repulsion.wgsl");

/// Prepend the SimParams struct and the neighbour grid's helpers to a shader.
fn with_params(shader: &str) -> String {
    format!("{SIM_PARAMS_WGSL}\n{NEIGHBOUR_GRID_WGSL}\n{shader}")
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

/// The neighbour grid over the bounds: cells at least as wide as both search radii (the 3×3×3
/// neighbourhood must reach every neighbour), wider if the bounds would need more than
/// MAX_GRID_CELLS.
fn grid_layout(params: &SteeringParams) -> GridLayout {
    let bounds = params.bounds_size.max(1.0);
    GridLayout::covering([-bounds; 3], [bounds; 3], params.separation_radius.max(params.repulsion_radius).max(1e-3), MAX_GRID_CELLS)
}

// ── SimParams uniform layout (36 f32 = 144 bytes, 9 vec4 blocks) ──
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
    const VERLET_ITERATIONS: usize = 13;
    const MOUSE_FORCE: usize = 14;
    const COHESION_STRENGTH: usize = 15;
    const ALIGNMENT_STRENGTH: usize = 16;
    const ATTRACTOR: usize = 17; // x, y, z
    const ATTRACTOR_STRENGTH: usize = 20;
    const REPULSION_STRENGTH: usize = 21;
    const REPULSION_RADIUS: usize = 22;
    const MAX_PER_CELL: usize = 23;
    const MOUSE_RAY_ORIGIN: usize = 24; // x, y, z
    const MOUSE_RAY_DIR: usize = 27; // x, y, z
    const MOUSE_DIR: usize = 30; // x, y, z; 33-35 pad
    const BUFFER_SIZE: usize = 36;
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
    params_buffer: wgpu::Buffer,
    /// Every letter sorted by cell each step, for the steering and repulsion searches.
    grid: NeighbourGrid,

    // Compute passes
    steering: Pass,
    verlet: Pass,
    integrate: Pass,
    repulsion: Pass,

    // Counts
    particle_count: u32,
    vehicle_count: u32,

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

        let mut grid = NeighbourGrid::new(&device, &queue, &NeighbourGridOptions {
            label: "SteeringSim/Grid",
            capacity: particle_count,
            layout: grid_layout(params),
            positions: &positions_buffer,
            sorted_copies: &[],
        });
        grid.set_count(particle_count);

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
        let steering = Pass::new(&device, "Steering", &with_params(STEERING_WGSL),
            &[storage(0), storage(1), uniform(2), storage_ro(3), storage_ro(4), storage_ro(5), storage_ro(6), storage_ro(7), uniform(8)]);
        let verlet = Pass::new(&device, "Verlet", &with_params(VERLET_WGSL),
            &[storage(0), storage_ro(1), storage_ro(2), uniform(3)]);
        let integrate = Pass::new(&device, "Integrate", &with_params(INTEGRATE_WGSL),
            &[storage(0), storage(1), uniform(2)]);
        let repulsion = Pass::new(&device, "Repulsion", &with_params(REPULSION_WGSL),
            &[storage(0), uniform(1), storage_ro(2), storage_ro(3), storage_ro(4), storage_ro(5), uniform(6)]);

        let mut sim = Self {
            device,
            queue,
            positions_buffer,
            velocities_buffer,
            word_meta_buffer,
            rest_lengths_buffer,
            vehicle_indices_buffer,
            params_buffer,
            grid,
            steering,
            verlet,
            integrate,
            repulsion,
            particle_count,
            vehicle_count,
            params_data: vec![0.0f32; ParamOffsets::BUFFER_SIZE],
        };
        sim.bind_passes();
        sim
    }

    /// (Re)create every pass's bind group from the current buffers.
    fn bind_passes(&mut self) {
        let d = &self.device;
        let g = &self.grid;
        // boids: reads each cell's range (cellOffsets) and size (cellCounts)
        self.steering.bind(d, &[
            &self.positions_buffer, &self.velocities_buffer, &self.params_buffer, g.sorted_indices(),
            g.cell_offsets(), g.cell_counts(), &self.vehicle_indices_buffer, &self.word_meta_buffer, g.params_buffer(),
        ]);
        self.verlet.bind(d, &[&self.positions_buffer, &self.word_meta_buffer, &self.rest_lengths_buffer, &self.params_buffer]);
        self.integrate.bind(d, &[&self.positions_buffer, &self.velocities_buffer, &self.params_buffer]);
        self.repulsion.bind(d, &[
            &self.positions_buffer, &self.params_buffer, g.sorted_indices(),
            g.cell_offsets(), g.cell_counts(), &self.word_meta_buffer, g.params_buffer(),
        ]);
    }

    /// Fit the neighbour grid to the current radii and bounds; a new cell count replaces its
    /// per-cell buffers, which the searches bind.
    fn fit_grid(&mut self, params: &SteeringParams) {
        if self.grid.set_layout(grid_layout(params)) {
            self.bind_passes();
        }
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
            f[ParamOffsets::ATTRACTOR + axis] = params.attractor_pos[axis];
            f[ParamOffsets::MOUSE_RAY_ORIGIN + axis] = mouse.ray_origin[axis];
            f[ParamOffsets::MOUSE_RAY_DIR + axis] = mouse.ray_dir[axis];
            f[ParamOffsets::MOUSE_DIR + axis] = mouse.dir[axis];
        }
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

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("SteeringSim") },
        );

        {
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None },
            );

            // every letter into its cell
            self.grid.encode(&mut cp);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_validate_and_the_params_match() {
        for (name, code) in [("steering", STEERING_WGSL), ("verlet", VERLET_WGSL), ("integrate", INTEGRATE_WGSL), ("repulsion", REPULSION_WGSL)] {
            let code = with_params(code);
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let span = module.types.iter().find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "SimParams" => Some(*span as usize),
                _ => None,
            });
            assert_eq!(span, Some(ParamOffsets::BUFFER_SIZE * 4), "{name}");
        }
    }
}
