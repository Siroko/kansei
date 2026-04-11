/// Fluid simulation configuration.
#[derive(Debug, Clone)]
pub struct FluidSimulationOptions {
    pub max_particles: u32,
    pub dimensions: u32,
    pub smoothing_radius: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub density_target: f32,
    pub viscosity: f32,
    pub damping: f32,
    pub gravity: [f32; 3],
    /// World-space center for radial gravity (only used when `radial_gravity` is true).
    pub gravity_center: [f32; 3],
    /// If true, gravity points toward `gravity_center` with magnitude `|gravity|`.
    pub radial_gravity: bool,
    pub return_to_origin_strength: f32,
    pub mouse_radius: f32,
    pub mouse_force: f32,
    pub substeps: u32,
    pub world_bounds_padding: f32,
}

pub const DEFAULT_OPTIONS: FluidSimulationOptions = FluidSimulationOptions {
    max_particles: 10000,
    dimensions: 2,
    smoothing_radius: 1.0,
    pressure_multiplier: 10.0,
    near_pressure_multiplier: 18.0,
    density_target: 1.5,
    viscosity: 0.3,
    damping: 0.998,
    gravity: [0.0, -9.8, 0.0],
    gravity_center: [0.0, 0.0, 0.0],
    radial_gravity: false,
    return_to_origin_strength: 0.0,
    mouse_radius: 0.1,
    mouse_force: 500.0,
    substeps: 3,
    world_bounds_padding: 0.2,
};

/// Offsets into the packed params uniform buffer (matches SimParams WGSL struct).
pub struct ParamOffsets;
impl ParamOffsets {
    pub const DT: usize = 0;
    pub const PARTICLE_COUNT: usize = 1;
    pub const DIMENSIONS: usize = 2;
    pub const SMOOTHING_RADIUS: usize = 3;
    pub const PRESSURE_MULTIPLIER: usize = 4;
    pub const DENSITY_TARGET: usize = 5;
    pub const NEAR_PRESSURE_MULTIPLIER: usize = 6;
    pub const VISCOSITY: usize = 7;
    pub const DAMPING: usize = 8;
    pub const RETURN_TO_ORIGIN_STRENGTH: usize = 9;
    pub const MOUSE_STRENGTH: usize = 10;
    pub const MOUSE_RADIUS: usize = 11;
    pub const GRAVITY_X: usize = 12;
    pub const GRAVITY_Y: usize = 13;
    pub const GRAVITY_Z: usize = 14;
    pub const MOUSE_FORCE: usize = 15;
    pub const MOUSE_POS_X: usize = 16;
    pub const MOUSE_POS_Y: usize = 17;
    pub const MOUSE_DIR_X: usize = 18;
    pub const MOUSE_DIR_Y: usize = 19;
    pub const GRID_DIMS_X: usize = 20;
    pub const GRID_DIMS_Y: usize = 21;
    pub const GRID_DIMS_Z: usize = 22;
    pub const CELL_SIZE: usize = 23;
    pub const GRID_ORIGIN_X: usize = 24;
    pub const GRID_ORIGIN_Y: usize = 25;
    pub const GRID_ORIGIN_Z: usize = 26;
    pub const TOTAL_CELLS: usize = 27;
    pub const WORLD_BOUNDS_MIN_X: usize = 28;
    pub const WORLD_BOUNDS_MIN_Y: usize = 29;
    pub const WORLD_BOUNDS_MIN_Z: usize = 30;
    pub const POLY6_FACTOR: usize = 31;
    pub const WORLD_BOUNDS_MAX_X: usize = 32;
    pub const WORLD_BOUNDS_MAX_Y: usize = 33;
    pub const WORLD_BOUNDS_MAX_Z: usize = 34;
    pub const SPIKY_POW2_FACTOR: usize = 35;
    pub const SPIKY_POW3_FACTOR: usize = 36;
    pub const SPIKY_POW2_DERIV_FACTOR: usize = 37;
    pub const SPIKY_POW3_DERIV_FACTOR: usize = 38;
    pub const PAD: usize = 39;
    pub const GRAVITY_CENTER_X: usize = 40;
    pub const GRAVITY_CENTER_Y: usize = 41;
    pub const GRAVITY_CENTER_Z: usize = 42;
    pub const RADIAL_GRAVITY: usize = 43;
    pub const BUFFER_SIZE: usize = 44;
}

/// Compute SPH kernel factors for 2D.
pub fn compute_kernel_factors_2d(h: f32) -> KernelFactors {
    let pi = std::f32::consts::PI;
    KernelFactors {
        poly6: 4.0 / (pi * h.powi(8)),
        spiky_pow2: 6.0 / (pi * h.powi(4)),
        spiky_pow3: 10.0 / (pi * h.powi(5)),
        spiky_pow2_deriv: 12.0 / (pi * h.powi(4)),
        spiky_pow3_deriv: 30.0 / (pi * h.powi(5)),
    }
}

/// Compute SPH kernel factors for 3D.
pub fn compute_kernel_factors_3d(h: f32) -> KernelFactors {
    let pi = std::f32::consts::PI;
    KernelFactors {
        poly6: 315.0 / (64.0 * pi * h.powi(9)),
        spiky_pow2: 15.0 / (pi * h.powi(6)),
        spiky_pow3: 15.0 / (pi * h.powi(6)),
        spiky_pow2_deriv: 45.0 / (pi * h.powi(6)),
        spiky_pow3_deriv: 45.0 / (pi * h.powi(6)),
    }
}

pub struct KernelFactors {
    pub poly6: f32,
    pub spiky_pow2: f32,
    pub spiky_pow3: f32,
    pub spiky_pow2_deriv: f32,
    pub spiky_pow3_deriv: f32,
}
