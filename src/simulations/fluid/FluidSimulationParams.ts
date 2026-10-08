/** The solver a `FluidSimulation` steps with: Smoothed Particle Hydrodynamics. */
export type FluidSolver = 'sph';

/** The `solver` word in `SimParams`, as the Rust engine packs it (0 = SPH, 1 = PBF). */
export const SOLVER_WORD: Record<FluidSolver, number> = { sph: 0 };

export interface FluidSimulationOptions {
    /** How many particles the buffers hold: the most there can be (see `FluidSimulation.emit`). */
    maxParticles: number;
    dimensions: 2 | 3;
    smoothingRadius: number;
    pressureMultiplier: number;
    nearPressureMultiplier: number;
    densityTarget: number;
    viscosity: number;
    damping: number;
    gravity: [number, number, number];
    /** World-space center for radial gravity (only used when `radialGravity` is true). */
    gravityCenter?: [number, number, number];
    /** If true, gravity points toward `gravityCenter` with magnitude |gravity|. */
    radialGravity?: boolean;
    returnToOriginStrength: number;
    mouseRadius: number;
    mouseForce: number;
    substeps: number;
    worldBoundsPadding: number;
    /**
     * How much of the pressure below the rest density acts (a pull between particles): 1 as
     * computed, less to weaken it. The pull is what strings a sparse free surface into
     * filaments (SPH's tensile instability); the near pressure still keeps particles apart.
     */
    negativePressureScale: number;
    /** The solver (SPH only for now). */
    solver: FluidSolver;
}

export const DEFAULT_OPTIONS: FluidSimulationOptions = {
    maxParticles: 10000,
    dimensions: 2,
    smoothingRadius: 1.0,
    pressureMultiplier: 2.0,
    nearPressureMultiplier: 15.0,
    densityTarget: 7.4,
    viscosity: 1.0,
    damping: 0.999,
    gravity: [0, -19.6, 0],
    gravityCenter: [0, 0, 0],
    radialGravity: false,
    returnToOriginStrength: 0.0,
    mouseRadius: 0.1,
    mouseForce: 1630.0,
    substeps: 3,
    worldBoundsPadding: 0.2,
    negativePressureScale: 1.0,
    solver: 'sph',
};

/**
 * `options`, tuned for `baseCount` particles, for `count` particles filling the same volume:
 * the smoothing radius scales by (count / baseCount)^-1/3 so a neighbourhood holds as many
 * particles, the near pressure with the radius, and the density target by the count ratio over
 * the radius. Pressure and time step are left: how stiff a fluid stays stable at a size is for
 * the caller to tune.
 */
export function scaledToCount(options: FluidSimulationOptions, baseCount: number, count: number): FluidSimulationOptions {
    const ratio = Math.max(count, 1) / Math.max(baseCount, 1);
    const radius = Math.pow(ratio, -1 / 3);
    return {
        ...options,
        maxParticles: count,
        smoothingRadius: options.smoothingRadius * radius,
        nearPressureMultiplier: options.nearPressureMultiplier * radius,
        densityTarget: options.densityTarget * ratio / radius,
    };
}

export interface FluidSimulationPreset extends Partial<FluidSimulationOptions> {
    name: string;
}

export const PRESETS: Record<string, FluidSimulationPreset> = {
    water: {
        name: 'Water',
        pressureMultiplier: 10,
        nearPressureMultiplier: 18,
        viscosity: 0.3,
        damping: 0.998,
        densityTarget: 1.5,
        returnToOriginStrength: 0.002,
        gravity: [0, -9.8, 0],
    },
    honey: {
        name: 'Viscous Honey',
        pressureMultiplier: 3,
        nearPressureMultiplier: 8,
        viscosity: 0.9,
        damping: 0.99,
        densityTarget: 3.0,
        returnToOriginStrength: 0.005,
        gravity: [0, -2.0, 0],
    },
    gas: {
        name: 'Gas',
        pressureMultiplier: 20,
        nearPressureMultiplier: 30,
        viscosity: 0.05,
        damping: 0.995,
        densityTarget: 0.5,
        returnToOriginStrength: 0.001,
        gravity: [0, 0, 0],
    },
    zeroG: {
        name: 'Zero-G Blob',
        pressureMultiplier: 8,
        nearPressureMultiplier: 14,
        viscosity: 0.6,
        damping: 0.997,
        densityTarget: 2.0,
        returnToOriginStrength: 0.0,
        gravity: [0, 0, 0],
    },
};

// SimParams uniform buffer layout (192 bytes = 48 f32s; sim-params.wgsl, Rust `ParamOffsets`)
// Fields marked [u32] must be written via Uint32Array view
export const PARAMS = {
    dt:                       0,  // f32
    particleCount:            1,  // [u32]
    dimensions:               2,  // [u32]
    smoothingRadius:          3,  // f32
    pressureMultiplier:       4,  // f32
    densityTarget:            5,  // f32
    nearPressureMultiplier:   6,  // f32
    viscosity:                7,  // f32
    damping:                  8,  // f32
    returnToOriginStrength:   9,  // f32
    mouseStrength:           10,  // f32
    mouseRadius:             11,  // f32
    // --- 16-byte aligned boundary (offset 48) ---
    gravityX:                12,  // vec3<f32> gravity
    gravityY:                13,
    gravityZ:                14,
    mouseForce:              15,  // f32 (packed after vec3)
    // --- 8-byte aligned boundary (offset 64) ---
    mousePosX:               16,  // vec2<f32> mousePos
    mousePosY:               17,
    mouseDirX:               18,  // vec2<f32> mouseDir
    mouseDirY:               19,
    // --- 16-byte aligned boundary (offset 80) ---
    gridDimsX:               20,  // [u32] vec3<u32> gridDims
    gridDimsY:               21,  // [u32]
    gridDimsZ:               22,  // [u32]
    cellSize:                23,  // f32
    // --- 16-byte aligned boundary (offset 96) ---
    gridOriginX:             24,  // vec3<f32> gridOrigin
    gridOriginY:             25,
    gridOriginZ:             26,
    totalCells:              27,  // [u32]
    // --- 16-byte aligned boundary (offset 112) ---
    worldBoundsMinX:         28,  // vec3<f32> worldBoundsMin
    worldBoundsMinY:         29,
    worldBoundsMinZ:         30,
    poly6Factor:             31,  // f32
    // --- 16-byte aligned boundary (offset 128) ---
    worldBoundsMaxX:         32,  // vec3<f32> worldBoundsMax
    worldBoundsMaxY:         33,
    worldBoundsMaxZ:         34,
    spikyPow2Factor:         35,  // f32
    // --- remaining kernel factors ---
    spikyPow3Factor:         36,  // f32
    spikyPow2DerivFactor:    37,  // f32
    spikyPow3DerivFactor:    38,  // f32
    negativePressureScale:   39,  // f32
    // --- 16-byte aligned boundary (offset 160) ---
    gravityCenterX:          40,  // vec3<f32> gravityCenter
    gravityCenterY:          41,
    gravityCenterZ:          42,
    radialGravity:           43,  // f32 (0 or 1)
    // --- 16-byte aligned boundary (offset 176) ---
    solver:                  44,  // [u32] 0 = SPH, 1 = PBF (then 3 words of padding)
    BUFFER_SIZE:             48,  // total f32 count
} as const;

export function computeKernelFactors2D(h: number) {
    const pi = Math.PI;
    return {
        poly6:           4.0 / (pi * Math.pow(h, 8)),
        spikyPow2:       6.0 / (pi * Math.pow(h, 4)),
        spikyPow3:       10.0 / (pi * Math.pow(h, 5)),
        spikyPow2Deriv:  12.0 / (Math.pow(h, 4) * pi),
        spikyPow3Deriv:  30.0 / (Math.pow(h, 5) * pi),
    };
}

export function computeKernelFactors3D(h: number) {
    const pi = Math.PI;
    return {
        poly6:           315.0 / (64.0 * pi * Math.pow(h, 9)),
        spikyPow2:       15.0 / (pi * Math.pow(h, 6)),
        spikyPow3:       15.0 / (pi * Math.pow(h, 6)),
        spikyPow2Deriv:  45.0 / (pi * Math.pow(h, 6)),
        spikyPow3Deriv:  45.0 / (pi * Math.pow(h, 6)),
    };
}
