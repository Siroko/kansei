export interface FluidSimulationOptions {
    maxParticles: number;
    dimensions: 2 | 3;
    smoothingRadius: number;
    pressureMultiplier: number;
    nearPressureMultiplier: number;
    densityTarget: number;
    viscosity: number;
    damping: number;
    gravity: [number, number, number];
    returnToOriginStrength: number;
    mouseRadius: number;
    mouseForce: number;
    substeps: number;
    worldBoundsPadding: number;
}

export const DEFAULT_OPTIONS: FluidSimulationOptions = {
    maxParticles: 10000,
    dimensions: 2,
    smoothingRadius: 1.0,
    pressureMultiplier: 10.0,
    nearPressureMultiplier: 18.0,
    densityTarget: 1.5,
    viscosity: 0.3,
    damping: 0.998,
    gravity: [0, 0, 0],
    returnToOriginStrength: 0.002,
    mouseRadius: 0.5,
    mouseForce: 300.0,
    substeps: 3,
    worldBoundsPadding: 0.2,
};

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

// SimParams uniform buffer layout (160 bytes = 40 f32s)
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
    _pad:                    39,  // f32 padding (struct size must be 16-byte multiple)
    BUFFER_SIZE:             40,  // total f32 count
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
