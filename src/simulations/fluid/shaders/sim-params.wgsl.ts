export const simParamsStruct = /* wgsl */`
struct SimParams {
    dt: f32,
    particleCount: u32,
    dimensions: u32,
    smoothingRadius: f32,

    pressureMultiplier: f32,
    densityTarget: f32,
    nearPressureMultiplier: f32,
    viscosity: f32,

    damping: f32,
    returnToOriginStrength: f32,
    mouseStrength: f32,
    mouseRadius: f32,

    gravity: vec3<f32>,
    mouseForce: f32,

    mousePos: vec2<f32>,
    mouseDir: vec2<f32>,

    gridDims: vec3<u32>,
    cellSize: f32,

    gridOrigin: vec3<f32>,
    totalCells: u32,

    worldBoundsMin: vec3<f32>,
    poly6Factor: f32,

    worldBoundsMax: vec3<f32>,
    spikyPow2Factor: f32,

    spikyPow3Factor: f32,
    spikyPow2DerivFactor: f32,
    spikyPow3DerivFactor: f32,
    _pad: f32,

    gravityCenter: vec3<f32>,
    radialGravity: f32, // 0 = directional, 1 = radial toward gravityCenter
};

fn getCellCoord(pos: vec3<f32>, params: SimParams) -> vec3<i32> {
    return vec3<i32>(floor((pos - params.gridOrigin) / params.cellSize));
}

fn cellHash(coord: vec3<i32>, params: SimParams) -> u32 {
    let c = clamp(coord, vec3<i32>(0), vec3<i32>(params.gridDims) - vec3<i32>(1));
    return u32(c.z) * params.gridDims.x * params.gridDims.y + u32(c.y) * params.gridDims.x + u32(c.x);
}
`;
