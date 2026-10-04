struct SimParams {
    dt: f32,
    particleCount: u32,
    vehicleCount: u32,
    separationStrength: f32,

    separationRadius: f32,
    wanderStrength: f32,
    wanderSpeed: f32,
    maxSpeed: f32,

    maxForce: f32,
    damping: f32,
    boundsSize: f32,
    time: f32,

    mouseStrength: f32,
    gridDimsX: u32,
    gridDimsY: u32,
    gridDimsZ: u32,

    cellSize: f32,
    gridOriginX: f32,
    gridOriginY: f32,
    gridOriginZ: f32,

    totalCells: u32,
    verletIterations: u32,
    mouseForce: f32,
    cohesionStrength: f32,

    alignmentStrength: f32,
    attractorX: f32,
    attractorY: f32,
    attractorZ: f32,

    attractorStrength: f32,
    repulsionStrength: f32,
    repulsionRadius: f32,
    maxPerCell: u32,

    // the cursor in world space: the ray from the camera through it, and its motion
    mouseRayOriginX: f32,
    mouseRayOriginY: f32,
    mouseRayOriginZ: f32,
    mouseRayDirX: f32,

    mouseRayDirY: f32,
    mouseRayDirZ: f32,
    mouseDirX: f32,
    mouseDirY: f32,

    mouseDirZ: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

fn getCellCoord(pos: vec3<f32>, params: SimParams) -> vec3<i32> {
    let origin = vec3<f32>(params.gridOriginX, params.gridOriginY, params.gridOriginZ);
    return vec3<i32>(floor((pos - origin) / params.cellSize));
}

fn cellHash(coord: vec3<i32>, params: SimParams) -> u32 {
    let dims = vec3<i32>(i32(params.gridDimsX), i32(params.gridDimsY), i32(params.gridDimsZ));
    let c = clamp(coord, vec3<i32>(0), dims - vec3<i32>(1));
    return u32(c.z) * params.gridDimsX * params.gridDimsY + u32(c.y) * params.gridDimsX + u32(c.x);
}
