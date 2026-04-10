@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> originalPositions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(7) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(8) var<uniform> projectionMatrix: mat4x4<f32>;
@group(0) @binding(9) var<uniform> inverseViewMatrix: mat4x4<f32>;
@group(0) @binding(10) var<uniform> worldMatrix: mat4x4<f32>;

fn pressureFromDensity(density: f32) -> f32 {
    return (density - params.densityTarget) * params.pressureMultiplier;
}

fn nearPressureFromDensity(nearDensity: f32) -> f32 {
    return nearDensity * params.nearPressureMultiplier;
}

fn densityDerivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return v * params.spikyPow2DerivFactor;
}

fn nearDensityDerivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return v * v * params.spikyPow3DerivFactor;
}

fn viscosityKernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h * h - dist * dist;
    return v * v * v * params.poly6Factor;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    var vel = velocities[idx].xyz;
    let myDensity = densities[idx];
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;

    let pressure = pressureFromDensity(myDensity.x);
    let nearPressure = nearPressureFromDensity(myDensity.y);

    var pressureForce = vec3<f32>(0.0);
    var viscosityForce = vec3<f32>(0.0);

    let zStart = select(-1, 0, params.dimensions == 2u);
    let zEnd = select(1, 0, params.dimensions == 2u);

    for (var dz = zStart; dz <= zEnd; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = coord + vec3<i32>(dx, dy, dz);
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= vec3<i32>(params.gridDims))) {
                    continue;
                }

                let neighborCell = cellHash(neighborCoord, params);
                let cellStart = cellOffsets[neighborCell];
                let cellEnd = select(cellOffsets[neighborCell + 1u], params.particleCount, neighborCell + 1u >= params.totalCells);

                for (var j = cellStart; j < cellEnd; j++) {
                    let neighborIdx = sortedIndices[j];
                    if (neighborIdx == idx) { continue; }

                    let neighborPos = positions[neighborIdx].xyz;
                    let neighborVel = velocities[neighborIdx].xyz;
                    let neighborDensity = densities[neighborIdx];

                    let diff = pos - neighborPos;
                    let dist = length(diff);
                    if (dist < 0.0001) { continue; }

                    let dir = diff / dist;

                    // Pressure force
                    let neighborPressure = pressureFromDensity(neighborDensity.x);
                    let neighborNearPressure = nearPressureFromDensity(neighborDensity.y);
                    let sharedPressure = (pressure + neighborPressure) * 0.5;
                    let sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5;

                    pressureForce += dir * (densityDerivative(dist, h) * sharedPressure
                                          + nearDensityDerivative(dist, h) * sharedNearPressure);

                    // Viscosity force
                    let viscWeight = viscosityKernel(dist, h);
                    viscosityForce += (neighborVel - vel) * viscWeight;
                }
            }
        }
    }

    // Apply pressure + viscosity
    let safeDensity = max(myDensity.x, 0.001);
    vel += (pressureForce / safeDensity + viscosityForce * params.viscosity) * params.dt;

    // Gravity
    vel += params.gravity * params.dt;

    // Return to original position
    let origPos = originalPositions[idx].xyz;
    let toOrigin = origPos - pos;
    vel += toOrigin * params.returnToOriginStrength;

    // Mouse interaction (NDC-space proximity)
    if (params.mouseStrength > 0.001) {
        let projected = projectionMatrix * viewMatrix * worldMatrix * vec4<f32>(pos, 1.0);
        let ndc = projected.xyz / projected.w;
        var ndcMouse = params.mousePos;
        ndcMouse.y *= -1.0;
        let distToMouse = distance(ndcMouse, ndc.xy);

        if (distToMouse < params.mouseRadius * 2.0) {
            let nDist = distToMouse / (params.mouseRadius * 2.0);
            let displaceNDC = vec2<f32>(
                params.mouseDir.x * params.mouseForce * (1.0 - nDist) * -1.0,
                params.mouseDir.y * params.mouseForce * (1.0 - nDist)
            );
            let worldDisplace = (inverseViewMatrix * vec4<f32>(displaceNDC, 0.0, 0.0)).xyz;
            vel += worldDisplace * params.mouseStrength;
        }
    }

    // Damping
    vel *= params.damping;

    // Velocity limit
    let maxVel = 1200.0;
    let speed = length(vel);
    if (speed > maxVel) {
        vel = vel / speed * maxVel;
    }

    velocities[idx] = vec4<f32>(vel, 0.0);
}
