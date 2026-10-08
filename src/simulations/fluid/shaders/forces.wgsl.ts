import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

// One thread per *sorted* slot (see density.wgsl). Inputs are the cell-ordered copies
// written by scatter; the result is written back to the particle's original index so
// 'velocities' stays in original order for everything outside the sim.
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sortedVelocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> densities: array<vec2<f32>>; // sorted order
@group(0) @binding(3) var<storage, read_write> originalPositions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(7) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(8) var<uniform> projectionMatrix: mat4x4<f32>;
@group(0) @binding(9) var<uniform> inverseViewMatrix: mat4x4<f32>;
@group(0) @binding(10) var<uniform> worldMatrix: mat4x4<f32>;
@group(0) @binding(11) var<storage, read_write> velocities: array<vec4<f32>>; // original order (output)

fn pressureFromDensity(density: f32) -> f32 {
    return (density - params.densityTarget) * params.pressureMultiplier;
}

fn nearPressureFromDensity(nearDensity: f32) -> f32 {
    return nearDensity * params.nearPressureMultiplier;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }
    let idx = sortedIndices[slot];

    let pos = sortedPositions[slot].xyz;
    var vel = sortedVelocities[slot].xyz;
    let myDensity = densities[slot];
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;
    let h2 = h * h;

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
                    if (j == slot) { continue; }

                    let diff = pos - sortedPositions[j].xyz;
                    let d2 = dot(diff, diff);
                    // One range test per pair; only then touch velocity/density.
                    if (d2 >= h2 || d2 < 1e-8) { continue; }

                    let neighborVel = sortedVelocities[j].xyz;
                    let neighborDensity = densities[j];

                    let dist = sqrt(d2);
                    let dir = diff / dist;
                    let v = h - dist;

                    // Pressure force: spiky derivatives (h-r) and (h-r)^2, normalised once below
                    let neighborPressure = pressureFromDensity(neighborDensity.x);
                    let neighborNearPressure = nearPressureFromDensity(neighborDensity.y);
                    let sharedPressure = (pressure + neighborPressure) * 0.5;
                    let sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5;
                    pressureForce += dir * (v * params.spikyPow2DerivFactor * sharedPressure
                                          + v * v * params.spikyPow3DerivFactor * sharedNearPressure);

                    // Viscosity: poly6 (h²-r²)^3, normalised once below
                    let q = h2 - d2;
                    viscosityForce += (neighborVel - vel) * (q * q * q);
                }
            }
        }
    }

    // Apply pressure + viscosity
    let safeDensity = max(myDensity.x, 0.001);
    vel += (pressureForce / safeDensity + viscosityForce * (params.poly6Factor * params.viscosity)) * params.dt;

    // Gravity (directional or radial toward params.gravityCenter)
    if (params.radialGravity > 0.5) {
        let toCenter = params.gravityCenter - pos;
        let dist = length(toCenter) + 0.0001;
        let dir = toCenter / dist;
        let g_mag = length(params.gravity);
        vel += dir * g_mag * params.dt;
    } else {
        vel += params.gravity * params.dt;
    }

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
`;
