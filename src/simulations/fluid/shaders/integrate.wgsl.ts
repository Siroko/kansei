import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    var pos = positions[idx];
    var vel = velocities[idx];

    // Integrate
    pos = vec4<f32>(pos.xyz + vel.xyz * params.dt, pos.w);

    // Boundary collision (reflect)
    let bMin = params.worldBoundsMin;
    let bMax = params.worldBoundsMax;
    let bounce = -0.5;

    if (pos.x < bMin.x) { pos.x = bMin.x; vel.x *= bounce; }
    if (pos.x > bMax.x) { pos.x = bMax.x; vel.x *= bounce; }
    if (pos.y < bMin.y) { pos.y = bMin.y; vel.y *= bounce; }
    if (pos.y > bMax.y) { pos.y = bMax.y; vel.y *= bounce; }

    if (params.dimensions == 3u) {
        if (pos.z < bMin.z) { pos.z = bMin.z; vel.z *= bounce; }
        if (pos.z > bMax.z) { pos.z = bMax.z; vel.z *= bounce; }
    } else {
        pos.z = 0.0;
        vel.z = 0.0;
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}
`;
