// Integrate: apply velocity to position, clamp to bounds, apply damping.
// Dispatched over particleCount.

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    var vel = velocities[idx].xyz;
    var pos = positions[idx].xyz;

    // Apply velocity
    pos += vel * params.dt;

    // Apply damping
    vel *= params.damping;

    // Clamp to bounds
    let bs = params.boundsSize;
    pos = clamp(pos, vec3<f32>(-bs), vec3<f32>(bs));

    positions[idx] = vec4<f32>(pos, 1.0);
    velocities[idx] = vec4<f32>(vel, 0.0);
}
