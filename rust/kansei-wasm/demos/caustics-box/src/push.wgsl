// The mouse's push (`push::FluidPush`): the particles in a vertical column round the point under
// the cursor are made to follow the mouse's motion, each substep `strength` of the way from their
// velocity to the push's, less toward the column's edge. Concatenated after the simulation's
// `SimParams` (`SIM_PARAMS_WGSL`).
struct Push {
    // the column's axis (x, z) and its bottom and top (y), simulation units
    center: vec2<f32>,
    bottom: f32,
    top: f32,
    velocity: vec3<f32>,
    radius: f32,
    strength: f32,
    on: u32,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<uniform> push: Push;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount || push.on == 0u) { return; }
    let p = positions[idx].xyz;
    if (p.y < push.bottom || p.y > push.top) { return; }
    let d = p.xz - push.center;
    let q = dot(d, d) / (push.radius * push.radius);
    if (q >= 1.0) { return; }
    let w = (1.0 - q) * (1.0 - q) * push.strength;
    let v4 = velocities[idx];
    let v = mix(v4.xyz, push.velocity, w);
    // Position Based Fluids derive the velocity from the move: w + 2 keeps this one (as the
    // colliders do), the held flag (w's low bit) kept
    let held = select(0.0, 1.0, v4.w - 2.0 * floor(v4.w * 0.5) > 0.5);
    velocities[idx] = vec4<f32>(v, held + select(0.0, 2.0, params.solver == 1u));
}
