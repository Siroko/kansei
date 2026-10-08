// The moving colliders' substep pass (`FluidColliders`): capsules that push the particles out and
// carry them along. Concatenated after `sim-params.wgsl`. Shared with the TS engine
// (`src/simulations/fluid/FluidColliders.ts`).
struct Colliders {
    count: u32,
    restitution: f32,
    drag: f32,
    _pad: f32,
};
struct Capsule {
    a: vec3<f32>,
    radius: f32,
    b: vec3<f32>,
    expansion: f32,
    velocity_a: vec3<f32>,
    _pad1: f32,
    velocity_b: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<uniform> colliders: Colliders;
@group(0) @binding(4) var<storage, read> capsules: array<Capsule>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount || colliders.count == 0u) { return; }
    var pos = positions[idx].xyz;
    let v4 = velocities[idx];
    var vel = v4.xyz;
    var touched = false;

    for (var k = 0u; k < colliders.count; k++) {
        let c = capsules[k];
        let ab = c.b - c.a;
        let t = clamp(dot(pos - c.a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        let q = c.a + ab * t;
        let d = pos - q;
        let dist = length(d);
        if (dist >= c.radius) { continue; }
        // out onto the surface; a particle on the axis leaves upward
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if (dist > 1e-5) { n = d / dist; }
        pos = q + n * c.radius;
        // the collider's surface moves with its axis: relative to it, what goes in bounces back
        // and what slides along is dragged
        let surface = mix(c.velocity_a, c.velocity_b, t) + n * c.expansion;
        let rel = vel - surface;
        let vn = dot(rel, n);
        let vt = (rel - n * vn) * (1.0 - colliders.drag);
        vel = surface + vt + n * max(vn, -vn * colliders.restitution);
        touched = true;
    }

    if (touched) {
        positions[idx] = vec4<f32>(pos, positions[idx].w);
        // Position Based Fluids derive the velocity from the move, which would turn the push out of
        // a collider into a burst: w + 2 marks this velocity (the collider's) as the one to keep
        let held = select(0.0, 1.0, v4.w - 2.0 * floor(v4.w * 0.5) > 0.5);
        velocities[idx] = vec4<f32>(vel, held + select(0.0, 2.0, params.solver == 1u));
    }
}
