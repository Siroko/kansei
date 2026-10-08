// The planar container's substep pass (`FluidContainer`): walls along a closed outline on XZ and a
// floor under it, from a grid of (signed distance, floor height) nodes. Concatenated after
// `sim-params.wgsl`. Shared with the TS engine (`src/simulations/fluid/FluidContainer.ts`).
struct Container {
    origin: vec2<f32>,
    cell: f32,
    wall_offset: f32,
    dims: vec2<u32>,
    margin: f32,
    restitution: f32,
    friction: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<uniform> container: Container;
// (signed distance to the outline, floor height) per node, x fastest
@group(0) @binding(4) var<storage, read> nodes: array<vec2<f32>>;

fn node(i: i32, j: i32) -> vec2<f32> {
    let d = vec2<i32>(container.dims);
    let c = clamp(vec2<i32>(i, j), vec2<i32>(0), d - vec2<i32>(1));
    return nodes[c.y * d.x + c.x];
}

// Bilinear (distance, floor) at p (x, z), clamped to the grid.
fn sample_field(p: vec2<f32>) -> vec2<f32> {
    let g = clamp((p - container.origin) / container.cell, vec2<f32>(0.0), vec2<f32>(container.dims - vec2<u32>(1u)));
    let i = vec2<i32>(floor(g));
    let t = g - floor(g);
    let a = mix(node(i.x, i.y), node(i.x + 1, i.y), t.x);
    let b = mix(node(i.x, i.y + 1), node(i.x + 1, i.y + 1), t.x);
    return mix(a, b, t.y);
}

// d(distance, floor)/dx and d/dz, by central differences half a cell apart.
fn gradient(p: vec2<f32>) -> mat2x2<f32> {
    let h = container.cell * 0.5;
    let dx = (sample_field(p + vec2<f32>(h, 0.0)) - sample_field(p - vec2<f32>(h, 0.0))) / (2.0 * h);
    let dz = (sample_field(p + vec2<f32>(0.0, h)) - sample_field(p - vec2<f32>(0.0, h))) / (2.0 * h);
    return mat2x2<f32>(dx, dz);
}

// The velocity after a contact on a surface of normal n (toward the fluid): what went into it
// bounces back by the restitution, what slid along it loses the friction.
fn collide(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let vn = dot(v, n);
    if (vn >= 0.0) { return v; }
    let vt = v - n * vn;
    return vt * (1.0 - container.friction) - n * vn * container.restitution;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }
    var pos = positions[idx].xyz;
    let v4 = velocities[idx];
    var vel = v4.xyz;

    // walls: out past them, back along the distance's gradient
    let over = sample_field(pos.xz).x - container.wall_offset + container.margin;
    if (over > 0.0) {
        let g = gradient(pos.xz);
        let outward = vec2<f32>(g[0].x, g[1].x);
        let len = length(outward);
        if (len > 1e-5) {
            let n2 = outward / len;
            pos = vec3<f32>(pos.x - n2.x * over, pos.y, pos.z - n2.y * over);
            vel = collide(vel, vec3<f32>(-n2.x, 0.0, -n2.y));
        }
    }

    // floor: out along its normal (not straight up: on a slope that would undo every move down
    // it, and a position-based solver, whose velocity is the move, would never slide), and off
    // its slope
    let floor_y = sample_field(pos.xz).y + container.margin;
    if (pos.y < floor_y) {
        let g = gradient(pos.xz);
        let n = normalize(vec3<f32>(-g[0].y, 1.0, -g[1].y));
        pos += n * ((floor_y - pos.y) * n.y);
        pos.y = max(pos.y, sample_field(pos.xz).y + container.margin * 0.5);
        vel = collide(vel, n);
    }

    positions[idx] = vec4<f32>(pos, positions[idx].w);
    velocities[idx] = vec4<f32>(vel, v4.w);
}
