// After the sim (particles_sim.wgsl): the instances past the live ones (spawned[2], compacted
// first) as no particle at all, far below the scene (out of the ray tracing grid's box) and of no
// size, so a draw of more instances than are alive (the CPU's count is a bound a few frames
// late) shows nothing stale.

struct Instance {
    placement : vec4f,
    rotation  : vec4f,
    motion    : vec4f,
    surface   : vec4u,
}

@group(0) @binding(0) var<storage, read> counts : array<u32, 4>;
@group(0) @binding(1) var<storage, read_write> instances : array<Instance>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&instances) || i < counts[2]) {
        return;
    }
    var none: Instance;
    none.placement = vec4f(0.0, -1000.0, 0.0, 0.0);
    none.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
    instances[i] = none;
}
