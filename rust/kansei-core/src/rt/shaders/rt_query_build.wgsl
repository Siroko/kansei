// The hardware backend's BLAS input (rt/query.rs), for one layer of the grid (its triangles from
// `base`, its counters): each gathered triangle's three vertices from its
// world triangle (v0, v0 + e1, v0 + e2), as the BLAS's vertex buffers read them (three f32 a
// vertex, nine a triangle). Opaque triangles fill the first geometry's slot of their id; with the
// split (alpha-tested or glass sources gathered), the others fill the second geometry's. A slot
// nothing fills holds a filler: a centimetre triangle on a lattice far below the grid's box, which
// no ray reaches (the trace stays within the box). (Metal's BLAS builds lost the real triangles
// among tens of thousands of inactive ones, NaN or degenerate, that Vulkan and DX12 would skip.)
// Included after rt_types.wgsl.

struct KanseiRtQueryParams {
    // the filler lattice's corner: below the grid's box
    filler   : vec3f,
    // slots a geometry: the grid's triangle capacity
    capacity : u32,
    // 1: a second, non-opaque geometry
    split    : u32,
    // the layer's first triangle
    base     : u32,
}

@group(0) @binding(0) var<storage, read> kansei_rt_triangles : array<vec4f>;
// [0]: triangles gathered
@group(0) @binding(1) var<storage, read> counters : array<u32>;
@group(0) @binding(2) var<storage, read_write> opaque_positions : array<f32>;
@group(0) @binding(3) var<uniform> qp : KanseiRtQueryParams;
@group(0) @binding(4) var<storage, read_write> other_positions : array<f32>;

fn write_vertex(other: bool, at: u32, v: vec3f) {
    if (other) {
        other_positions[at] = v.x;
        other_positions[at + 1u] = v.y;
        other_positions[at + 2u] = v.z;
    } else {
        opaque_positions[at] = v.x;
        opaque_positions[at + 1u] = v.y;
        opaque_positions[at + 2u] = v.z;
    }
}

fn write_slot(other: bool, slot: u32, v0: vec3f, v1: vec3f, v2: vec3f) {
    write_vertex(other, slot * 9u, v0);
    write_vertex(other, slot * 9u + 3u, v1);
    write_vertex(other, slot * 9u + 6u, v2);
}

fn fill_slot(other: bool, slot: u32) {
    let p = qp.filler - 0.1 * vec3f(f32(slot & 255u), f32((slot >> 8u) & 255u), f32(slot >> 16u));
    write_slot(other, slot, p, p + vec3f(0.01, 0.0, 0.0), p + vec3f(0.0, 0.0, 0.01));
}

@compute @workgroup_size(64)
fn positions_main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let id = gid.x + gid.y * groups.x * 64u;
    if (id >= qp.capacity) {
        return;
    }
    let split = qp.split != 0u;
    if (id >= min(counters[0], qp.capacity)) {
        fill_slot(false, id);
        if (split) {
            fill_slot(true, id);
        }
        return;
    }
    let at = qp.base + id;
    let a = kansei_rt_triangles[at * 4u];
    let v0 = a.xyz;
    let v1 = v0 + kansei_rt_triangles[at * 4u + 1u].xyz;
    let v2 = v0 + kansei_rt_triangles[at * 4u + 2u].xyz;
    let other = split && (bitcast<u32>(a.w) & (KANSEI_RT_ALPHA | KANSEI_RT_GLASS)) != 0u;
    write_slot(other, id, v0, v1, v2);
    if (split) {
        fill_slot(!other, id);
    }
}

// Nothing: a submit to wait for.
@compute @workgroup_size(1)
fn mark_main() {
}
