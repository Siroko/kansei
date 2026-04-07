// ── TLAS BVH4 construction from Morton-sorted instances ───────────────────────
//
// Two entry points:
//   buildLeaves   — groups 4 consecutive instances into BVH4 leaf nodes
//   buildInternal — groups 4 consecutive child nodes into BVH4 internal nodes
//
// Node layout (8 x vec4f = 128 bytes = 1 cache line):
//   [0] childMinX  [1] childMaxX
//   [2] childMinY  [3] childMaxY
//   [4] childMinZ  [5] childMaxZ
//   [6] children (vec4i: negative = leaf instance -(idx+1), positive = internal node)
//   [7] counts   (vec4u: 1 = valid leaf, 0 = sentinel/internal)

struct InstanceData {
    transform0     : vec4f,
    transform1     : vec4f,
    transform2     : vec4f,
    inv_transform0 : vec4f,
    inv_transform1 : vec4f,
    inv_transform2 : vec4f,
    blas_node_offset: u32,
    blas_tri_offset : u32,
    blas_tri_count  : u32,
    material_index  : u32,
}

struct AABB {
    mn : vec3f,
    mx : vec3f,
}

struct BuildParams {
    instance_count : u32,
    node_offset    : u32,
    child_offset   : u32,
    node_count     : u32,
    child_count    : u32,
    _pad0          : u32,
    _pad1          : u32,
    _pad2          : u32,
}

@group(0) @binding(0) var<storage, read>       instances  : array<InstanceData>;
@group(0) @binding(1) var<storage, read>       bvh4_nodes : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tlas_bvh4  : array<vec4f>;
@group(0) @binding(3) var<uniform>             params     : BuildParams;

fn transform_point(inst: InstanceData, p: vec3f) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

fn instance_aabb(inst_idx: u32) -> AABB {
    let inst = instances[inst_idx];
    let base = inst.blas_node_offset * 8u;
    let cmnx = bvh4_nodes[base + 0u];
    let cmxx = bvh4_nodes[base + 1u];
    let cmny = bvh4_nodes[base + 2u];
    let cmxy = bvh4_nodes[base + 3u];
    let cmnz = bvh4_nodes[base + 4u];
    let cmxz = bvh4_nodes[base + 5u];
    let local_min = vec3f(
        min(min(cmnx.x, cmnx.y), min(cmnx.z, cmnx.w)),
        min(min(cmny.x, cmny.y), min(cmny.z, cmny.w)),
        min(min(cmnz.x, cmnz.y), min(cmnz.z, cmnz.w)),
    );
    let local_max = vec3f(
        max(max(cmxx.x, cmxx.y), max(cmxx.z, cmxx.w)),
        max(max(cmxy.x, cmxy.y), max(cmxy.z, cmxy.w)),
        max(max(cmxz.x, cmxz.y), max(cmxz.z, cmxz.w)),
    );
    var world_min = vec3f(1e30);
    var world_max = vec3f(-1e30);
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3f(
            select(local_min.x, local_max.x, (i & 1u) != 0u),
            select(local_min.y, local_max.y, (i & 2u) != 0u),
            select(local_min.z, local_max.z, (i & 4u) != 0u),
        );
        let world = transform_point(inst, corner);
        world_min = min(world_min, world);
        world_max = max(world_max, world);
    }
    return AABB(world_min, world_max);
}

fn child_union_aabb(node_idx: u32) -> AABB {
    let b = node_idx * 8u;
    let cmnx = tlas_bvh4[b + 0u]; let cmxx = tlas_bvh4[b + 1u];
    let cmny = tlas_bvh4[b + 2u]; let cmxy = tlas_bvh4[b + 3u];
    let cmnz = tlas_bvh4[b + 4u]; let cmxz = tlas_bvh4[b + 5u];
    return AABB(
        vec3f(
            min(min(cmnx.x, cmnx.y), min(cmnx.z, cmnx.w)),
            min(min(cmny.x, cmny.y), min(cmny.z, cmny.w)),
            min(min(cmnz.x, cmnz.y), min(cmnz.z, cmnz.w)),
        ),
        vec3f(
            max(max(cmxx.x, cmxx.y), max(cmxx.z, cmxx.w)),
            max(max(cmxy.x, cmxy.y), max(cmxy.z, cmxy.w)),
            max(max(cmxz.x, cmxz.y), max(cmxz.z, cmxz.w)),
        ),
    );
}

fn write_node(base: u32, mn_x: array<f32, 4>, mx_x: array<f32, 4>,
              mn_y: array<f32, 4>, mx_y: array<f32, 4>,
              mn_z: array<f32, 4>, mx_z: array<f32, 4>,
              ch: array<i32, 4>, cnt: array<u32, 4>) {
    tlas_bvh4[base + 0u] = vec4f(mn_x[0], mn_x[1], mn_x[2], mn_x[3]);
    tlas_bvh4[base + 1u] = vec4f(mx_x[0], mx_x[1], mx_x[2], mx_x[3]);
    tlas_bvh4[base + 2u] = vec4f(mn_y[0], mn_y[1], mn_y[2], mn_y[3]);
    tlas_bvh4[base + 3u] = vec4f(mx_y[0], mx_y[1], mx_y[2], mx_y[3]);
    tlas_bvh4[base + 4u] = vec4f(mn_z[0], mn_z[1], mn_z[2], mn_z[3]);
    tlas_bvh4[base + 5u] = vec4f(mx_z[0], mx_z[1], mx_z[2], mx_z[3]);
    tlas_bvh4[base + 6u] = bitcast<vec4f>(vec4i(ch[0], ch[1], ch[2], ch[3]));
    tlas_bvh4[base + 7u] = bitcast<vec4f>(vec4u(cnt[0], cnt[1], cnt[2], cnt[3]));
}

// Build BVH4 leaf nodes — each groups up to 4 consecutive Morton-sorted instances
@compute @workgroup_size(256)
fn build_leaves(@builtin(global_invocation_id) gid: vec3u) {
    let leaf_idx = gid.x;
    if (leaf_idx >= params.node_count) { return; }

    let node_idx = params.node_offset + leaf_idx;
    let base = node_idx * 8u;

    // Sentinel values: inverted AABBs ensure AABB test conceptually fails,
    // children=-1 (leaf) + counts=0 (skip) provides safety net
    var mn_x = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mx_x = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mn_y = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mx_y = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mn_z = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mx_z = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var ch = array<i32, 4>(-1, -1, -1, -1);
    var cnt = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var j = 0u; j < 4u; j++) {
        let inst_idx = leaf_idx * 4u + j;
        if (inst_idx < params.instance_count) {
            let aabb = instance_aabb(inst_idx);
            mn_x[j] = aabb.mn.x; mx_x[j] = aabb.mx.x;
            mn_y[j] = aabb.mn.y; mx_y[j] = aabb.mx.y;
            mn_z[j] = aabb.mn.z; mx_z[j] = aabb.mx.z;
            ch[j] = -(i32(inst_idx) + 1);
            cnt[j] = 1u;
        }
    }

    write_node(base, mn_x, mx_x, mn_y, mx_y, mn_z, mx_z, ch, cnt);
}

// Build BVH4 internal nodes — each groups up to 4 consecutive child nodes
@compute @workgroup_size(256)
fn build_internal(@builtin(global_invocation_id) gid: vec3u) {
    let parent_idx = gid.x;
    if (parent_idx >= params.node_count) { return; }

    let node_idx = params.node_offset + parent_idx;
    let base = node_idx * 8u;

    var mn_x = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mx_x = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mn_y = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mx_y = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mn_z = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mx_z = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var ch = array<i32, 4>(-1, -1, -1, -1);
    var cnt = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var j = 0u; j < 4u; j++) {
        let child_local_idx = parent_idx * 4u + j;
        if (child_local_idx < params.child_count) {
            let child_node_idx = params.child_offset + child_local_idx;
            let aabb = child_union_aabb(child_node_idx);
            mn_x[j] = aabb.mn.x; mx_x[j] = aabb.mx.x;
            mn_y[j] = aabb.mn.y; mx_y[j] = aabb.mx.y;
            mn_z[j] = aabb.mn.z; mx_z[j] = aabb.mx.z;
            ch[j] = i32(child_node_idx);
        }
    }

    write_node(base, mn_x, mx_x, mn_y, mx_y, mn_z, mx_z, ch, cnt);
}
