/**
 * GPU compute shaders for building a BVH4 (4-wide) TLAS from Morton-sorted instances.
 *
 * Two entry points:
 *   buildLeaves   — groups 4 consecutive instances into BVH4 leaf nodes
 *   buildInternal — groups 4 consecutive child nodes into BVH4 internal nodes
 *
 * Node layout (8 × vec4f = 128 bytes = 1 cache line):
 *   [0] childMinX  [1] childMaxX
 *   [2] childMinY  [3] childMaxY
 *   [4] childMinZ  [5] childMaxZ
 *   [6] children (vec4i: negative = leaf instance, positive = internal node)
 *   [7] counts   (vec4u: 1 = valid leaf, 0 = sentinel/internal)
 */
export const tlasBuildShader = /* wgsl */`

struct InstanceData {
    transform0    : vec4f,
    transform1    : vec4f,
    transform2    : vec4f,
    invTransform0 : vec4f,
    invTransform1 : vec4f,
    invTransform2 : vec4f,
    blasNodeOffset: u32,
    blasTriOffset : u32,
    blasTriCount  : u32,
    materialIndex : u32,
}

struct AABB {
    mn : vec3f,
    mx : vec3f,
}

struct BuildParams {
    instanceCount : u32,
    nodeOffset    : u32,
    childOffset   : u32,
    nodeCount     : u32,
    childCount    : u32,
    _pad0         : u32,
    _pad1         : u32,
    _pad2         : u32,
}

@group(0) @binding(0) var<storage, read>       instances : array<InstanceData>;
@group(0) @binding(1) var<storage, read>       bvh4Nodes : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tlasBvh4  : array<vec4f>;
@group(0) @binding(3) var<uniform>             params    : BuildParams;

fn transformPoint(inst: InstanceData, p: vec3f) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

fn instanceAABB(instIdx: u32) -> AABB {
    let inst = instances[instIdx];
    let base = inst.blasNodeOffset * 8u;
    let cmnx = bvh4Nodes[base + 0u];
    let cmxx = bvh4Nodes[base + 1u];
    let cmny = bvh4Nodes[base + 2u];
    let cmxy = bvh4Nodes[base + 3u];
    let cmnz = bvh4Nodes[base + 4u];
    let cmxz = bvh4Nodes[base + 5u];
    let localMin = vec3f(
        min(min(cmnx.x, cmnx.y), min(cmnx.z, cmnx.w)),
        min(min(cmny.x, cmny.y), min(cmny.z, cmny.w)),
        min(min(cmnz.x, cmnz.y), min(cmnz.z, cmnz.w)),
    );
    let localMax = vec3f(
        max(max(cmxx.x, cmxx.y), max(cmxx.z, cmxx.w)),
        max(max(cmxy.x, cmxy.y), max(cmxy.z, cmxy.w)),
        max(max(cmxz.x, cmxz.y), max(cmxz.z, cmxz.w)),
    );
    var worldMin = vec3f(1e30);
    var worldMax = vec3f(-1e30);
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3f(
            select(localMin.x, localMax.x, (i & 1u) != 0u),
            select(localMin.y, localMax.y, (i & 2u) != 0u),
            select(localMin.z, localMax.z, (i & 4u) != 0u),
        );
        let world = transformPoint(inst, corner);
        worldMin = min(worldMin, world);
        worldMax = max(worldMax, world);
    }
    return AABB(worldMin, worldMax);
}

fn childUnionAABB(nodeIdx: u32) -> AABB {
    let b = nodeIdx * 8u;
    let cmnx = tlasBvh4[b + 0u]; let cmxx = tlasBvh4[b + 1u];
    let cmny = tlasBvh4[b + 2u]; let cmxy = tlasBvh4[b + 3u];
    let cmnz = tlasBvh4[b + 4u]; let cmxz = tlasBvh4[b + 5u];
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

fn writeNode(base: u32, mnX: array<f32, 4>, mxX: array<f32, 4>,
             mnY: array<f32, 4>, mxY: array<f32, 4>,
             mnZ: array<f32, 4>, mxZ: array<f32, 4>,
             ch: array<i32, 4>, cnt: array<u32, 4>) {
    tlasBvh4[base + 0u] = vec4f(mnX[0], mnX[1], mnX[2], mnX[3]);
    tlasBvh4[base + 1u] = vec4f(mxX[0], mxX[1], mxX[2], mxX[3]);
    tlasBvh4[base + 2u] = vec4f(mnY[0], mnY[1], mnY[2], mnY[3]);
    tlasBvh4[base + 3u] = vec4f(mxY[0], mxY[1], mxY[2], mxY[3]);
    tlasBvh4[base + 4u] = vec4f(mnZ[0], mnZ[1], mnZ[2], mnZ[3]);
    tlasBvh4[base + 5u] = vec4f(mxZ[0], mxZ[1], mxZ[2], mxZ[3]);
    tlasBvh4[base + 6u] = bitcast<vec4f>(vec4i(ch[0], ch[1], ch[2], ch[3]));
    tlasBvh4[base + 7u] = bitcast<vec4f>(vec4u(cnt[0], cnt[1], cnt[2], cnt[3]));
}

// Build BVH4 leaf nodes — each groups up to 4 consecutive Morton-sorted instances
@compute @workgroup_size(256)
fn buildLeaves(@builtin(global_invocation_id) gid: vec3u) {
    let leafIdx = gid.x;
    if (leafIdx >= params.nodeCount) { return; }

    let nodeIdx = params.nodeOffset + leafIdx;
    let base = nodeIdx * 8u;

    // Sentinel values: inverted AABBs ensure AABB test conceptually fails,
    // children=-1 (leaf) + counts=0 (skip) provides safety net
    var mnX = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mxX = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mnY = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mxY = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mnZ = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mxZ = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var ch = array<i32, 4>(-1, -1, -1, -1);
    var cnt = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var j = 0u; j < 4u; j++) {
        let instIdx = leafIdx * 4u + j;
        if (instIdx < params.instanceCount) {
            let aabb = instanceAABB(instIdx);
            mnX[j] = aabb.mn.x; mxX[j] = aabb.mx.x;
            mnY[j] = aabb.mn.y; mxY[j] = aabb.mx.y;
            mnZ[j] = aabb.mn.z; mxZ[j] = aabb.mx.z;
            ch[j] = -(i32(instIdx) + 1);
            cnt[j] = 1u;
        }
    }

    writeNode(base, mnX, mxX, mnY, mxY, mnZ, mxZ, ch, cnt);
}

// Build BVH4 internal nodes — each groups up to 4 consecutive child nodes
@compute @workgroup_size(256)
fn buildInternal(@builtin(global_invocation_id) gid: vec3u) {
    let parentIdx = gid.x;
    if (parentIdx >= params.nodeCount) { return; }

    let nodeIdx = params.nodeOffset + parentIdx;
    let base = nodeIdx * 8u;

    var mnX = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mxX = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mnY = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mxY = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var mnZ = array<f32, 4>(1e30, 1e30, 1e30, 1e30);
    var mxZ = array<f32, 4>(-1e30, -1e30, -1e30, -1e30);
    var ch = array<i32, 4>(-1, -1, -1, -1);
    var cnt = array<u32, 4>(0u, 0u, 0u, 0u);

    for (var j = 0u; j < 4u; j++) {
        let childLocalIdx = parentIdx * 4u + j;
        if (childLocalIdx < params.childCount) {
            let childNodeIdx = params.childOffset + childLocalIdx;
            let aabb = childUnionAABB(childNodeIdx);
            mnX[j] = aabb.mn.x; mxX[j] = aabb.mx.x;
            mnY[j] = aabb.mn.y; mxY[j] = aabb.mx.y;
            mnZ[j] = aabb.mn.z; mxZ[j] = aabb.mx.z;
            ch[j] = i32(childNodeIdx);
        }
    }

    writeNode(base, mnX, mxX, mnY, mxY, mnZ, mxZ, ch, cnt);
}
`;
