export const tlasRefitShader = /* wgsl */`
struct BVHNode {
    boundsMin  : vec3f,
    leftChild  : i32,
    boundsMax  : vec3f,
    rightChild : i32,
}

struct InstanceData {
    transform0   : vec4f,
    transform1   : vec4f,
    transform2   : vec4f,
    invTransform0: vec4f,
    invTransform1: vec4f,
    invTransform2: vec4f,
    blasNodeOffset: u32,
    blasTriOffset : u32,
    blasTriCount  : u32,
    materialIndex: u32,
}

struct AABB {
    aabbMin : vec3f,
    aabbMax : vec3f,
}

struct Params {
    leafCount : u32,
    _pad0     : u32,
    _pad1     : u32,
    _pad2     : u32,
}

@group(0) @binding(0) var<storage, read_write> nodes     : array<BVHNode>;
@group(0) @binding(1) var<storage, read>       instances : array<InstanceData>;
@group(0) @binding(2) var<storage, read>       blasNodes : array<BVHNode>;
@group(0) @binding(3) var<uniform>             params    : Params;

fn transformPoint(inst: InstanceData, p: vec3f) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

fn instanceAABB(instIdx: u32) -> AABB {
    let inst = instances[instIdx];
    let blasRoot = blasNodes[inst.blasNodeOffset];
    let localMin = blasRoot.boundsMin;
    let localMax = blasRoot.boundsMax;

    // Transform all 8 corners to world space
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

// Pass 1: Initialize leaf node bounds from instance world-space AABBs.
// Instances are read directly by index (no sorted order needed).
// Tree structure (child pointers) is pre-uploaded by CPU.
@compute @workgroup_size(256)
fn initLeaves(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    if (idx >= n) { return; }

    // Leaf nodes are at indices [n-1, 2n-2]
    let leafNodeIdx = idx + n - 1u;
    let aabb = instanceAABB(idx);
    nodes[leafNodeIdx].boundsMin = aabb.aabbMin;
    nodes[leafNodeIdx].boundsMax = aabb.aabbMax;
    // Encode instance index in leaf child pointer
    nodes[leafNodeIdx].leftChild  = -i32(idx) - 1;
    nodes[leafNodeIdx].rightChild = -1;
}

// Pass 2+: Merge internal node bounds from children (run ceil(log2(N)) times).
// Child pointers were set by CPU; this pass only computes AABB bounds.
@compute @workgroup_size(256)
fn mergeNodes(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    // Internal nodes are [0, n-2]
    if (idx >= n - 1u) { return; }

    let left  = u32(nodes[idx].leftChild);
    let right = u32(nodes[idx].rightChild);
    nodes[idx].boundsMin = min(nodes[left].boundsMin, nodes[right].boundsMin);
    nodes[idx].boundsMax = max(nodes[left].boundsMax, nodes[right].boundsMax);
}
`;
