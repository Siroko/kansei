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

struct Params {
    leafCount : u32,
    _pad      : vec3u,
}

@group(0) @binding(0) var<storage, read_write> nodes         : array<BVHNode>;
@group(0) @binding(1) var<storage, read>       instances     : array<InstanceData>;
@group(0) @binding(2) var<storage, read>       blasNodes     : array<BVHNode>;
@group(0) @binding(3) var<storage, read_write> atomicFlags   : array<atomic<u32>>;
@group(0) @binding(4) var<uniform>             params        : Params;
@group(0) @binding(5) var<storage, read>       parents       : array<i32>;
@group(0) @binding(6) var<storage, read>       sortedIndices : array<u32>;

fn transformPoint(inst: InstanceData, p: vec3f) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

fn instanceAABB(sortedLeafIdx: u32) -> vec2<vec3f> {
    let instIdx = sortedIndices[sortedLeafIdx];
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
    return vec2<vec3f>(worldMin, worldMax);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    if (idx >= n) { return; }

    // Leaf nodes are at indices [n-1, 2n-2]
    let leafNodeIdx = idx + n - 1u;
    let aabb = instanceAABB(idx);
    nodes[leafNodeIdx].boundsMin = aabb[0];
    nodes[leafNodeIdx].boundsMax = aabb[1];
    // Leaf children encode the sorted instance index
    nodes[leafNodeIdx].leftChild  = -i32(sortedIndices[idx]) - 1;
    nodes[leafNodeIdx].rightChild = -1;

    // Walk up the tree
    var current = parents[leafNodeIdx];
    while (current >= 0) {
        let old = atomicAdd(&atomicFlags[u32(current)], 1u);
        if (old == 0u) { return; } // first child -- bail

        let left  = u32(nodes[u32(current)].leftChild);
        let right = u32(nodes[u32(current)].rightChild);
        nodes[u32(current)].boundsMin = min(nodes[left].boundsMin, nodes[right].boundsMin);
        nodes[u32(current)].boundsMax = max(nodes[left].boundsMax, nodes[right].boundsMax);

        current = parents[u32(current)];
    }
}
`;
