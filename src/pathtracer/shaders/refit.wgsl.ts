export const refitShader = /* wgsl */`
struct BVHNode {
    boundsMin  : vec3f,
    leftChild  : i32,
    boundsMax  : vec3f,
    rightChild : i32,
}

struct Params {
    leafCount : u32,
    triOffset : u32,
    triStride : u32,
    _pad      : u32,
}

@group(0) @binding(0) var<storage, read_write> nodes       : array<BVHNode>;
@group(0) @binding(1) var<storage, read>       triangles   : array<f32>;
@group(0) @binding(2) var<storage, read_write> atomicFlags : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             params      : Params;
@group(0) @binding(4) var<storage, read>       parents     : array<i32>;
@group(0) @binding(5) var<storage, read>       sortedIndices : array<u32>;

fn triAABB(sortedLeafIdx: u32) -> vec2<vec3f> {
    let triIdx = sortedIndices[sortedLeafIdx];
    let base = (params.triOffset + triIdx) * params.triStride;
    let v0 = vec3f(triangles[base+0u], triangles[base+1u], triangles[base+2u]);
    let v1 = vec3f(triangles[base+3u], triangles[base+4u], triangles[base+5u]);
    let v2 = vec3f(triangles[base+6u], triangles[base+7u], triangles[base+8u]);
    return vec2<vec3f>(min(min(v0, v1), v2), max(max(v0, v1), v2));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    if (idx >= n) { return; }

    // Leaf nodes are at indices [n-1, 2n-2]
    let leafNodeIdx = idx + n - 1u;
    let aabb = triAABB(idx);
    nodes[leafNodeIdx].boundsMin = aabb[0];
    nodes[leafNodeIdx].boundsMax = aabb[1];
    // Leaf children encode the sorted triangle index
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
