export const refitLeavesShader = /* wgsl */`
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

@group(0) @binding(0) var<storage, read_write> nodes         : array<BVHNode>;
@group(0) @binding(1) var<storage, read>       triangles     : array<f32>;
@group(0) @binding(2) var<storage, read_write> ready         : array<u32>;
@group(0) @binding(3) var<uniform>             params        : Params;
@group(0) @binding(4) var<storage, read>       sortedIndices : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    if (idx >= n) { return; }

    // Leaf nodes are at indices [n-1, 2n-2]
    let leafNodeIdx = idx + n - 1u;
    let triIdx = sortedIndices[idx];
    let base = (params.triOffset + triIdx) * params.triStride;
    let v0 = vec3f(triangles[base+0u], triangles[base+1u], triangles[base+2u]);
    let v1 = vec3f(triangles[base+3u], triangles[base+4u], triangles[base+5u]);
    let v2 = vec3f(triangles[base+6u], triangles[base+7u], triangles[base+8u]);

    nodes[leafNodeIdx].boundsMin = min(min(v0, v1), v2);
    nodes[leafNodeIdx].boundsMax = max(max(v0, v1), v2);
    nodes[leafNodeIdx].leftChild  = -i32(triIdx) - 1;
    nodes[leafNodeIdx].rightChild = -1;

    ready[leafNodeIdx] = 1u;
}
`;

export const refitInternalShader = /* wgsl */`
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

@group(0) @binding(0) var<storage, read_write> nodes : array<BVHNode>;
@group(0) @binding(1) var<storage, read_write> ready : array<u32>;
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    // Internal nodes are [0..n-2]
    if (idx >= n - 1u) { return; }

    // Skip if already computed or children not ready
    if (ready[idx] == 1u) { return; }

    let left  = u32(nodes[idx].leftChild);
    let right = u32(nodes[idx].rightChild);

    if (ready[left] == 0u || ready[right] == 0u) { return; }

    nodes[idx].boundsMin = min(nodes[left].boundsMin, nodes[right].boundsMin);
    nodes[idx].boundsMax = max(nodes[left].boundsMax, nodes[right].boundsMax);
    ready[idx] = 1u;
}
`;
