export const instanceAABBShader = /* wgsl */`
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
    instanceCount : u32,
    _pad          : vec3u,
}

@group(0) @binding(0) var<storage, read>       instances  : array<InstanceData>;
@group(0) @binding(1) var<storage, read>       blasNodes  : array<BVHNode>;
@group(0) @binding(2) var<storage, read_write> centroids  : array<vec4f>;
@group(0) @binding(3) var<uniform>             params     : Params;

fn transformPoint(inst: InstanceData, p: vec3f) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.instanceCount) { return; }

    let inst = instances[idx];
    let blasRoot = blasNodes[inst.blasNodeOffset];
    let localMin = blasRoot.boundsMin;
    let localMax = blasRoot.boundsMax;

    // Transform 8 corners of local AABB to world space
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

    centroids[idx] = vec4f((worldMin + worldMax) * 0.5, 0.0);
}
`;
