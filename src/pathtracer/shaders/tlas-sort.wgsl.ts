/**
 * GPU compute shaders for spatially sorting TLAS instances by Morton code.
 *
 * Two entry points:
 *   computeKeys      — reads instance translations, writes (mortonCode, index) pairs
 *   gatherInstances  — reads sorted indices, reorders instances into a destination buffer
 */
export const tlasSortShader = /* wgsl */`

struct Instance {
    transform0    : vec4f,
    transform1    : vec4f,
    transform2    : vec4f,
    invTransform0 : vec4f,
    invTransform1 : vec4f,
    invTransform2 : vec4f,
    metadata      : vec4u,
}

struct SortParams {
    count     : u32,
    sceneMinX : f32,
    sceneMinY : f32,
    sceneMinZ : f32,
    sceneExtX : f32,   // 1.0 / (max - min) per axis
    sceneExtY : f32,
    sceneExtZ : f32,
    _pad      : u32,
}

@group(0) @binding(0) var<storage, read>       instances  : array<Instance>;
@group(0) @binding(1) var<storage, read_write> mortonKeys : array<u32>;
@group(0) @binding(2) var<storage, read_write> mortonVals : array<u32>;
@group(0) @binding(3) var<uniform>             params     : SortParams;

fn expandBits10(v: u32) -> u32 {
    var x = v & 0x3FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}

fn morton3D(x: f32, y: f32, z: f32) -> u32 {
    let ix = u32(clamp(x * 1024.0, 0.0, 1023.0));
    let iy = u32(clamp(y * 1024.0, 0.0, 1023.0));
    let iz = u32(clamp(z * 1024.0, 0.0, 1023.0));
    return (expandBits10(ix) << 2u) | (expandBits10(iy) << 1u) | expandBits10(iz);
}

// Compute Morton codes from instance centroids (translation component of transform)
@compute @workgroup_size(256)
fn computeKeys(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let inst = instances[idx];
    let cx = inst.transform0.w;
    let cy = inst.transform1.w;
    let cz = inst.transform2.w;

    let nx = (cx - params.sceneMinX) * params.sceneExtX;
    let ny = (cy - params.sceneMinY) * params.sceneExtY;
    let nz = (cz - params.sceneMinZ) * params.sceneExtZ;

    mortonKeys[idx] = morton3D(nx, ny, nz);
    mortonVals[idx] = idx;
}
`;

/**
 * Shader to gather/reorder instances based on sorted indices.
 */
export const tlasGatherShader = /* wgsl */`

struct Instance {
    transform0    : vec4f,
    transform1    : vec4f,
    transform2    : vec4f,
    invTransform0 : vec4f,
    invTransform1 : vec4f,
    invTransform2 : vec4f,
    metadata      : vec4u,
}

struct GatherParams {
    count : u32,
    _p0   : u32,
    _p1   : u32,
    _p2   : u32,
}

@group(0) @binding(0) var<storage, read>       sortedVals   : array<u32>;
@group(0) @binding(1) var<storage, read>       srcInstances : array<Instance>;
@group(0) @binding(2) var<storage, read_write> dstInstances : array<Instance>;
@group(0) @binding(3) var<uniform>             params       : GatherParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let srcIdx = sortedVals[idx];
    dstInstances[idx] = srcInstances[srcIdx];
}
`;
