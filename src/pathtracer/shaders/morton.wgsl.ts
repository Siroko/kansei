export const mortonShader = /* wgsl */`
struct Params {
    count      : u32,
    sceneMinX  : f32,
    sceneMinY  : f32,
    sceneMinZ  : f32,
    sceneExtX  : f32,
    sceneExtY  : f32,
    sceneExtZ  : f32,
    _pad       : u32,
}

@group(0) @binding(0) var<storage, read>       centroids  : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> mortonKeys : array<u32>;
@group(0) @binding(2) var<storage, read_write> mortonVals : array<u32>;
@group(0) @binding(3) var<uniform>             params     : Params;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let c = centroids[idx];
    let nx = (c.x - params.sceneMinX) * params.sceneExtX;
    let ny = (c.y - params.sceneMinY) * params.sceneExtY;
    let nz = (c.z - params.sceneMinZ) * params.sceneExtZ;

    mortonKeys[idx] = morton3D(nx, ny, nz);
    mortonVals[idx] = idx;
}
`;
