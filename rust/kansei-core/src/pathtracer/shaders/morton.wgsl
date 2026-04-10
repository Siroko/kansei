// ── Morton code generation (30-bit, 10 bits per axis) ─────────────────────────
//
// Converts 3D position (normalized to [0,1]) into a space-filling curve index.
// Used to spatially sort TLAS instances for coherent BVH4 construction.

struct Params {
    count      : u32,
    scene_min_x: f32,
    scene_min_y: f32,
    scene_min_z: f32,
    scene_ext_x: f32,   // 1.0 / (max - min) per axis
    scene_ext_y: f32,
    scene_ext_z: f32,
    _pad       : u32,
}

@group(0) @binding(0) var<storage, read>       centroids   : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> morton_keys : array<u32>;
@group(0) @binding(2) var<storage, read_write> morton_vals : array<u32>;
@group(0) @binding(3) var<uniform>             params      : Params;

fn expand_bits_10(v: u32) -> u32 {
    var x = v & 0x3FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}

fn morton_3d(x: f32, y: f32, z: f32) -> u32 {
    let ix = u32(clamp(x * 1024.0, 0.0, 1023.0));
    let iy = u32(clamp(y * 1024.0, 0.0, 1023.0));
    let iz = u32(clamp(z * 1024.0, 0.0, 1023.0));
    return (expand_bits_10(ix) << 2u) | (expand_bits_10(iy) << 1u) | expand_bits_10(iz);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let c = centroids[idx];
    let nx = (c.x - params.scene_min_x) * params.scene_ext_x;
    let ny = (c.y - params.scene_min_y) * params.scene_ext_y;
    let nz = (c.z - params.scene_min_z) * params.scene_ext_z;

    morton_keys[idx] = morton_3d(nx, ny, nz);
    morton_vals[idx] = idx;
}
