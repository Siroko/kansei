// ── TLAS instance Morton code computation ─────────────────────────────────────
//
// Reads instance transforms, extracts translation (centroid), maps to [0,1]
// within scene bounds, and generates Morton codes for radix sorting.
//
// Entry point: computeKeys
// Also includes: gatherInstances (reorder instances by sorted indices)

// ── computeKeys shader ────────────────────────────────────────────────────────

struct Instance {
    transform0    : vec4f,
    transform1    : vec4f,
    transform2    : vec4f,
    inv_transform0: vec4f,
    inv_transform1: vec4f,
    inv_transform2: vec4f,
    metadata      : vec4u,
}

struct SortParams {
    count      : u32,
    scene_min_x: f32,
    scene_min_y: f32,
    scene_min_z: f32,
    scene_ext_x: f32,   // 1.0 / (max - min) per axis
    scene_ext_y: f32,
    scene_ext_z: f32,
    _pad       : u32,
}

@group(0) @binding(0) var<storage, read>       instances   : array<Instance>;
@group(0) @binding(1) var<storage, read_write> morton_keys : array<u32>;
@group(0) @binding(2) var<storage, read_write> morton_vals : array<u32>;
@group(0) @binding(3) var<uniform>             params      : SortParams;

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

// Compute Morton codes from instance centroids (translation component of transform)
@compute @workgroup_size(256)
fn compute_keys(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let inst = instances[idx];
    let cx = inst.transform0.w;
    let cy = inst.transform1.w;
    let cz = inst.transform2.w;

    let nx = (cx - params.scene_min_x) * params.scene_ext_x;
    let ny = (cy - params.scene_min_y) * params.scene_ext_y;
    let nz = (cz - params.scene_min_z) * params.scene_ext_z;

    morton_keys[idx] = morton_3d(nx, ny, nz);
    morton_vals[idx] = idx;
}
