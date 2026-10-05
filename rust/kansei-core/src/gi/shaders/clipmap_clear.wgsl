// Clears a region of a voxel clipmap level before it is voxelized again (gi::ClipmapVoxelizer):
// each voxel's surface words and its texel of the level's radiance, at the texel that holds it
// (voxel c in texel c mod dims), so the light the texel held for the voxel it stood for before
// the window moved is gone before the injection relights it.

struct ClearRegion {
    lo    : vec3i,   // the region's first voxel, in the level's lattice
    words : u32,     // u32 per voxel in the surfaces
    size  : vec3u,   // the region's voxels
    _pad0 : u32,
    dims  : vec3u,   // the level's window
    _pad1 : u32,
}

@group(0) @binding(0) var<uniform> cr : ClearRegion;
@group(0) @binding(1) var<storage, read_write> surfaces : array<u32>;
@group(0) @binding(2) var radiance : texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid >= cr.size)) { return; }
    let d = vec3i(cr.dims);
    let t = vec3u((((cr.lo + vec3i(gid)) % d) + d) % d);
    let idx = (t.z * cr.dims.y + t.y) * cr.dims.x + t.x;
    for (var w = 0u; w < cr.words; w++) {
        surfaces[idx * cr.words + w] = 0u;
    }
    textureStore(radiance, t, vec4f(0.0));
}
