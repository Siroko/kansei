// A voxel volume of the scene's light (gi::VoxelVolume): an rgba16float 3D texture with mips,
// rgb the radiance leaving each voxel premultiplied by its coverage, a its opacity across one
// voxel, so a 2x2x2 average makes the next mip. Radiance is stored divided by `radianceScale`
// (the volume's reference) to keep sums of bright emitters inside f16 and the fixed-point
// accumulators; multiply what you sample by it.
struct VoxelVolume {
    origin        : vec3f,   // world position of voxel (0, 0, 0)'s corner
    voxelSize     : f32,     // metres, the same on every axis
    dims          : vec3u,   // voxels of mip 0
    mipCount      : u32,
    invExtent     : vec3f,   // 1 / (dims * voxelSize): world to texture coordinates
    radianceScale : f32,     // stored radiance times this is scene radiance
}

fn voxelUvw(vol: VoxelVolume, p: vec3f) -> vec3f {
    return (p - vol.origin) * vol.invExtent;
}

fn voxelLinearIndex(vol: VoxelVolume, c: vec3u) -> u32 {
    return (c.z * vol.dims.y + c.y) * vol.dims.x + c.x;
}
