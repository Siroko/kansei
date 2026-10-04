// Reading a JumpFloodSdf (gi::SDF_WGSL): the unsigned distance in metres to the nearest occupied
// voxel, over the volume's placement (`VoxelVolume`), sampled trilinearly. Needs
// voxel_volume.wgsl. Bind the field (`JumpFloodSdf::view`) as `texture_3d<f32>` and the volume's
// sampler and uniform.

// The distance at `p`; 1e4 outside the volume.
fn sdfDistance(vol: VoxelVolume, sdf: texture_3d<f32>, linearClamp: sampler, p: vec3f) -> f32 {
    let uvw = voxelUvw(vol, p);
    if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { return 1e4; }
    return textureSampleLevel(sdf, linearClamp, uvw, 0.0).r;
}

// The share of a light seen from `p` toward `toLight` (unit), up to `maxT` metres: sphere traced
// through the field, with Quilez's penumbra estimate (k: higher is harder). The march starts two
// voxels out (`p` should already sit off its surface) and leaves the volume unoccluded.
fn sdfSoftShadow(vol: VoxelVolume, sdf: texture_3d<f32>, linearClamp: sampler, p: vec3f, toLight: vec3f, k: f32, maxT: f32) -> f32 {
    var t = 2.0 * vol.voxelSize;
    var visibility = 1.0;
    for (var i = 0u; i < 64u; i++) {
        if (t >= maxT) { break; }
        let q = p + toLight * t;
        let uvw = voxelUvw(vol, q);
        if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { break; }
        let d = textureSampleLevel(sdf, linearClamp, uvw, 0.0).r;
        // a surface voxel reads half a voxel or less
        let clearance = d - 0.5 * vol.voxelSize;
        visibility = min(visibility, k * clearance / t);
        if (visibility <= 0.0) { return 0.0; }
        t += max(d, 0.5 * vol.voxelSize);
    }
    return clamp(visibility, 0.0, 1.0);
}

// Ambient occlusion around a surface at `p` (normal `n`): five taps out along the normal, from
// 2.5 to 8.5 voxels, each comparing the field with the distance walked less a voxel (the surface's
// own voxel may sit up to that much off it), so an open surface reads 1 and a nearby wall or
// crease less (distance-field AO).
fn sdfAo(vol: VoxelVolume, sdf: texture_3d<f32>, linearClamp: sampler, p: vec3f, n: vec3f) -> f32 {
    var occlusion = 0.0;
    var weight = 0.5;
    for (var i = 1u; i <= 5u; i++) {
        let free = (1.5 * f32(i)) * vol.voxelSize;
        let d = sdfDistance(vol, sdf, linearClamp, p + n * (free + vol.voxelSize));
        occlusion += weight * clamp((free - d) / free, 0.0, 1.0);
        weight *= 0.6;
    }
    return clamp(1.0 - occlusion, 0.0, 1.0);
}
