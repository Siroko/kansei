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

// sdfSoftShadow for a point on a surface of normal `n` (`p` a little off it): what the field
// measures along the ray is set against how far the surface's own plane is at that point (the
// field at the start, plus the ray's climb, t n.l), and only something nearer than that counts as
// an occluder. A ray grazing its own surface (a wall lit from high above along it) then stays lit
// instead of being shadowed by the wall it leaves, in bands that follow the voxels, while a block
// standing on the surface still shadows it.
fn sdfSurfaceShadow(vol: VoxelVolume, sdf: texture_3d<f32>, linearClamp: sampler, p: vec3f, n: vec3f, toLight: vec3f, k: f32, maxT: f32) -> f32 {
    let start = sdfDistance(vol, sdf, linearClamp, p);
    let climb = max(dot(n, toLight), 0.0);
    var t = 2.0 * vol.voxelSize;
    var visibility = 1.0;
    for (var i = 0u; i < 64u; i++) {
        if (t >= maxT) { break; }
        let q = p + toLight * t;
        let uvw = voxelUvw(vol, q);
        if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { break; }
        let d = textureSampleLevel(sdf, linearClamp, uvw, 0.0).r;
        // the own plane is this far off the ray here; something nearer, by a voxel, occludes
        if (d < start + t * climb - vol.voxelSize) {
            visibility = min(visibility, k * (d - 0.5 * vol.voxelSize) / t);
            if (visibility <= 0.0) { return 0.0; }
        }
        t += max(d, 0.5 * vol.voxelSize);
    }
    return clamp(visibility, 0.0, 1.0);
}

// The soft shadow's hardness and reach for a light of radius `radius` `distance` metres away:
// the light's disk seen from the point spans radius / distance (Quilez's k is its inverse), and
// the march stops short of the light, where things next to the lamp (the ceiling it hangs under)
// are no occluders of it. Returns (k, maxT).
fn sdfLightShape(vol: VoxelVolume, radius: f32, distance: f32) -> vec2f {
    let r = max(radius, vol.voxelSize);
    return vec2f(clamp(distance / r, 2.0, 64.0), max(distance - 2.0 * r, 0.0));
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
