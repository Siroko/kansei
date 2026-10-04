// Cone tracing through a VoxelVolume (needs voxel_volume.wgsl): the step through the mips that
// both of miaumiau.cat/?p=1476's consumers and Crassin et al. 2011 use. Each step samples the mip
// whose voxels are as wide as the cone there and advances half that width, compositing front to
// back. Returns the scene radiance gathered (rgb) and the transmittance left (a): add
// `a * skyRadiance(sky, dir)` (or any escape term) for the light from past the volume.
fn voxelConeTrace(
    vol: VoxelVolume,
    radiance: texture_3d<f32>,
    linearClamp: sampler,
    origin: vec3f,
    dir: vec3f,
    tanHalf: f32,
    startDist: f32,
    maxDist: f32,
    maxSteps: u32,
) -> vec4f {
    return voxelConeTraceSplit(vol, radiance, linearClamp, origin, dir, tanHalf, startDist, maxDist, maxDist, maxSteps).far;
}

// A cone traced as voxelConeTrace does, which also gives the transmittance it had left on
// reaching `nearDist`: what the same cone stopped there would return (an occlusion cone that is
// the first metres of a light cone).
struct VoxelSplitCone {
    far      : vec4f,   // voxelConeTrace's result
    nearOpen : f32,     // the transmittance at nearDist (or at the end, if it stopped before)
}

fn voxelConeTraceSplit(
    vol: VoxelVolume,
    radiance: texture_3d<f32>,
    linearClamp: sampler,
    origin: vec3f,
    dir: vec3f,
    tanHalf: f32,
    startDist: f32,
    nearDist: f32,
    maxDist: f32,
    maxSteps: u32,
) -> VoxelSplitCone {
    var color = vec3f(0.0);
    var transmittance = 1.0;
    var nearOpen = -1.0;
    var dist = startDist;
    let maxLod = f32(vol.mipCount - 1u);
    for (var i = 0u; i < maxSteps; i++) {
        if (nearOpen < 0.0 && dist >= nearDist) { nearOpen = transmittance; }
        if (dist >= maxDist || transmittance < 0.01) { break; }
        let diameter = max(vol.voxelSize, 2.0 * tanHalf * dist);
        let uvw = voxelUvw(vol, origin + dir * dist);
        if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { break; }
        let lod = min(log2(diameter / vol.voxelSize), maxLod);
        let s = textureSampleLevel(radiance, linearClamp, uvw, lod);
        // the step crosses `step / footprint` of the sampled voxel: correct its opacity for that
        // length (Beer-Lambert), and its premultiplied radiance by the same share
        let step = 0.5 * diameter;
        let crossed = step / (vol.voxelSize * exp2(lod));
        let a = 1.0 - pow(max(1.0 - s.a, 0.0), crossed);
        let share = select(crossed, a / s.a, s.a > 1e-4);
        color += transmittance * s.rgb * share;
        transmittance *= 1.0 - a;
        dist += step;
    }
    return VoxelSplitCone(vec4f(color * vol.radianceScale, transmittance), select(nearOpen, transmittance, nearOpen < 0.0));
}

// The hemisphere above a normal as five cones (Crassin et al. 2011): one along `n` and four at 60
// degrees round it, each 60 degrees wide (`VOXEL_HEMISPHERE_TAN`). Cone `k` (0..5): its direction
// (xyz) and its weight (w) for irradiance, so that the sum over k of
// `w * (cone radiance)` is the irradiance a surface facing `n` receives:
//
//     for (var k = 0u; k < VOXEL_HEMISPHERE_CONES; k++) {
//         let cone = voxelHemisphereCone(n, k);
//         let c = voxelConeTrace(vol, radiance, linearClamp, o, cone.xyz, VOXEL_HEMISPHERE_TAN, start, 1e4, steps);
//         irradiance += cone.w * (c.rgb + c.a * sky(cone.xyz));
//     }
const VOXEL_HEMISPHERE_CONES : u32 = 5u;
const VOXEL_HEMISPHERE_TAN : f32 = 0.577;

fn voxelHemisphereCone(n: vec3f, k: u32) -> vec4f {
    let t = normalize(select(cross(n, vec3f(0.0, 1.0, 0.0)), cross(n, vec3f(1.0, 0.0, 0.0)), abs(n.y) > 0.9));
    let b = cross(n, t);
    var dirs = array<vec3f, 5>(n, 0.5 * n + 0.866 * t, 0.5 * n - 0.866 * t, 0.5 * n + 0.866 * b, 0.5 * n - 0.866 * b);
    // cosine-weighted shares (0.25 straight up, 0.15 for each side cone) times pi, normalised to 1
    let share = select(0.15, 0.25, k == 0u) / 0.85;
    return vec4f(dirs[min(k, 4u)], 3.14159265 * share);
}
