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
    var color = vec3f(0.0);
    var transmittance = 1.0;
    var dist = startDist;
    let maxLod = f32(vol.mipCount - 1u);
    for (var i = 0u; i < maxSteps; i++) {
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
    return vec4f(color * vol.radianceScale, transmittance);
}
