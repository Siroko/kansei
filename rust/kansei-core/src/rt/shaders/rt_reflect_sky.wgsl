// The sky past the voxels for the RT reflections and glass (rt_reflect_hit.wgsl's rtSky): along
// d for a ray whose lobe has this tan (rtLobeTan), the environment cubemap at the lobe's roughness
// (RT_REFLECT_ENV: RtReflectionsEffect::set_environment_map; SKY_ENVIRONMENT_WGSL), else the sky's
// SH (`sky`, the SkyLighting uniform the trace and the glass pass declare).

@group(0) @binding(16) var rtEnvironment : texture_cube<f32>;
@group(0) @binding(17) var rtEnvironmentSampler : sampler;

fn rtSky(d: vec3f, tanHalf: f32) -> vec3f {
    if ((rp.flags & RT_REFLECT_ENV) != 0u) {
        return skyEnvironment(rtEnvironment, rtEnvironmentSampler, d, sqrt(max(tanHalf, 0.0)));
    }
    return skyRadiance(sky, d);
}
