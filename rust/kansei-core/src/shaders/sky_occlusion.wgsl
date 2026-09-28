// Sky occlusion for materials (Renderer::enable_sky_occlusion): the share of the sky a point sees
// past the canopy around the camera (1 in the open, less under trees), to multiply
// the sky's ambient light by, as Lumen occludes Unreal's sky light. Bind `SkyOcclusion::volume`
// (texture_3d<f32>, filterable), a linear clamping sampler and `SkyOcclusion::params` (uniform
// SkyOcclusionParams). It is 1 before the first build, outside the volume and above it.

struct SkyOcclusionParams {
    center    : vec2f,   // world xz of the volume's centre
    invExtent : f32,     // 1 / its side (m)
    minY      : f32,     // world heights it spans
    invHeight : f32,
    enabled   : f32,
    _pad0     : f32,
    _pad1     : f32,
}

fn skyVisibility(volume: texture_3d<f32>, volumeSampler: sampler, p: SkyOcclusionParams, worldPos: vec3f) -> f32 {
    if (p.enabled < 0.5) { return 1.0; }
    let q = (worldPos.xz - p.center) * p.invExtent;
    let h = (worldPos.y - p.minY) * p.invHeight;
    if (h > 1.0) { return 1.0; }
    let v = textureSampleLevel(volume, volumeSampler, vec3f(q.x + 0.5, clamp(h, 0.0, 1.0), q.y + 0.5), 0.0).r;
    // fade to open sky toward the volume's edge
    return mix(v, 1.0, smoothstep(0.4, 0.5, max(abs(q.x), abs(q.y))));
}
