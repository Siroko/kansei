// The GBuffer's four colour targets, for materials drawn through a PostProcessingVolume
// (`MaterialOptions::mrt_output_count = Some(4)`): the shaded colour, the emitted part of it, the
// world normal and the albedo, which screen-space and voxel GI read. Prepend
// `materials::GBUFFER_OUT_WGSL` and return `kansei_gbuffer_out(..)` from the fragment shader.

struct KanseiGBufferOut {
    @location(0) color    : vec4f,   // shaded radiance (HDR)
    @location(1) emissive : vec4f,   // the emitted part of it; alpha 0 (1 marks the fluid surface)
    @location(2) normal   : vec4f,   // world normal, encoded n * 0.5 + 0.5
    @location(3) albedo   : vec4f,   // diffuse albedo
}

fn kansei_gbuffer_out(color: vec3f, emissive: vec3f, N: vec3f, albedo: vec3f) -> KanseiGBufferOut {
    var out: KanseiGBufferOut;
    out.color = vec4f(color, 1.0);
    // alpha 0: FluidMask::EmissiveAlpha takes an alpha of 0.5 and over for the fluid's surface
    out.emissive = vec4f(emissive, 0.0);
    out.normal = vec4f(N * 0.5 + 0.5, 1.0);
    out.albedo = vec4f(albedo, 1.0);
    return out;
}
