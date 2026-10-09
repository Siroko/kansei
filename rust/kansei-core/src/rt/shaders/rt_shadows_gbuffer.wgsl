// A surface whose direct light rt::RtShadowsEffect adds (ray-traced shadows): its material writes
// the GBuffer with `kansei_gbuffer_out_rt_lit` and leaves the sun and the spot lights out of its
// colour (only its emission and any fill light). The emissive target's alpha marks it, holding its
// roughness: 0.05 + 0.4 x roughness, between the 0 other materials write and the 0.5 and over the
// fluid's surface writes. A reflective one (`reflective`) also writes its F0 and roughness as
// `kansei_gbuffer_out_specular` does, for rt::RtReflectionsEffect. Prepend
// materials::GBUFFER_OUT_WGSL.

fn kansei_gbuffer_out_rt_lit(color: vec3f, emissive: vec3f, N: vec3f, albedo: vec3f, roughness: f32, f0: f32, reflective: bool) -> KanseiGBufferOut {
    var out = kansei_gbuffer_out(color, emissive, N, albedo);
    if (reflective) {
        out = kansei_gbuffer_out_specular(color, emissive, N, albedo, f0, roughness);
    }
    out.emissive.a = 0.05 + 0.4 * clamp(roughness, 0.0, 1.0);
    return out;
}
