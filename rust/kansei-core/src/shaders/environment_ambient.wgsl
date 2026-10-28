// The standard lit material's ambient from an environment (Material::standard_lit_environment):
// the irradiance of its SH (SKY_LIGHTING_WGSL) over pi on the diffuse albedo, as the GI effects'
// `material_ambient` takes a material's own sky light to be, and, where the RT reflections don't
// trace the surface, the split sum's specular: the environment cubemap at the roughness
// (SKY_ENVIRONMENT_WGSL; the SH's radiance without one) by the analytic environment BRDF. Group 0
// binding 1 is the SkyLighting uniform, 2 and 3 the cubemap and its sampler (with
// KANSEI_ENVIRONMENT_MAP). Prefixed with SKY_LIGHTING_WGSL and SKY_ENVIRONMENT_WGSL.

@group(0) @binding(1) var<uniform> kansei_sky : SkyLighting;
KANSEI_ENVIRONMENT_BINDINGS

fn kansei_environment_reflection(r: vec3f, roughness: f32) -> vec3f {
    KANSEI_ENVIRONMENT_REFLECTION
}

// The light the environment leaves a surface of normal n, seen along v (toward the eye), with
// `traced` the standard material's (0 untraced, 1 reflective, 2 glass).
fn kansei_environment_ambient(n: vec3f, v: vec3f, base: vec3f, roughness: f32, metallic: f32, traced: f32) -> vec3f {
    var c = base * (1.0 - metallic) * skyIrradiance(kansei_sky, n) * 0.31830989;
    if (traced < 0.5) {
        let f0 = mix(vec3f(0.04), base, metallic);
        let nv = clamp(dot(n, v), 1e-4, 1.0);
        c += kansei_environment_reflection(reflect(-v, n), roughness) * skyEnvironmentBrdf(f0, roughness, nv);
    }
    return c;
}
