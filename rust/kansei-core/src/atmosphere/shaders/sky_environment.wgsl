// Sampling the sky environment cubemap (SkyAtmosphereBindings::environment, a texture_cube with a
// GGX-prefiltered mip chain) from materials: the split-sum specular term of a sky reflection.

// Prefiltered sky radiance for a GGX lobe of `roughness` (0 mirror .. 1) around the reflection r.
fn skyEnvironment(env: texture_cube<f32>, envSampler: sampler, r: vec3f, roughness: f32) -> vec3f {
    let mips = f32(textureNumLevels(env));
    return textureSampleLevel(env, envSampler, r, saturate(roughness) * (mips - 1.0)).rgb;
}

// The split sum's environment BRDF, analytic (Karis 2014, "Physically Based Shading on Mobile"):
// specular reflectance of the sky for a surface of `specularColor` (F0), roughness and N.V.
fn skyEnvironmentBrdf(specularColor: vec3f, roughness: f32, nv: f32) -> vec3f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * saturate(nv))) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    return specularColor * ab.x + ab.y;
}
