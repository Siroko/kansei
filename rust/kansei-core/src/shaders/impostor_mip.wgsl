// Impostor bake: one mip of both atlases from the mip before, 2 x 2 texels each (frames are a
// power of two texels wide, so a block never straddles two frames). Coverage is averaged; the
// albedo, normal and depth are averaged over coverage (plainly where none is covered, so the
// dilated values carry down).

@group(0) @binding(0) var albedoIn : texture_2d<f32>;
@group(0) @binding(1) var normalDepthIn : texture_2d<f32>;
@group(0) @binding(2) var albedoOut : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var normalDepthOut : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= textureDimensions(albedoOut))) { return; }
    var weighted = vec4f(0.0);   // albedo times coverage, and the coverage
    var plain = vec3f(0.0);
    var normalWeighted = vec4f(0.0);   // (normal, depth) times coverage
    var normalPlain = vec4f(0.0);
    for (var k = 0u; k < 4u; k++) {
        let p = vec2i(gid.xy * 2u + vec2u(k & 1u, k >> 1u));
        let a = textureLoad(albedoIn, p, 0);
        let nd = textureLoad(normalDepthIn, p, 0);
        let n = vec4f(nd.xyz * 2.0 - 1.0, nd.a);
        weighted += vec4f(a.rgb * a.a, a.a);
        plain += a.rgb;
        normalWeighted += n * a.a;
        normalPlain += n;
    }
    var albedo = plain * 0.25;
    var nd = normalPlain * 0.25;
    if (weighted.a > 0.0) {
        albedo = weighted.rgb / weighted.a;
        nd = normalWeighted / weighted.a;
    }
    let normal = normalize(nd.xyz + vec3f(0.0, 1e-6, 0.0));
    textureStore(albedoOut, vec2i(gid.xy), vec4f(albedo, weighted.a * 0.25));
    textureStore(normalDepthOut, vec2i(gid.xy), vec4f(normal * 0.5 + 0.5, nd.a));
}
