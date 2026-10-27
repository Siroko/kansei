// Screen-space ambient occlusion, the composite (after ssao_common.wgsl, sky_lighting.wgsl and
// ssao_blur.wgsl): the occlusion blurred down, then the occluded share of the ambient light taken
// out of the colour, the direct light and the emission left as they are. The ambient is the
// albedo times the sky's light around the normal: the hemisphere (`skyUp`, `skyDown`) blended by
// the normal's height, as Material::standard_lit adds it, or the irradiance of the sky's SH over
// pi; for unlit materials, the whole colour less its emissive part. Glass (an albedo alpha under
// 0.5) and pixels with no albedo have no ambient of their own and keep their colour.

@group(0) @binding(4) var colorTex    : texture_2d<f32>;
@group(0) @binding(5) var albedoTex   : texture_2d<f32>;
@group(0) @binding(6) var emissiveTex : texture_2d<f32>;
@group(0) @binding(7) var<uniform> sky : SkyLighting;
@group(0) @binding(8) var outTex      : texture_storage_2d<rgba16float, write>;

// The ambient light in the colour `color` at pixel `px`.
fn ssaoAmbient(px: vec2i, color: vec3f) -> vec3f {
    let g = ssaoGBufferPixel(px);
    if (params.ambient == 2u) {
        return max(color - textureLoad(emissiveTex, g, 0).rgb, vec3f(0.0));
    }
    let albedo = textureLoad(albedoTex, g, 0);
    if (albedo.a < 0.5 || all(albedo.rgb <= vec3f(0.0))) { return vec3f(0.0); }
    let viewN = ssaoViewNormal(px, ssaoViewPosAt(px));
    let n = normalize((params.invView * vec4f(viewN, 0.0)).xyz);
    if (params.ambient == 1u) {
        return albedo.rgb * skyIrradiance(sky, n) / SSAO_PI;
    }
    return albedo.rgb * mix(params.skyDown.rgb, params.skyUp.rgb, n.y * 0.5 + 0.5);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= params.size)) { return; }
    let px = vec2i(gid.xy);
    let color = textureLoad(colorTex, px, 0);
    let ao = ssaoBlur(px, vec2i(0, 1));
    if (params.debug != 0u) {
        textureStore(outTex, px, vec4f(vec3f(ao.x), color.a));
        return;
    }
    let occluded = (1.0 - ao.x) * params.strength;
    if (ao.y <= 0.0 || occluded <= 0.0) {
        textureStore(outTex, px, color);
        return;
    }
    let result = max(color.rgb - ssaoAmbient(px, color.rgb) * occluded, vec3f(0.0));
    textureStore(outTex, px, vec4f(result, color.a));
}
