// Planar reflections in materials (reflections::PLANAR_REFLECTION_WGSL). Bind
// `PlanarReflection::material_texture()` as a texture_2d<f32> and a trilinear sampler, then:
//
//     let uv = kansei_screen_uv(in.clip_pos);                       // the pixel's screen uv
//     let offset = kansei_reflection_offset(view_matrix, up, n, 0.05); // ripples
//     let r = kansei_planar_reflection(tex, samp, uv, offset, roughness);
//
// r.rgb is the reflected radiance, r.a the length of the reflected path from the camera in
// metres (fog it with that), >= KANSEI_REFLECTION_SKY where the mirrored view saw no geometry.

const KANSEI_REFLECTION_SKY : f32 = 65504.0;

// Screen uv ([0, 1], y down) of a clip-space position (pass the vertex shader's clip position
// through as a varying; @builtin(position) in the fragment stage is already in pixels).
fn kansei_screen_uv(clip: vec4f) -> vec2f {
    let ndc = clip.xy / clip.w;
    return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// Screen-space offset of the lookup from a perturbed surface normal: its deviation from the
// plane normal, in view space, times `strength` (uv per unit of tilt; 0.02-0.1 for ripples).
fn kansei_reflection_offset(viewMatrix: mat4x4f, planeNormal: vec3f, normal: vec3f, strength: f32) -> vec2f {
    let d = (viewMatrix * vec4f(normal - planeNormal, 0.0)).xy;
    return vec2f(d.x, -d.y) * strength;
}

// The reflection seen at `screenUV`, displaced by `offset` and blurred by `roughness` (0 mirror,
// 1 the smallest mip). The reflection is rendered mirrored left-right, hence the flip.
fn kansei_planar_reflection(tex: texture_2d<f32>, samp: sampler, screenUV: vec2f, offset: vec2f, roughness: f32) -> vec4f {
    let uv = clamp(vec2f(1.0 - (screenUV.x + offset.x), screenUV.y + offset.y), vec2f(0.0), vec2f(1.0));
    let lod = saturate(roughness) * f32(textureNumLevels(tex) - 1u);
    return textureSampleLevel(tex, samp, uv, lod);
}
