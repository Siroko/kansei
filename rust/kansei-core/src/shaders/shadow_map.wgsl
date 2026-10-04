// Kansei's single directional shadow map and point-light cube shadow for materials (group 3
// bindings 0-4). The map (`Renderer::enable_shadows`) is rendered from the scene's first
// directional light, the cube (`enable_point_shadows`) from the first shadow-casting point light,
// at `kansei_shadow.pointLightPos`. Prepend `shadows::SHADOW_MAP_WGSL` and multiply those lights by
//
//     kansei_shadow_map(worldPos, N)        kansei_point_shadow(worldPos)
//
// (1 lit, 0 shadowed; 1 when off). Both may be called in non-uniform control flow.

struct KanseiShadowMap {
    viewProj            : mat4x4f,
    bias                : f32,
    normalBias          : f32,
    enabled             : f32,
    pointShadowEnabled  : f32,
    pointLightPos       : vec3f,
    pointShadowFar      : f32,
}

@group(3) @binding(0) var kansei_shadow_depth : texture_depth_2d;
@group(3) @binding(1) var kansei_shadow_sampler : sampler_comparison;
@group(3) @binding(2) var<uniform> kansei_shadow : KanseiShadowMap;
@group(3) @binding(3) var kansei_point_shadow_faces : texture_2d_array<f32>;

fn kansei_shadow_map(worldPos: vec3f, N: vec3f) -> f32 {
    if (kansei_shadow.enabled < 0.5) { return 1.0; }
    let clip = kansei_shadow.viewProj * vec4f(worldPos + N * kansei_shadow.normalBias, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x, -ndc.y) * 0.5 + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0) { return 1.0; }
    let texel = 1.0 / vec2f(textureDimensions(kansei_shadow_depth));
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(kansei_shadow_depth, kansei_shadow_sampler, uv + vec2f(f32(x), f32(y)) * texel, ndc.z - kansei_shadow.bias);
        }
    }
    return lit / 9.0;
}

// Whether `light` (a KanseiPointLight's position) is the one the cube shadow was rendered from.
fn kansei_is_point_shadow_light(position: vec3f) -> bool {
    return kansei_shadow.pointShadowEnabled > 0.5 && all(abs(position - kansei_shadow.pointLightPos) < vec3f(1e-3));
}

fn kansei_point_shadow(worldPos: vec3f) -> f32 {
    if (kansei_shadow.pointShadowEnabled < 0.5) { return 1.0; }
    let toFrag = worldPos - kansei_shadow.pointLightPos;
    let dist = length(toFrag);
    let dir = toFrag / max(dist, 1e-6);
    let a = abs(dir);
    var face: i32;
    var uv: vec2f;
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x > 0.0) { face = 0; uv = vec2f(-dir.z, -dir.y) / a.x; } else { face = 1; uv = vec2f(dir.z, -dir.y) / a.x; }
    } else if (a.y >= a.x && a.y >= a.z) {
        if (dir.y > 0.0) { face = 2; uv = vec2f(dir.x, dir.z) / a.y; } else { face = 3; uv = vec2f(dir.x, -dir.z) / a.y; }
    } else {
        if (dir.z > 0.0) { face = 4; uv = vec2f(dir.x, -dir.y) / a.z; } else { face = 5; uv = vec2f(-dir.x, -dir.y) / a.z; }
    }
    // texel rows run down the face as rendered (NDC y up), so v flips
    // (cubemap_shadow_map.rs: the_point_shadow_lookup_finds_the_texel_each_face_rendered)
    uv = vec2f(uv.x, -uv.y) * 0.5 + 0.5;
    let size = vec2f(textureDimensions(kansei_point_shadow_faces));
    let texel = vec2i(clamp(uv * size, vec2f(0.0), size - 1.0));
    let stored = textureLoad(kansei_point_shadow_faces, texel, face, 0).r;
    return select(1.0, 0.0, dist - 0.05 > stored);
}
