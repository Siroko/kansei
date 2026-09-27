// Kansei cascaded sun (or moon) shadows for materials: the renderer's cascaded shadow map of its
// first shadow-casting directional light (group 3, bindings 10-12). Prepend
// `shadows::CASCADED_SHADOWS_WGSL` and multiply the sun's light by
//
//     kansei_sun_shadow(worldPos, N, fragCoord.xy)
//
// (1 lit, 0 shadowed). kansei_cascades.lightDirection and .lightColor are that light's direction
// of travel and colour times illuminance (lux), for shading it. Can be combined with
// SPOT_LIGHTS_WGSL (the helper names don't collide).

struct KanseiCascade {
    viewProj   : mat4x4f,   // world -> the cascade's orthographic light clip space
    texelWorld : f32,       // metres per shadow-map texel
    depthRange : f32,       // metres the cascade's [0, 1] depth spans
    radius     : f32,       // metres, half the cascade's width
    _pad       : f32,
}

struct KanseiCascades {
    cascades         : array<KanseiCascade, 4>,
    lightDirection   : vec3f,   // the direction the light travels
    count            : u32,     // cascades in use; 0: no cascaded shadows (everything lit)
    lightColor       : vec3f,   // colour times illuminance (lux)
    tanAngularRadius : f32,     // PCSS: the light's apparent radius; 0 = a fixed PCF kernel
    normalBias       : f32,     // receiver offset along the normal, in texels
    blend            : f32,     // fraction of a cascade's half-width it dithers into the next over
    maxDistance      : f32,     // shadows fade out toward this distance from the camera
    _pad0            : f32,
    cameraPos        : vec3f,
    _pad1            : f32,
}

@group(3) @binding(10) var kansei_cascade_atlas : texture_depth_2d_array;
@group(3) @binding(11) var<uniform> kansei_cascades : KanseiCascades;
@group(3) @binding(12) var kansei_cascade_sampler : sampler_comparison;

fn kansei_csm_noise(pixel: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
}

fn kansei_csm_vogel(i: u32, n: u32, phi: f32) -> vec2f {
    let r = sqrt((f32(i) + 0.5) / f32(n));
    let theta = f32(i) * 2.39996323 + phi;
    return r * vec2f(cos(theta), sin(theta));
}

// Visibility in cascade `c` at a light-space uv and depth ([0, 1]): PCSS when the light has an
// apparent size (penumbrae widen with the caster's distance above the receiver), else PCF.
fn kansei_cascade_visibility(c: u32, uv: vec2f, depth: f32, pixel: vec2f) -> f32 {
    let cascade = kansei_cascades.cascades[c];
    let texelUV = cascade.texelWorld / (2.0 * cascade.radius);
    let phi = kansei_csm_noise(pixel) * 6.2831853;
    var filterUV = texelUV * 1.5;
    if (kansei_cascades.tanAngularRadius > 0.0) {
        // search as far as the penumbra of a caster up to 50 m above the receiver reaches
        let searchUV = clamp(50.0 * kansei_cascades.tanAngularRadius / cascade.texelWorld, 1.5, 8.0) * texelUV;
        let dim = vec2f(textureDimensions(kansei_cascade_atlas));
        var blockerSum = 0.0;
        var blockers = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let t = vec2i(clamp((uv + kansei_csm_vogel(i, 8u, phi) * searchUV) * dim, vec2f(0.0), dim - 1.0));
            let d = textureLoad(kansei_cascade_atlas, t, i32(c), 0);
            if (d < depth) {
                blockerSum += d;
                blockers += 1.0;
            }
        }
        if (blockers == 0.0) { return 1.0; }
        // penumbra half-width at the receiver: its height above the blockers times tan(radius)
        let heightAbove = (depth - blockerSum / blockers) * cascade.depthRange;
        filterUV = clamp(heightAbove * kansei_cascades.tanAngularRadius / (2.0 * cascade.radius), texelUV, texelUV * 12.0);
    }
    var lit = 0.0;
    for (var i = 0u; i < 16u; i++) {
        lit += textureSampleCompareLevel(kansei_cascade_atlas, kansei_cascade_sampler,
                                         uv + kansei_csm_vogel(i, 16u, phi) * filterUV, i32(c), depth);
    }
    return lit / 16.0;
}

fn kansei_sun_shadow(worldPos: vec3f, N: vec3f, pixel: vec2f) -> f32 {
    let count = kansei_cascades.count;
    if (count == 0u) { return 1.0; }
    let toLight = -kansei_cascades.lightDirection;
    let cosTheta = saturate(dot(N, toLight));
    let slope = max(sqrt(1.0 - cosTheta * cosTheta), 0.2);
    let dither = kansei_csm_noise(pixel + vec2f(17.0, 59.0));
    // fade the shadows out over the last tenth of their reach
    let fade = saturate((distance(worldPos, kansei_cascades.cameraPos) / kansei_cascades.maxDistance - 0.9) * 10.0);

    for (var c = 0u; c < count; c++) {
        let cascade = kansei_cascades.cascades[c];
        let biased = worldPos + N * (kansei_cascades.normalBias * cascade.texelWorld * slope);
        let clip = cascade.viewProj * vec4f(biased, 1.0);
        let uv = vec2f(clip.x, -clip.y) * 0.5 + 0.5;
        // 0 at the cascade's edge, 1 at its centre; keep a filter's width inside
        let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y)) * 2.0;
        if (edge <= 16.0 * cascade.texelWorld / cascade.radius || clip.z > 1.0) { continue; }
        // near its edge, a cascade hands a dithered share of its pixels to the next one
        if (c + 1u < count && edge < kansei_cascades.blend && dither > edge / kansei_cascades.blend) { continue; }
        return mix(kansei_cascade_visibility(c, uv, clip.z, pixel), 1.0, fade);
    }
    return 1.0;
}

// The cascade that shadows a world point (without the dithered hand-over), or the cascade count
// beyond them all; for debug views.
fn kansei_sun_cascade(worldPos: vec3f) -> u32 {
    for (var c = 0u; c < kansei_cascades.count; c++) {
        let cascade = kansei_cascades.cascades[c];
        let clip = cascade.viewProj * vec4f(worldPos, 1.0);
        let uv = vec2f(clip.x, -clip.y) * 0.5 + 0.5;
        let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y)) * 2.0;
        if (edge > 16.0 * cascade.texelWorld / cascade.radius && clip.z <= 1.0) { return c; }
    }
    return kansei_cascades.count;
}
