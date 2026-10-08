// What every forest material shares (after ambient.wgsl, CASCADED_SHADOWS_WGSL, GBUFFER_OUT_WGSL,
// VOXEL_WRITE_WGSL, MOTION_VECTORS_WGSL and LOD_FADE_WGSL): the camera, a hash, the tangent frame
// from derivatives and the sun. The materials are the Raggare intro's (raggare-web's
// crates/intro/src/shaders), lit here by the sun through Kansei's cascades and the sky through
// ambient.wgsl instead of the film's dusk and headlights, and writing Kansei's GBuffer (albedo and
// normal for the GI) and voxels.
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

const PI: f32 = 3.14159265;

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Tangent frame from screen-space derivatives (no tangents in Kansei's vertex layout).
fn cotangent_frame(n: vec3<f32>, dp1: vec3<f32>, dp2: vec3<f32>, duv1: vec2<f32>, duv2: vec2<f32>) -> mat3x3<f32> {
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let inv = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    return mat3x3<f32>(t * inv, b * inv, n);
}

// The sun's irradiance on a surface facing n at `world` (lux), through the cascades.
fn sun_light(world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let nl = dot(n, -kansei_cascades.lightDirection);
    if (nl <= 0.0) { return vec3<f32>(0.0); }
    return kansei_cascades.lightColor * nl * kansei_sun_shadow(world, n, pixel);
}

// The sun's visibility for foliage, from CASCADED_SHADOWS_WGSL's cascades as `kansei_sun_shadow`
// picks them, but filtered by four comparison taps rather than its PCSS (24 texel reads): the
// foliage is drawn many layers deep and every layer shades (alpha-tested, it loses the GPU's
// hidden-surface removal), and in the thin sprays the penumbrae don't show; TAA smooths the
// taps' turn per pixel.
fn foliage_sun_shadow(world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> f32 {
    let count = kansei_cascades.count;
    if (count == 0u) { return 1.0; }
    let cos_theta = saturate(dot(n, -kansei_cascades.lightDirection));
    let slope = max(sqrt(1.0 - cos_theta * cos_theta), 0.2);
    let dither = kansei_csm_noise(pixel + vec2<f32>(17.0, 59.0));
    let fade = saturate((distance(world, kansei_cascades.cameraPos) / kansei_cascades.maxDistance - 0.9) * 10.0);
    for (var c = 0u; c < count; c++) {
        let cascade = kansei_cascades.cascades[c];
        let clip = cascade.viewProj * vec4<f32>(world + n * (kansei_cascades.normalBias * cascade.texelWorld * slope), 1.0);
        let uv = vec2<f32>(clip.x, -clip.y) * 0.5 + 0.5;
        let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y)) * 2.0;
        if (edge <= 16.0 * cascade.texelWorld / cascade.radius || clip.z > 1.0) { continue; }
        if (c + 1u < count && edge < kansei_cascades.blend && dither > edge / kansei_cascades.blend) { continue; }
        let texel_uv = cascade.texelWorld / (2.0 * cascade.radius);
        let phi = kansei_csm_noise(pixel) * 6.2831853;
        var lit = 0.0;
        for (var i = 0u; i < 4u; i++) {
            lit += textureSampleCompareLevel(kansei_cascade_atlas, kansei_cascade_sampler, uv + kansei_csm_vogel(i, 4u, phi) * texel_uv * 1.5, i32(c), clip.z);
        }
        return mix(lit * 0.25, 1.0, fade);
    }
    return 1.0;
}

// The sun on a leaf, a spray of needles or a blade: thin and translucent, it takes the light from
// either side, a grazing one still catching FOLIAGE_WRAP of it (the film's headlights_foliage).
const FOLIAGE_WRAP: f32 = 0.5;
fn sun_light_foliage(world: vec3<f32>, n: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let nl = dot(n, -kansei_cascades.lightDirection);
    // the shadow's normal offset toward the side the light comes from
    let facing = select(-n, n, nl >= 0.0);
    // the far side passes on less than the lit one
    let side = select(0.6, 1.0, nl >= 0.0);
    return kansei_cascades.lightColor * mix(FOLIAGE_WRAP, 1.0, abs(nl)) * side * foliage_sun_shadow(world, facing, pixel);
}

// GGX specular (Smith height-correlated visibility, Schlick Fresnel) for one light direction.
fn ggx_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: f32) -> f32 {
    let h = normalize(v + l);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let a = max(roughness * roughness, 2e-3);
    let a2 = a * a;
    let dn = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dn * dn);
    let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-5);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    return d * vis * f * nl;
}

// The camera's position, from the view matrix.
fn view_eye() -> vec3<f32> {
    let r = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    return -(transpose(r) * view_matrix[3].xyz);
}

// Whether a culled instance's LOD crossfade drops this pixel, turned each frame.
fn lod_fade_out(fade: f32, pixel: vec2<f32>) -> bool {
    return kansei_lod_fade_discard(fade, pixel, kansei_camera_temporal.frame);
}
