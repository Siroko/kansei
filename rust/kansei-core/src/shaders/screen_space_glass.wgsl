// Screen-space glass (postprocessing/effects/screen_space_glass.rs): the pixels whose material
// wrote glass (GBUFFER_OUT_WGSL's kansei_gbuffer_out_glass, or StandardLitOptions::glass: the tint
// in the albedo, its alpha 0.45 - 0.4 x roughness, 1 / ior in the normal's alpha) show the scene
// behind them from the GBuffer's background (the opaque colour copied before the transmissive
// renderables, MaterialOptions::transmissive), refracted, plus their Fresnel reflection of the sky.
//
// Each pixel's glass is taken for a ball `thickness` across touching its surface there (a sphere
// proxy): the view ray bends into it, crosses it, bends out of its far side (or reflects back in
// where it can't leave), and the background is read where it points `behind` metres on (each
// channel at its own index of refraction for the dispersion), unless something nearer than the
// glass is drawn there (the background holds it too), where the unbent background stands in. So a
// ball turns what is behind it over and magnifies it as a lens does; a flat face bends it less
// than it should. The tint absorbs over the path inside and a frosted surface blurs the
// background. A cheap stand-in for rt::RtReflectionsEffect's glass: no exact second surface, a
// guessed depth behind, no glass behind glass.

struct SsGlassParams {
    viewProj    : mat4x4f,
    invViewProj : mat4x4f,
    eye         : vec3f,
    // the sphere proxy's diameter (m)
    thickness   : f32,
    size        : vec2f,
    // the spread of the index of refraction from red to blue (a share of ior - 1)
    dispersion  : f32,
    // the background's blur at full frost, pixels
    frostPx     : f32,
    skyScale    : f32,
    hasSky      : u32,
    // how far behind the glass the background is taken to be (m)
    behind      : f32,
    _pad        : f32,
}

@group(0) @binding(0) var<uniform> params : SsGlassParams;
@group(0) @binding(1) var inputTex : texture_2d<f32>;
@group(0) @binding(2) var backgroundTex : texture_2d<f32>;
@group(0) @binding(3) var depthTex : texture_depth_2d;
@group(0) @binding(4) var normalTex : texture_2d<f32>;
@group(0) @binding(5) var albedoTex : texture_2d<f32>;
@group(0) @binding(6) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var linearClamp : sampler;
@group(0) @binding(8) var<uniform> sky : SkyLighting;

fn ssgWorld(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = params.invViewProj * ndc;
    return w.xyz / w.w;
}

// Where world point p falls on the screen (uv), and its depth.
fn ssgProject(p: vec3f) -> vec3f {
    let c = params.viewProj * vec4f(p, 1.0);
    let ndc = c.xyz / max(c.w, 1e-6);
    return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

// The background behind the glass along a ray bent at p (surface normal n facing the eye, view
// direction v) by 1 / `eta` into the sphere proxy and out of its far side; `own` is this pixel's
// uv. Its w: the path's length inside.
fn ssgThrough(p: vec3f, v: vec3f, n: vec3f, eta: f32, own: vec2f, glassDepth: f32, blur: f32) -> vec4f {
    let d = refract(v, n, eta);
    var uv = own;
    var inside = 0.0;
    if (dot(d, d) > 0.5) {
        // across the ball of radius r whose surface touches p with normal n
        let r = 0.5 * params.thickness;
        let centre = p - n * r;
        inside = max(-2.0 * r * dot(n, d), 0.0);
        let exit = p + d * inside;
        let out = normalize(exit - centre);
        var leave = refract(d, -out, 1.0 / eta);
        if (dot(leave, leave) < 0.5) {
            leave = reflect(d, -out);
        }
        let q = ssgProject(exit + leave * params.behind);
        // something nearer than the glass drawn there is in the background too: not behind it
        let px = vec2i(clamp(q.xy, vec2f(0.0), vec2f(0.9999)) * params.size);
        if (all(q.xy >= vec2f(0.0)) && all(q.xy <= vec2f(1.0)) && textureLoad(depthTex, px, 0) >= glassDepth - 1e-5) {
            uv = q.xy;
        }
    }
    if (blur < 0.5) {
        return vec4f(textureSampleLevel(backgroundTex, linearClamp, uv, 0.0).rgb, inside);
    }
    // frosted: a disc of taps
    var sum = vec3f(0.0);
    for (var k = 0u; k < 8u; k++) {
        let a = f32(k) * 2.3999632;
        let r = sqrt((f32(k) + 0.5) / 8.0) * blur;
        sum += textureSampleLevel(backgroundTex, linearClamp, uv + vec2f(cos(a), sin(a)) * r / params.size, 0.0).rgb;
    }
    return vec4f(sum / 8.0, inside);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let px = vec2i(gid.xy);
    if (any(vec2f(gid.xy) >= params.size)) {
        return;
    }
    let color = textureLoad(inputTex, px, 0);
    let depth = textureLoad(depthTex, px, 0);
    let albedo = textureLoad(albedoTex, px, 0);
    // glass: the albedo's alpha 0.05 to 0.45 (rt_reflect_common.wgsl's rtIsGlassAlbedo)
    if (depth >= 1.0 || albedo.a <= 0.025 || albedo.a >= 0.475) {
        textureStore(outTex, px, color);
        return;
    }
    let raw = textureLoad(normalTex, px, 0);
    let eta = clamp(raw.w, 0.05, 1.0);
    let ior = 1.0 / eta;
    let roughness = clamp((0.45 - albedo.a) / 0.4, 0.0, 1.0);
    let uv = (vec2f(gid.xy) + 0.5) / params.size;
    let p = ssgWorld(uv, depth);
    let v = normalize(p - params.eye);
    var n = normalize(raw.xyz * 2.0 - 1.0);
    if (dot(n, v) > 0.0) {
        n = -n;
    }
    // Schlick's Fresnel at the surface
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let cosI = clamp(-dot(v, n), 0.0, 1.0);
    let fr = f0 + (1.0 - f0) * pow(1.0 - cosI, 5.0);
    // the light through it, each channel bent by its own index of refraction
    let blur = roughness * params.frostPx;
    let spread = params.dispersion * (ior - 1.0);
    let etas = vec3f(1.0 / (ior - spread), eta, 1.0 / (ior + spread));
    let green = ssgThrough(p, v, n, etas.g, uv, depth, blur);
    var through = vec3f(ssgThrough(p, v, n, etas.r, uv, depth, blur).r, green.g, ssgThrough(p, v, n, etas.b, uv, depth, blur).b);
    through *= pow(clamp(albedo.rgb, vec3f(1e-4), vec3f(1.0)), vec3f(green.w));
    // the reflection: the sky round the reflected ray (low-pass, as rough as it is)
    var reflection = vec3f(0.0);
    if (params.hasSky != 0u) {
        reflection = skyRadiance(sky, reflect(v, n)) * params.skyScale;
    }
    let result = color.rgb + fr * reflection + (1.0 - fr) * through;
    textureStore(outTex, px, vec4f(result, color.a));
}
