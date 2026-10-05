// Ray-traced reflections (rt::RtReflectionsEffect), resolve: at full resolution, the frame's
// traced pixels round each pixel upsampled (the four nearest, weighted by distance, depth and
// normal), blended with the history reprojected by the surface (clamped to the traced pixels'
// range, so a moving reflection doesn't ghost), and composited: the lit colour times 1 - F plus F
// times the reflection, F Schlick's Fresnel of the surface's F0 (lessened on rough surfaces).
// Writes the output and the next history. Prefixed with rt_reflect_common.wgsl.

@group(0) @binding(0) var<uniform> rp : RtReflectParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var inputTex : texture_2d<f32>;
@group(0) @binding(5) var traceTex : texture_2d<f32>;
@group(0) @binding(6) var historyTex : texture_2d<f32>;
@group(0) @binding(7) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var historyOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var linearSampler : sampler;

fn viewDepth(px: vec2u) -> f32 {
    return -rtViewPos((vec2f(px) + 0.5) / rp.fullSize, textureLoad(depthTex, px, 0)).z;
}

fn heat(x: f32) -> vec3f {
    let t = clamp(x, 0.0, 1.0);
    return clamp(vec3f(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= rp.fullSize)) { return; }
    let px = gid.xy;
    let color = textureLoad(inputTex, px, 0);
    let s = rtSurface(px);
    if (!s.valid) {
        textureStore(outTex, px, select(color, vec4f(0.0, 0.0, 0.0, 1.0), rp.view != 0u));
        textureStore(historyOut, px, vec4f(0.0));
        return;
    }
    // this frame's traced pixels round px: trace texel t is pixel t * downscale + jitter
    let d = rp.downscale;
    let j = rtJitter(rp.frame, d);
    let tp = (vec2f(px) - vec2f(j)) / f32(d);
    let t0 = vec2i(floor(tp));
    let f = tp - floor(tp);
    let z = viewDepth(px);
    let last = vec2i(rp.traceSize) - 1;
    var acc = vec4f(0.0);
    var wsum = 0.0;
    var lo = vec4f(1e30);
    var hi = vec4f(-1e30);
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1u);
        let t = clamp(t0 + o, vec2i(0), last);
        let sample = textureLoad(traceTex, vec2u(t), 0);
        if (sample.a <= 0.0) {
            continue;
        }
        let src = min(vec2u(t) * d + j, vec2u(rp.fullSize) - 1u);
        let ns = normalize(textureLoad(normalTex, src, 0).xyz * 2.0 - 1.0);
        let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let w = (bilinear + 1e-3) * exp(-abs(viewDepth(src) - z) / max(z * 0.02, 1e-3)) * pow(max(dot(ns, s.n), 0.0), 8.0);
        acc += w * sample;
        wsum += w;
        lo = min(lo, sample);
        hi = max(hi, sample);
    }
    let fresh = wsum > 1e-4;
    let current = acc / max(wsum, 1e-4);
    // the history, where the surface was last frame
    var history = vec4f(0.0);
    if ((rp.flags & RT_REFLECT_HISTORY) != 0u) {
        let clip = rp.prevViewProj * vec4f(s.world, 1.0);
        let uv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
        if (clip.w > 0.0 && all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
            history = textureSampleLevel(historyTex, linearSampler, uv, 0.0);
        }
    }
    var result = current;
    if (history.a > 0.5 && fresh) {
        let pad = (hi - lo) * 0.25 + 0.05 * hi;
        result = mix(clamp(history, lo - pad, hi + pad), current, rp.blend);
    } else if (history.a > 0.5) {
        result = history;
    } else if (!fresh) {
        result = vec4f(0.0);
    }
    textureStore(historyOut, px, vec4f(result.rgb, select(0.0, 1.0, fresh || history.a > 0.5)));
    let eye = rp.invView[3].xyz;
    let cosT = max(dot(normalize(eye - s.world), s.n), 0.0);
    let fresnel = s.f0 + (max(1.0 - s.roughness, s.f0) - s.f0) * pow(1.0 - cosT, 5.0);
    let reflection = result.rgb * rp.intensity;
    switch (rp.view) {
        case 1u: { textureStore(outTex, px, vec4f(fresnel * reflection, 1.0)); }
        case 2u: { textureStore(outTex, px, vec4f(reflection, 1.0)); }
        case 3u: { textureStore(outTex, px, vec4f(heat((current.a - 1.0) / 400.0) * rp.heatScale, 1.0)); }
        default: { textureStore(outTex, px, vec4f(color.rgb * (1.0 - fresnel) + fresnel * reflection, color.a)); }
    }
}
