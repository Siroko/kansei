// rt::RtShadowsEffect, the denoiser: `temporal` blends this frame's visibility with last frame's
// where the surface was (reprojected by the velocity target, else by depth), its taps kept only on
// the same surface (depth and normal); `atrous` is one step of an a-trous wavelet (a 3 x 3 B-spline
// spread `stepWidth` texels apart) with depth and normal edge stops, more of it where the history
// is short.

// ---- temporal ----

@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var velocityTex : texture_2d<f32>;
@group(0) @binding(4) var curA : texture_2d<f32>;
@group(0) @binding(5) var curB : texture_2d<f32>;
@group(0) @binding(6) var curGuide : texture_2d<f32>;
@group(0) @binding(7) var prevA : texture_2d<f32>;
@group(0) @binding(8) var prevB : texture_2d<f32>;
@group(0) @binding(9) var prevGuide : texture_2d<f32>;
@group(0) @binding(10) var outA : texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var outB : texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var outGuide : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) id: vec3u) {
    let t = id.xy;
    if (any(vec2f(t) >= sp.traceSize)) { return; }
    let g = textureLoad(curGuide, t, 0);
    let a = textureLoad(curA, t, 0);
    let b = textureLoad(curB, t, 0);
    if (g.z <= 0.0) {
        textureStore(outA, t, a);
        textureStore(outB, t, b);
        textureStore(outGuide, t, g);
        return;
    }
    let px = min(t * sp.downscale, vec2u(sp.fullSize) - 1u);
    let uv = (vec2f(px) + 0.5) / sp.fullSize;
    let depth = textureLoad(depthTex, px, 0);
    let world = shWorldPos(uv, depth);
    let n = shDecodeNormal(g.xy);
    let clip = sp.prevViewProj * vec4f(world, 1.0);
    var prevUv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    let v = textureLoad(velocityTex, px, 0).xy;
    if (abs(v.x) < RT_SHADOW_NO_VELOCITY) {
        prevUv = uv - v;
    }
    let tolerance = clip.w * 0.04;
    var ha = vec4f(0.0);
    var hb = vec4f(0.0);
    var len = 0.0;
    var wsum = 0.0;
    if ((sp.flags & RT_SHADOW_HISTORY) != 0u && clip.w > 0.0) {
        let tp = (prevUv * sp.fullSize - 0.5) / f32(sp.downscale);
        let t0 = vec2i(floor(tp));
        let f = tp - floor(tp);
        let last = vec2i(sp.traceSize) - 1;
        for (var k = 0; k < 4; k++) {
            let o = vec2i(k & 1, k >> 1u);
            let q = t0 + o;
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let pg = textureLoad(prevGuide, q, 0);
            if (pg.z <= 0.0 || abs(pg.z - clip.w) > tolerance || dot(shDecodeNormal(pg.xy), n) < 0.9) { continue; }
            let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
            ha += w * textureLoad(prevA, q, 0);
            hb += w * textureLoad(prevB, q, 0);
            len += w * pg.w;
            wsum += w;
        }
    }
    var frames = 1.0;
    var oa = a;
    var ob = b;
    if (wsum > 1e-3) {
        ha /= wsum;
        hb /= wsum;
        frames = min(len / wsum + 1.0, sp.maxHistory);
        let alpha = max(sp.temporalAlpha, 1.0 / frames);
        oa = mix(ha, a, alpha);
        ob = mix(hb, b, alpha);
    }
    textureStore(outA, t, oa);
    textureStore(outB, t, ob);
    textureStore(outGuide, t, vec4f(g.xyz, frames));
}

// ---- a-trous ----

@group(0) @binding(13) var inA : texture_2d<f32>;
@group(0) @binding(14) var inB : texture_2d<f32>;
@group(0) @binding(15) var guideTex : texture_2d<f32>;
@group(0) @binding(16) var filtA : texture_storage_2d<rgba16float, write>;
@group(0) @binding(17) var filtB : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) id: vec3u) {
    let t = vec2i(id.xy);
    if (any(vec2f(id.xy) >= sp.traceSize)) { return; }
    let g = textureLoad(guideTex, t, 0);
    let ca = textureLoad(inA, t, 0);
    let cb = textureLoad(inB, t, 0);
    if (g.z <= 0.0) {
        textureStore(filtA, t, ca);
        textureStore(filtB, t, cb);
        return;
    }
    let n = shDecodeNormal(g.xy);
    // a long history needs less of the wavelet
    let strength = clamp(1.5 - g.w / 12.0, 0.25, 1.0);
    // the 3 x 3 B-spline: 1/2 at the centre, 1/4 each side, each way
    let kernel = array<f32, 2>(0.5, 0.25);
    var sa = ca * 0.25;
    var sb = cb * 0.25;
    var wsum = 0.25;
    let step = i32(sp.stepWidth);
    let last = vec2i(sp.traceSize) - 1;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let q = t + vec2i(dx, dy) * step;
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let gq = textureLoad(guideTex, q, 0);
            if (gq.z <= 0.0) { continue; }
            let wn = pow(max(dot(shDecodeNormal(gq.xy), n), 0.0), sp.phiNormal);
            let wz = exp(-abs(gq.z - g.z) / (sp.phiDepth * g.z * f32(step) * 0.02 + 1e-4));
            let w = kernel[abs(dx)] * kernel[abs(dy)] * wn * wz * strength;
            sa += w * textureLoad(inA, q, 0);
            sb += w * textureLoad(inB, q, 0);
            wsum += w;
        }
    }
    textureStore(filtA, t, sa / wsum);
    textureStore(filtB, t, sb / wsum);
}
