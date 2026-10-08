// Ray-traced diffuse GI (rt::RtDiffuseGiEffect), the denoiser: Spatiotemporal Variance-Guided
// Filtering (Schied et al. 2017) at the trace resolution, three entry points.
//
// - `temporal`: reprojects the history (motion vectors where the materials wrote them, the
//   camera's motion elsewhere) with a bilinear footprint whose taps are each kept only where last
//   frame's guide (linear depth, normal) matches this surface, blends this frame's sample in
//   (1 / frames seen, no less than alphaColor), and keeps the luminance's first two moments for a
//   variance (in f32: the second overflows f16 past a luminance of 255). Writes this frame's guide
//   for the next one.
//
// The f16 targets and the wavelet's texels carry the standard deviation, not the variance (which
// would overflow f16 likewise); the passes square it to filter it.
// - `variance`: where fewer than 4 frames are seen, the variance from the moments of a 7x7
//   neighbourhood (edge-stopped) instead.
// - `atrous`: one iteration of the edge-avoiding a-trous wavelet (the 5x5 B3 kernel or the 3x3
//   1-2-1, taps `step` apart) with SVGF's depth (here the distance to the centre's tangent
//   plane), normal and variance-scaled luminance stops; the variance filtered by the squared
//   weights. The first iteration's output is next frame's colour history.
//
// Prefixed with rt_gi_common.wgsl.

@group(0) @binding(20) var<uniform> gp : RtGiParams;

// The view-space direction through a pixel, at unit depth (an unjittered symmetric projection).
fn viewRay(uv: vec2f) -> vec3f {
    return vec3f((uv.x * 2.0 - 1.0) * gp.invProj[0][0], (1.0 - uv.y * 2.0) * gp.invProj[1][1], -1.0);
}

// The view-space normal of world normal n.
fn viewNormal(n: vec3f) -> vec3f {
    return transpose(mat3x3f(gp.invView[0].xyz, gp.invView[1].xyz, gp.invView[2].xyz)) * n;
}

// ---- temporal ----

@group(0) @binding(21) var traceTex : texture_2d<f32>;
@group(0) @binding(22) var depthTex : texture_depth_2d;
@group(0) @binding(23) var normalTex : texture_2d<f32>;
@group(0) @binding(24) var velocityTex : texture_2d<f32>;
@group(0) @binding(25) var prevColor : texture_2d<f32>;
@group(0) @binding(26) var prevMoments : texture_2d<f32>;
@group(0) @binding(27) var prevGuide : texture_2d<u32>;
@group(0) @binding(28) var outIntegrated : texture_storage_2d<rgba16float, write>;
@group(0) @binding(29) var outMoments : texture_storage_2d<rgba32float, write>;
@group(0) @binding(30) var outGuide : texture_storage_2d<rg32uint, write>;

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let t = gid.xy;
    let px = giPixel(t);
    let depth = textureLoad(depthTex, px, 0);
    let raw = textureLoad(normalTex, px, 0);
    if (depth >= 1.0 || dot(raw.xyz, raw.xyz) < 1e-4) {
        textureStore(outIntegrated, t, vec4f(0.0));
        textureStore(outMoments, t, vec4f(0.0));
        textureStore(outGuide, t, vec4u(0u));
        return;
    }
    let n = normalize(raw.xyz * 2.0 - 1.0);
    let uv = giUv(px);
    let z = -giViewPos(uv, depth).z;
    let world = giWorldPos(uv, depth);
    textureStore(outGuide, t, giEncodeGuide(z, n));
    let sample = textureLoad(traceTex, t, 0).rgb;
    let l = giLuminance(sample);

    // where the surface was last frame
    let clip = gp.prevViewProj * vec4f(world, 1.0);
    var prevUv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    let v = textureLoad(velocityTex, px, 0).xy;
    if (abs(v.x) < RT_GI_NO_VELOCITY) {
        prevUv = uv - v;
    }
    let zExpected = clip.w;
    let tolerance = zExpected * (0.03 + 4.0 * gp.pixelSize * f32(gp.downscale));
    var color = vec3f(0.0);
    var moments = vec3f(0.0);
    var wsum = 0.0;
    if ((gp.flags & RT_GI_HISTORY) != 0u && clip.w > 0.0) {
        let tp = (prevUv * gp.fullSize - 0.5) / f32(gp.downscale);
        let t0 = vec2i(floor(tp));
        let f = tp - floor(tp);
        let last = vec2i(gp.traceSize) - 1;
        for (var k = 0; k < 4; k++) {
            let o = vec2i(k & 1, k >> 1u);
            let q = t0 + o;
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let g = giDecodeGuide(textureLoad(prevGuide, q, 0));
            if (g.x <= 0.0 || abs(g.x - zExpected) > tolerance || dot(g.yzw, n) < 0.9) { continue; }
            let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
            color += w * textureLoad(prevColor, q, 0).rgb;
            moments += w * textureLoad(prevMoments, q, 0).xyz;
            wsum += w;
        }
    }
    var frames = 1.0;
    var integrated = sample;
    var m = vec2f(l, l * l);
    if (wsum > 1e-3) {
        color /= wsum;
        moments /= wsum;
        frames = min(moments.z + 1.0, gp.maxHistory);
        let ac = max(gp.alphaColor, 1.0 / frames);
        let am = max(gp.alphaMoments, 1.0 / frames);
        integrated = mix(color, sample, ac);
        m = mix(moments.xy, m, am);
    }
    let variance = max(m.y - m.x * m.x, 0.0);
    textureStore(outIntegrated, t, vec4f(integrated, sqrt(variance)));
    textureStore(outMoments, t, vec4f(m, frames, 0.0));
}

// ---- variance (few frames seen) ----

@group(0) @binding(31) var integratedTex : texture_2d<f32>;
@group(0) @binding(32) var momentsTex : texture_2d<f32>;
@group(0) @binding(33) var guideTex : texture_2d<u32>;
// the wavelet's texel: the guide (x, y) and the colour and variance as four f16 (z, w), one load
@group(0) @binding(34) var<storage, read_write> outPacked : array<vec4u>;

fn giIndex(t: vec2i) -> u32 {
    return u32(t.y) * u32(gp.traceSize.x) + u32(t.x);
}

fn giPack(guide: vec4u, c: vec4f) -> vec4u {
    return vec4u(guide.x, guide.y, pack2x16float(c.rg), pack2x16float(c.ba));
}

fn giUnpackColor(p: vec4u) -> vec4f {
    return vec4f(unpack2x16float(p.z), unpack2x16float(p.w));
}

// The edge stops between the centre (view position pc, view normal nvc, world normal nc,
// luminance lc) and a neighbour `offset` trace texels away: invDepth is one over the plane
// distance allowed a texel away, invL one over the luminance's.
fn edgeWeight(pc: vec3f, nvc: vec3f, nc: vec3f, invDepth: f32, lc: f32, invL: f32, q: vec2i, offset: f32, g: vec4f, lq: f32) -> f32 {
    let uv = (vec2f(vec2u(q) * gp.downscale) + 0.5) / gp.fullSize;
    let pq = viewRay(uv) * g.x;
    let plane = abs(dot(nvc, pq - pc));
    let wz = plane * invDepth / max(offset, 1.0);
    var wn = max(dot(nc, g.yzw), 0.0);
    if (gp.phiNormal == 128.0) {
        // (seven squarings: pow's exp2 and log2 are quarter-rate)
        wn *= wn; wn *= wn; wn *= wn; wn *= wn; wn *= wn; wn *= wn; wn *= wn;
    } else {
        wn = pow(wn, gp.phiNormal);
    }
    let wl = abs(lc - lq) * invL;
    return wn * exp(-wz - wl);
}

fn depthStop(z: f32) -> f32 {
    return 1.0 / (gp.phiDepth * z * gp.pixelSize * f32(gp.downscale) + 1e-4);
}

@compute @workgroup_size(8, 8)
fn variance(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let t = vec2i(gid.xy);
    let c = textureLoad(integratedTex, t, 0);
    let mc = textureLoad(momentsTex, t, 0);
    let rawGuide = textureLoad(guideTex, t, 0);
    let gc = giDecodeGuide(rawGuide);
    if (gc.x <= 0.0 || mc.z >= 4.0) {
        outPacked[giIndex(t)] = giPack(rawGuide, c);
        return;
    }
    let uv = giUv(giPixel(gid.xy));
    let pc = viewRay(uv) * gc.x;
    let nvc = viewNormal(gc.yzw);
    let lc = giLuminance(c.rgb);
    let last = vec2i(gp.traceSize) - 1;
    var sum = vec3f(0.0);
    var sumM = vec2f(0.0);
    var wsum = 0.0;
    for (var y = -3; y <= 3; y++) {
        for (var x = -3; x <= 3; x++) {
            let q = t + vec2i(x, y);
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let g = giDecodeGuide(textureLoad(guideTex, q, 0));
            if (g.x <= 0.0) { continue; }
            let cq = textureLoad(integratedTex, q, 0);
            let w = edgeWeight(pc, nvc, gc.yzw, depthStop(gc.x), lc, 1.0 / (gp.phiColor * 10.0), q, length(vec2f(f32(x), f32(y))), g, giLuminance(cq.rgb));
            sum += w * cq.rgb;
            sumM += w * textureLoad(momentsTex, q, 0).xy;
            wsum += w;
        }
    }
    sum /= max(wsum, 1e-6);
    sumM /= max(wsum, 1e-6);
    // fewer frames: a larger variance, so the wavelet blurs more
    let v = max(sumM.y - sumM.x * sumM.x, 0.0) * 4.0 / max(mc.z, 1.0);
    outPacked[giIndex(t)] = giPack(rawGuide, vec4f(sum, sqrt(v)));
}

// ---- a-trous ----

struct AtrousParams {
    step     : u32,
    feedback : u32,   // 1: also write next frame's colour history
    last     : u32,   // 1: also write the result for the composite
    _pad0    : u32,
}

@group(0) @binding(35) var<uniform> ap : AtrousParams;
@group(0) @binding(36) var<storage, read> atrousIn : array<vec4u>;
@group(0) @binding(37) var<storage, read_write> atrousOut : array<vec4u>;
@group(0) @binding(38) var historyOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(39) var finalOut : texture_storage_2d<rgba16float, write>;

fn atrousStore(t: vec2i, packed: vec4u, c: vec4f) {
    atrousOut[giIndex(t)] = packed;
    if (ap.feedback != 0u) { textureStore(historyOut, t, c); }
    if (ap.last != 0u) { textureStore(finalOut, t, c); }
}

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let t = vec2i(gid.xy);
    let pcen = atrousIn[giIndex(t)];
    let c = giUnpackColor(pcen);
    let gc = giDecodeGuide(pcen);
    if (gc.x <= 0.0) {
        atrousStore(t, pcen, c);
        return;
    }
    let last = vec2i(gp.traceSize) - 1;
    // the variance, blurred (the centre and its four neighbours), for the luminance stop
    var vblur = 0.5 * c.a * c.a;
    for (var k = 0; k < 4; k++) {
        let o = select(vec2i(0, 2 * (k & 1) - 1), vec2i(2 * (k & 1) - 1, 0), k < 2);
        let sd = unpack2x16float(atrousIn[giIndex(clamp(t + o, vec2i(0), last))].w).y;
        vblur += 0.125 * sd * sd;
    }
    let invL = 1.0 / (gp.phiColor * sqrt(max(vblur, 0.0)) + 1e-6);
    let invDepth = depthStop(gc.x);
    let uv = giUv(giPixel(gid.xy));
    let pc = viewRay(uv) * gc.x;
    let nvc = viewNormal(gc.yzw);
    let lc = giLuminance(c.rgb);
    // the B3 spline's 5 taps, or (atrousRadius 1) the 1-2-1 kernel's 3
    let radius = i32(gp.atrousRadius);
    var kernel = array<f32, 3>(0.375, 0.25, 0.0625);
    if (radius == 1) {
        kernel = array<f32, 3>(0.5, 0.25, 0.0);
    }
    let h0 = kernel[0] * kernel[0];
    var sum = c.rgb * h0;
    var sumV = c.a * c.a * h0 * h0;
    var wsum = h0;
    let step = i32(ap.step);
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            if (x == 0 && y == 0) { continue; }
            let q = t + vec2i(x, y) * step;
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let pq = atrousIn[giIndex(q)];
            let g = giDecodeGuide(pq);
            if (g.x <= 0.0) { continue; }
            let cq = giUnpackColor(pq);
            let h = kernel[abs(x)] * kernel[abs(y)];
            let w = h * edgeWeight(pc, nvc, gc.yzw, invDepth, lc, invL, q, length(vec2f(f32(x), f32(y))) * f32(step), g, giLuminance(cq.rgb));
            sum += w * cq.rgb;
            sumV += w * w * cq.a * cq.a;
            wsum += w;
        }
    }
    let out = vec4f(sum / wsum, sqrt(sumV) / wsum);
    atrousStore(t, giPack(pcen, out), out);
}
