// The path tracer's denoiser (PathTracerDenoise): Spatiotemporal Variance-Guided Filtering
// (Schied et al. 2017) of the tracer's guide buffers (trace.wgsl, PT_GUIDES), the diffuse and the
// specular light apart, each over its albedo. Four entry points:
//
// - `temporal`: reprojects last frame's history along the tracer's motion with a bilinear
//   footprint whose taps are each kept only where last frame's surface (linear depth, normal)
//   is this one's, holds it within the 3x3 neighbourhood's mean and spread (`clamp_sigma`
//   standard deviations), blends this frame's sample in (1 / frames seen, no less than
//   `alpha_color`), and keeps both signals' luminance moments for their variance (in f32: the
//   second overflows f16 past a luminance of 255). Writes this frame's surface and history
//   length for the next one.
// - `variance`: where fewer than 4 frames are seen, the variance from the moments of a 7x7
//   neighbourhood (edge-stopped), and the light averaged over it, instead.
// - `atrous`: one iteration of the edge-avoiding a-trous wavelet (the 5x5 B3 kernel or the 3x3
//   1-2-1, taps `step` apart) with SVGF's depth (the distance to the centre's tangent plane),
//   normal and luminance stops, the luminance's scaled by the variance prefiltered 3x3; the
//   variance filtered by the squared weights. The specular light's neighbours count by the lobe's
//   roughness: a mirror or clear glass shows what it reflects or refracts sharp, and only the
//   temporal passes average it. The iteration `feedback` names writes next frame's history.
// - `composite`: the filtered light times its albedos plus what bypassed the filter, or a guide
//   buffer to look at (`view`).
// - `resolve`: temporal anti-aliasing of the composite (as SVGF ends; the TAA resolve's method,
//   shaders/taa_resolve.wgsl, at 1:1): the jittered frames' 3x3 neighbourhood reconstructed at
//   the pixel's centre, last frame's output reprojected along the nearest surface's motion
//   (Catmull-Rom), clipped toward the neighbourhood's variance box, dropped where it saw another
//   depth. The light is demodulated per frame by the albedo its jittered sample hit, so without
//   it edges take one side's albedo a frame and never average.
//
// The f16 words and textures carry the standard deviation, not the variance (which would
// overflow f16 likewise); the passes square it to filter it.

struct SvgfParams {
    view_matrix   : mat4x4f,  // world -> view (normals into view space)
    size          : vec2f,    // pixels
    proj_scale    : vec2f,    // tan of the half field of view, x and y (the view ray at unit depth)
    alpha_color   : f32,      // the temporal blend floors
    alpha_moments : f32,
    max_history   : f32,
    clamp_sigma   : f32,      // 0: no neighbourhood clamp
    phi_color     : f32,      // the edge stops
    phi_normal    : f32,
    phi_depth     : f32,
    pixel_size    : f32,      // a pixel's height at unit depth
    view          : u32,      // SVGF_VIEW_*
    flags         : u32,      // SVGF_*
    heat_scale    : f32,
    radius        : u32,      // the wavelet's taps each way: 2 (5x5) or 1 (3x3)
    jitter        : vec2f,    // this frame's samples' offset from the pixel centres, pixels
    taa_feedback  : f32,      // the resolve's history weight when still
    taa_gamma     : f32,      // its clip box half-size in standard deviations
}

// last frame's history is this view's (off: the first frame, or a reset)
const SVGF_HISTORY : u32 = 1u;
// filter (off: the composite shows the tracer's light unfiltered)
const SVGF_FILTER : u32 = 2u;
// the wavelet runs (off: temporal accumulation alone)
const SVGF_SPATIAL : u32 = 4u;
// last frame's resolve output is this view's
const SVGF_TAA_HISTORY : u32 = 8u;

const SVGF_VIEW_FINAL           : u32 = 0u;
const SVGF_VIEW_NOISY           : u32 = 1u;
const SVGF_VIEW_DIFFUSE_ALBEDO  : u32 = 2u;
const SVGF_VIEW_SPECULAR_ALBEDO : u32 = 3u;
const SVGF_VIEW_NORMAL          : u32 = 4u;
const SVGF_VIEW_DEPTH           : u32 = 5u;
const SVGF_VIEW_MOTION          : u32 = 6u;
const SVGF_VIEW_DIFFUSE         : u32 = 7u;
const SVGF_VIEW_SPECULAR        : u32 = 8u;
const SVGF_VIEW_VARIANCE        : u32 = 9u;
const SVGF_VIEW_HISTORY         : u32 = 10u;
const SVGF_VIEW_TEMPORAL        : u32 = 11u;

@group(0) @binding(0) var<uniform> sp : SvgfParams;

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn oct_decode(word: u32) -> vec3f {
    let p = unpack2x16snorm(word);
    var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    n = vec3f(n.xy + select(vec2f(t), vec2f(-t), n.xy >= vec2f(0.0)), n.z);
    return normalize(n);
}

// The light the signal word carries: diffuse (rgb) and specular (rgb), each over its albedo.
fn signal_diffuse(s: vec4u) -> vec3f {
    return vec3f(unpack2x16float(s.x), unpack2x16float(s.y).x);
}

fn signal_specular(s: vec4u) -> vec3f {
    return vec3f(unpack2x16float(s.y).y, unpack2x16float(s.z));
}

// What bypasses the filter (sky, emitters), from the signal and albedo words.
fn bypass(s: vec4u, a: vec4u) -> vec3f {
    return vec3f(unpack2x16float(s.w), unpack2x16float(a.w).x);
}

// How much the specular light's neighbours count, from the albedo word's lobe roughness: none
// for a mirror or clear glass, fully from a roughness of `SPECULAR_BLUR_ROUGHNESS`.
const SPECULAR_BLUR_ROUGHNESS : f32 = 0.25;

fn specular_spread(a: vec4u) -> f32 {
    return saturate(unpack2x16float(a.w).y / SPECULAR_BLUR_ROUGHNESS);
}

// The view-space position of the pixel at `t` and linear depth z.
fn view_pos(t: vec2i, z: f32) -> vec3f {
    let ndc = (vec2f(t) + 0.5) / sp.size * 2.0 - 1.0;
    return vec3f(ndc.x * sp.proj_scale.x, -ndc.y * sp.proj_scale.y, -1.0) * z;
}

fn view_normal(n: vec3f) -> vec3f {
    return (sp.view_matrix * vec4f(n, 0.0)).xyz;
}

// ---- temporal ----

@group(0) @binding(1) var signalTex : texture_2d<u32>;
@group(0) @binding(2) var surfaceTex : texture_2d<u32>;
@group(0) @binding(3) var prevDiffuse : texture_2d<f32>;
@group(0) @binding(4) var prevSpecular : texture_2d<f32>;
@group(0) @binding(5) var prevMoments : texture_2d<f32>;
@group(0) @binding(6) var prevSurface : texture_2d<u32>;
@group(0) @binding(7) var outDiffuse : texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var outSpecular : texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var outMoments : texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var outSurface : texture_storage_2d<rgba32uint, write>;

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.size)) { return; }
    let t = vec2i(gid.xy);
    let surface = textureLoad(surfaceTex, t, 0);
    let z = bitcast<f32>(surface.x);
    if (z <= 0.0) {
        textureStore(outDiffuse, t, vec4f(0.0));
        textureStore(outSpecular, t, vec4f(0.0));
        textureStore(outMoments, t, vec4f(0.0));
        textureStore(outSurface, t, vec4u(0u));
        return;
    }
    let n = oct_decode(surface.y);
    let signal = textureLoad(signalTex, t, 0);
    let diffuse = signal_diffuse(signal);
    let specular = signal_specular(signal);
    let last = vec2i(sp.size) - 1;

    // the 3x3 neighbourhood's mean and spread, for the clamp
    var m1d = vec3f(0.0);
    var m2d = vec3f(0.0);
    var m1s = vec3f(0.0);
    var m2s = vec3f(0.0);
    var count = 0.0;
    if (sp.clamp_sigma > 0.0) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let q = clamp(t + vec2i(x, y), vec2i(0), last);
                if (bitcast<f32>(textureLoad(surfaceTex, q, 0).x) <= 0.0) { continue; }
                let s = textureLoad(signalTex, q, 0);
                let d = signal_diffuse(s);
                let sq = signal_specular(s);
                m1d += d;
                m2d += d * d;
                m1s += sq;
                m2s += sq * sq;
                count += 1.0;
            }
        }
        m1d /= count;
        m2d /= count;
        m1s /= count;
        m2s /= count;
    }

    // where the surface was last frame, and its depth then
    let motion = unpack2x16float(surface.z);
    let z_expected = bitcast<f32>(surface.w);
    let prev_px = (vec2f(t) + 0.5) - motion * sp.size - 0.5;
    let tolerance = z_expected * (0.03 + 4.0 * sp.pixel_size);
    var hd = vec3f(0.0);
    var hs = vec3f(0.0);
    var hm = vec4f(0.0);
    var frames = 0.0;
    var wsum = 0.0;
    if ((sp.flags & SVGF_HISTORY) != 0u && z_expected > 0.0) {
        let p0 = vec2i(floor(prev_px));
        let f = prev_px - floor(prev_px);
        for (var k = 0; k < 4; k++) {
            let o = vec2i(k & 1, k >> 1u);
            let q = p0 + o;
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let g = textureLoad(prevSurface, q, 0);
            let gz = bitcast<f32>(g.x);
            if (gz <= 0.0 || abs(gz - z_expected) > tolerance || dot(oct_decode(g.y), n) < 0.9) { continue; }
            let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
            hd += w * textureLoad(prevDiffuse, q, 0).rgb;
            hs += w * textureLoad(prevSpecular, q, 0).rgb;
            hm += w * textureLoad(prevMoments, q, 0);
            frames += w * bitcast<f32>(g.z);
            wsum += w;
        }
    }

    let ld = luminance(diffuse);
    let ls = luminance(specular);
    var md = vec2f(ld, ld * ld);
    var ms = vec2f(ls, ls * ls);
    var id = diffuse;
    var is = specular;
    var seen = 1.0;
    if (wsum > 1e-3) {
        hd /= wsum;
        hs /= wsum;
        hm /= wsum;
        seen = min(frames / wsum + 1.0, sp.max_history);
        if (sp.clamp_sigma > 0.0) {
            let sd = sqrt(max(m2d - m1d * m1d, vec3f(0.0))) * sp.clamp_sigma;
            let ss = sqrt(max(m2s - m1s * m1s, vec3f(0.0))) * sp.clamp_sigma;
            hd = clamp(hd, m1d - sd, m1d + sd);
            hs = clamp(hs, m1s - ss, m1s + ss);
        }
        let ac = max(sp.alpha_color, 1.0 / seen);
        let am = max(sp.alpha_moments, 1.0 / seen);
        id = mix(hd, diffuse, ac);
        is = mix(hs, specular, ac);
        md = mix(hm.xy, md, am);
        ms = mix(hm.zw, ms, am);
    }
    let vd = max(md.y - md.x * md.x, 0.0);
    let vs = max(ms.y - ms.x * ms.x, 0.0);
    textureStore(outDiffuse, t, vec4f(id, sqrt(vd)));
    textureStore(outSpecular, t, vec4f(is, sqrt(vs)));
    textureStore(outMoments, t, vec4f(md, ms));
    textureStore(outSurface, t, vec4u(surface.x, surface.y, bitcast<u32>(seen), 0u));
}

// ---- the wavelet's texel ----

// The surface (linear depth, normal) and both signals with their standard deviations as f16:
// what one tap reads, in one load.
struct Texel {
    surface  : vec2u,
    diffuse  : vec2u,
    specular : vec2u,
}

fn texel_index(t: vec2i) -> u32 {
    return u32(t.y) * u32(sp.size.x) + u32(t.x);
}

fn pack_light(c: vec4f) -> vec2u {
    return vec2u(pack2x16float(c.rg), pack2x16float(c.ba));
}

fn unpack_light(p: vec2u) -> vec4f {
    return vec4f(unpack2x16float(p.x), unpack2x16float(p.y));
}

// The edge stops between the centre (view position pc, view normal nvc, world normal nc) and a
// neighbour `offset` pixels away (view position pq, world normal nq): invDepth is one over the
// plane distance allowed a pixel away.
fn geometry_weight(pc: vec3f, nvc: vec3f, nc: vec3f, inv_depth: f32, pq: vec3f, nq: vec3f, offset: f32) -> f32 {
    let wz = abs(dot(nvc, pq - pc)) * inv_depth / max(offset, 1.0);
    var wn = max(dot(nc, nq), 0.0);
    if (sp.phi_normal == 128.0) {
        // (seven squarings: pow's exp2 and log2 are quarter-rate)
        wn *= wn; wn *= wn; wn *= wn; wn *= wn; wn *= wn; wn *= wn; wn *= wn;
    } else {
        wn = pow(wn, sp.phi_normal);
    }
    return wn * exp(-wz);
}

fn depth_stop(z: f32) -> f32 {
    return 1.0 / (sp.phi_depth * z * sp.pixel_size + 1e-4);
}

// ---- variance (few frames seen) ----

@group(0) @binding(11) var integratedDiffuse : texture_2d<f32>;
@group(0) @binding(12) var integratedSpecular : texture_2d<f32>;
@group(0) @binding(13) var momentsTex : texture_2d<f32>;
@group(0) @binding(14) var historySurface : texture_2d<u32>;
@group(0) @binding(15) var<storage, read_write> outTexels : array<Texel>;
@group(0) @binding(35) var varianceAlbedo : texture_2d<u32>;

@compute @workgroup_size(8, 8)
fn variance(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.size)) { return; }
    let t = vec2i(gid.xy);
    let d = textureLoad(integratedDiffuse, t, 0);
    let s = textureLoad(integratedSpecular, t, 0);
    let g = textureLoad(historySurface, t, 0);
    let z = bitcast<f32>(g.x);
    let seen = bitcast<f32>(g.z);
    if (z <= 0.0 || seen >= 4.0 || (sp.flags & SVGF_SPATIAL) == 0u) {
        outTexels[texel_index(t)] = Texel(g.xy, pack_light(d), pack_light(s));
        return;
    }
    let n = oct_decode(g.y);
    let pc = view_pos(t, z);
    let nvc = view_normal(n);
    let inv_depth = depth_stop(z);
    let ldc = luminance(d.rgb);
    let lsc = luminance(s.rgb);
    // the luminance stops: the variance isn't known yet, so a broad one
    let inv_ld = 1.0 / (sp.phi_color * 10.0 * max(d.a, 1e-3));
    let inv_ls = 1.0 / (sp.phi_color * 10.0 * max(s.a, 1e-3));
    let spread = specular_spread(textureLoad(varianceAlbedo, t, 0));
    let last = vec2i(sp.size) - 1;
    var sd = vec3f(0.0);
    var ss = vec3f(0.0);
    var sm = vec4f(0.0);
    var wd = 0.0;
    var ws = 0.0;
    for (var y = -3; y <= 3; y++) {
        for (var x = -3; x <= 3; x++) {
            let q = t + vec2i(x, y);
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let gq = textureLoad(historySurface, q, 0);
            let zq = bitcast<f32>(gq.x);
            if (zq <= 0.0) { continue; }
            let w = geometry_weight(pc, nvc, n, inv_depth, view_pos(q, zq), oct_decode(gq.y), length(vec2f(f32(x), f32(y))));
            let dq = textureLoad(integratedDiffuse, q, 0).rgb;
            let sq = textureLoad(integratedSpecular, q, 0).rgb;
            let mq = textureLoad(momentsTex, q, 0);
            let w_d = w * exp(-abs(ldc - luminance(dq)) * inv_ld);
            let w_s = select(spread, 1.0, x == 0 && y == 0) * w * exp(-abs(lsc - luminance(sq)) * inv_ls);
            sd += w_d * dq;
            ss += w_s * sq;
            sm += vec4f(w_d * mq.xy, w_s * mq.zw);
            wd += w_d;
            ws += w_s;
        }
    }
    sd /= max(wd, 1e-6);
    ss /= max(ws, 1e-6);
    let md = sm.xy / max(wd, 1e-6);
    let ms = sm.zw / max(ws, 1e-6);
    // fewer frames: a larger variance, so the wavelet blurs more
    let boost = 4.0 / max(seen, 1.0);
    let vd = max(md.y - md.x * md.x, 0.0) * boost;
    let vs = max(ms.y - ms.x * ms.x, 0.0) * boost;
    // (no spread: the light as it was, the variance the moments give)
    ss = mix(s.rgb, ss, spread);
    outTexels[texel_index(t)] = Texel(g.xy, pack_light(vec4f(sd, sqrt(vd))), pack_light(vec4f(ss, sqrt(vs))));
}

// ---- a-trous ----

struct AtrousParams {
    step     : u32,
    feedback : u32,   // 1: also write next frame's history
    _pad0    : u32,
    _pad1    : u32,
}

@group(0) @binding(16) var<uniform> ap : AtrousParams;
@group(0) @binding(17) var<storage, read> atrousIn : array<Texel>;
@group(0) @binding(18) var<storage, read_write> atrousOut : array<Texel>;
@group(0) @binding(19) var historyDiffuse : texture_storage_2d<rgba16float, write>;
@group(0) @binding(20) var historySpecular : texture_storage_2d<rgba16float, write>;
@group(0) @binding(36) var atrousAlbedo : texture_2d<u32>;

fn atrous_store(t: vec2i, texel: Texel) {
    atrousOut[texel_index(t)] = texel;
    if (ap.feedback != 0u) {
        textureStore(historyDiffuse, t, unpack_light(texel.diffuse));
        textureStore(historySpecular, t, unpack_light(texel.specular));
    }
}

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.size)) { return; }
    let t = vec2i(gid.xy);
    let centre = atrousIn[texel_index(t)];
    let zc = bitcast<f32>(centre.surface.x);
    if (zc <= 0.0) {
        atrous_store(t, centre);
        return;
    }
    let d = unpack_light(centre.diffuse);
    let s = unpack_light(centre.specular);
    let last = vec2i(sp.size) - 1;
    // the variance prefiltered by a 3x3 Gaussian, for the luminance stops
    var vd = 0.0;
    var vs = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            // 1/4 at the centre, 1/8 beside, 1/16 at the corners
            let k = 0.25 / f32(1 << u32(abs(x) + abs(y)));
            let q = atrousIn[texel_index(clamp(t + vec2i(x, y), vec2i(0), last))];
            let sdq = unpack2x16float(q.diffuse.y).y;
            let ssq = unpack2x16float(q.specular.y).y;
            vd += k * sdq * sdq;
            vs += k * ssq * ssq;
        }
    }
    let inv_ld = 1.0 / (sp.phi_color * sqrt(max(vd, 0.0)) + 1e-6);
    let inv_ls = 1.0 / (sp.phi_color * sqrt(max(vs, 0.0)) + 1e-6);
    let nc = oct_decode(centre.surface.y);
    let pc = view_pos(t, zc);
    let nvc = view_normal(nc);
    let inv_depth = depth_stop(zc);
    let ldc = luminance(d.rgb);
    let lsc = luminance(s.rgb);
    let spread = specular_spread(textureLoad(atrousAlbedo, t, 0));
    // the B3 spline's 5 taps, or (radius 1) the 1-2-1 kernel's 3
    let radius = i32(sp.radius);
    var kernel = array<f32, 3>(0.375, 0.25, 0.0625);
    if (radius == 1) {
        kernel = array<f32, 3>(0.5, 0.25, 0.0);
    }
    let h0 = kernel[0] * kernel[0];
    var sum_d = d.rgb * h0;
    var sum_s = s.rgb * h0;
    var var_d = d.a * d.a * h0 * h0;
    var var_s = s.a * s.a * h0 * h0;
    var w_d = h0;
    var w_s = h0;
    let step = i32(ap.step);
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            if (x == 0 && y == 0) { continue; }
            let q = t + vec2i(x, y) * step;
            if (any(q < vec2i(0)) || any(q > last)) { continue; }
            let tq = atrousIn[texel_index(q)];
            let zq = bitcast<f32>(tq.surface.x);
            if (zq <= 0.0) { continue; }
            let h = kernel[abs(x)] * kernel[abs(y)];
            let wg = h * geometry_weight(pc, nvc, nc, inv_depth, view_pos(q, zq), oct_decode(tq.surface.y), length(vec2f(f32(x), f32(y))) * f32(step));
            let dq = unpack_light(tq.diffuse);
            let sq = unpack_light(tq.specular);
            let wdq = wg * exp(-abs(ldc - luminance(dq.rgb)) * inv_ld);
            let wsq = spread * wg * exp(-abs(lsc - luminance(sq.rgb)) * inv_ls);
            sum_d += wdq * dq.rgb;
            sum_s += wsq * sq.rgb;
            var_d += wdq * wdq * dq.a * dq.a;
            var_s += wsq * wsq * sq.a * sq.a;
            w_d += wdq;
            w_s += wsq;
        }
    }
    atrous_store(t, Texel(
        centre.surface,
        pack_light(vec4f(sum_d / w_d, sqrt(var_d) / w_d)),
        pack_light(vec4f(sum_s / w_s, sqrt(var_s) / w_s)),
    ));
}

// ---- composite ----

@group(0) @binding(21) var<storage, read> filtered : array<Texel>;
@group(0) @binding(22) var albedoTex : texture_2d<u32>;
@group(0) @binding(23) var compositeSignal : texture_2d<u32>;
@group(0) @binding(24) var compositeSurface : texture_2d<u32>;
@group(0) @binding(25) var temporalDiffuse : texture_2d<f32>;
@group(0) @binding(26) var temporalSpecular : texture_2d<f32>;
@group(0) @binding(27) var compositeHistory : texture_2d<u32>;
@group(0) @binding(28) var outColor : texture_storage_2d<rgba16float, write>;

fn heat(x: f32) -> vec3f {
    let t = clamp(x, 0.0, 1.0);
    return clamp(vec3f(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.size)) { return; }
    let t = vec2i(gid.xy);
    let a = textureLoad(albedoTex, t, 0);
    let s = textureLoad(compositeSignal, t, 0);
    let da = vec3f(unpack2x16float(a.x), unpack2x16float(a.y).x);
    let sa = vec3f(unpack2x16float(a.y).y, unpack2x16float(a.z));
    let through = bypass(s, a);
    let surface = textureLoad(compositeSurface, t, 0);
    let z = bitcast<f32>(surface.x);
    var out = vec3f(0.0);
    switch (sp.view) {
        case SVGF_VIEW_NOISY: {
            out = signal_diffuse(s) * da + signal_specular(s) * sa + through;
        }
        case SVGF_VIEW_DIFFUSE_ALBEDO: {
            out = da;
        }
        case SVGF_VIEW_SPECULAR_ALBEDO: {
            out = sa;
        }
        case SVGF_VIEW_NORMAL: {
            out = select(vec3f(0.0), oct_decode(surface.y) * 0.5 + 0.5, z > 0.0);
        }
        case SVGF_VIEW_DEPTH: {
            out = select(vec3f(0.0), heat(z / sp.heat_scale), z > 0.0);
        }
        case SVGF_VIEW_MOTION: {
            // pixels a frame: red right, green down, a pixel at heat_scale 1
            let m = select(vec2f(0.0), unpack2x16float(surface.z) * sp.size, z > 0.0);
            out = vec3f(abs(m) * sp.heat_scale, 0.0);
        }
        case SVGF_VIEW_DIFFUSE: {
            out = signal_diffuse(s);
        }
        case SVGF_VIEW_SPECULAR: {
            out = signal_specular(s);
        }
        case SVGF_VIEW_VARIANCE: {
            // the variance the wavelet starts from, relative to the light (diffuse red, specular green)
            let f = filtered[texel_index(t)];
            let d = unpack_light(f.diffuse);
            let sp2 = unpack_light(f.specular);
            out = vec3f(d.a / max(luminance(d.rgb), 1e-3), sp2.a / max(luminance(sp2.rgb), 1e-3), 0.0) * sp.heat_scale;
        }
        case SVGF_VIEW_HISTORY: {
            let seen = bitcast<f32>(textureLoad(compositeHistory, t, 0).z);
            out = select(vec3f(0.0), heat(seen / sp.max_history), z > 0.0);
        }
        case SVGF_VIEW_TEMPORAL: {
            out = textureLoad(temporalDiffuse, t, 0).rgb * da + textureLoad(temporalSpecular, t, 0).rgb * sa + through;
        }
        default: {
            if ((sp.flags & SVGF_FILTER) == 0u) {
                out = signal_diffuse(s) * da + signal_specular(s) * sa + through;
            } else {
                let f = filtered[texel_index(t)];
                out = unpack_light(f.diffuse).rgb * da + unpack_light(f.specular).rgb * sa + through;
            }
        }
    }
    textureStore(outColor, t, vec4f(out, 1.0));
}

// ---- resolve ----

@group(0) @binding(29) var currentTex : texture_2d<f32>;
@group(0) @binding(30) var resolveSurface : texture_2d<u32>;
@group(0) @binding(31) var taaHistory : texture_2d<f32>;
@group(0) @binding(32) var linearSampler : sampler;
@group(0) @binding(33) var resolveOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(34) var taaHistoryOut : texture_storage_2d<rgba16float, write>;

fn rgb_to_ycocg(c: vec3f) -> vec3f {
    return vec3f(dot(c, vec3f(0.25, 0.5, 0.25)), dot(c, vec3f(0.5, 0.0, -0.5)), dot(c, vec3f(-0.25, 0.5, -0.25)));
}

fn ycocg_to_rgb(c: vec3f) -> vec3f {
    return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// light -> x / (1 + luma(x)) -> YCoCg (single bright samples can't flicker), and back
fn to_working(c: vec3f) -> vec3f {
    let e = max(c, vec3f(0.0));
    return rgb_to_ycocg(e / (1.0 + luminance(e)));
}

fn from_working(c: vec3f) -> vec3f {
    let t = max(ycocg_to_rgb(c), vec3f(0.0));
    return t / max(1.0 - luminance(t), 1e-4);
}

// Clip toward the box centre (Playdead): keeps the history's hue where clamping would not.
fn clip_to_box(h: vec3f, lo: vec3f, hi: vec3f) -> vec3f {
    let centre = 0.5 * (hi + lo);
    let extents = 0.5 * (hi - lo) + 1e-6;
    let v = h - centre;
    let a = abs(v / extents);
    let m = max(a.x, max(a.y, a.z));
    return select(h, centre + v / m, m > 1.0);
}

// Catmull-Rom history lookup from 5 bilinear taps (the corner taps' weights are negligible).
fn sample_history(uv: vec2f) -> vec3f {
    let pos = uv * sp.size;
    let t1 = floor(pos - 0.5) + 0.5;
    let f = pos - t1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (t1 - 1.0) / sp.size;
    let t3 = (t1 + 2.0) / sp.size;
    let t12 = (t1 + w2 / w12) / sp.size;
    var c = textureSampleLevel(taaHistory, linearSampler, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
    c += textureSampleLevel(taaHistory, linearSampler, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
    c += textureSampleLevel(taaHistory, linearSampler, t12, 0.0).rgb * (w12.x * w12.y);
    c += textureSampleLevel(taaHistory, linearSampler, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
    c += textureSampleLevel(taaHistory, linearSampler, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    let w = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / w, vec3f(0.0));
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.size)) { return; }
    let t = vec2i(gid.xy);
    let last = vec2i(sp.size) - 1;
    var filtered = vec3f(0.0);
    var filter_weight = 0.0;
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    var lo = vec3f(1e9);
    var hi = vec3f(-1e9);
    // the nearest surface around the pixel: its motion (thin edges keep theirs)
    var closest_z = 1e30;
    var closest = vec4u(0u);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(t + vec2i(x, y), vec2i(0), last);
            let c = to_working(textureLoad(currentTex, q, 0).rgb);
            // pixel q's sample sat at its centre plus the jitter; exp(-2.29 d^2) fits the
            // Blackman-Harris reconstruction filter
            let d = vec2f(f32(x), f32(y)) + sp.jitter;
            let w = exp(-2.29 * dot(d, d));
            filtered += c * w;
            filter_weight += w;
            m1 += c;
            m2 += c * c;
            lo = min(lo, c);
            hi = max(hi, c);
            let g = textureLoad(resolveSurface, q, 0);
            let z = bitcast<f32>(g.x);
            if (z > 0.0 && z < closest_z) {
                closest_z = z;
                closest = g;
            }
        }
    }
    let current = filtered / filter_weight;
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
    let box_min = max(lo, mean - sp.taa_gamma * sigma);
    let box_max = min(hi, mean + sp.taa_gamma * sigma);

    // (no surface around: the sky, which the tracer holds constant, so no motion matters)
    let motion = select(vec2f(0.0), unpack2x16float(closest.z), closest_z < 1e30);
    let uv = (vec2f(t) + 0.5) / sp.size;
    let prev_uv = uv - motion;
    let centre = textureLoad(resolveSurface, t, 0);
    let prev_z = select(0.0, bitcast<f32>(centre.w), bitcast<f32>(centre.x) > 0.0);

    var result = current;
    if ((sp.flags & SVGF_TAA_HISTORY) != 0u && all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0))) {
        let history = clip_to_box(to_working(sample_history(prev_uv)), box_min, box_max);
        // resampling a moving history blurs it, so trust it less the faster things move
        let motion_px = length(motion * sp.size);
        var feedback = mix(sp.taa_feedback, 0.8 * sp.taa_feedback, saturate(motion_px / 8.0));
        // disocclusion: the nearest history texel saw a surface at another depth
        let history_z = textureLoad(taaHistory, clamp(vec2i(prev_uv * sp.size), vec2i(0), last), 0).a;
        if (prev_z > 0.0 && history_z > 0.0) {
            let mismatch = abs(history_z - prev_z) / max(prev_z, 1e-3);
            feedback *= 1.0 - smoothstep(0.02, 0.08, mismatch);
        }
        result = mix(current, history, feedback);
    }

    var rgb = from_working(result);
    if (any(rgb != rgb)) { rgb = vec3f(0.0); }   // NaN guard: never poison the history
    textureStore(resolveOut, t, vec4f(rgb, 1.0));
    textureStore(taaHistoryOut, t, vec4f(rgb, min(bitcast<f32>(centre.x), 65000.0)));
}
