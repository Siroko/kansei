// Ray-traced diffuse GI (rt::RtDiffuseGiEffect), composite: at full resolution, the signal (the
// denoiser's output, the raw trace, or the running mean while accumulating) upsampled from the
// trace resolution (the four nearest trace texels, weighted by distance, depth and normal), then
// the lit colour plus albedo times it; with the sky's lighting bound, the material's own sky
// ambient (albedo / pi times the sky's irradiance round its normal) is taken out `ambient` times,
// as voxel GI does.
// Prefixed with SKY_LIGHTING_WGSL and rt_gi_common.wgsl.

@group(0) @binding(20) var<uniform> gp : RtGiParams;
@group(0) @binding(21) var colorTex : texture_2d<f32>;
@group(0) @binding(22) var depthTex : texture_depth_2d;
@group(0) @binding(23) var normalTex : texture_2d<f32>;
@group(0) @binding(24) var albedoTex : texture_2d<f32>;
@group(0) @binding(25) var signalTex : texture_2d<f32>;
@group(0) @binding(26) var guideTex : texture_2d<u32>;
@group(0) @binding(27) var<storage, read> accum : array<vec4f>;
@group(0) @binding(28) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(29) var<uniform> sky : SkyLighting;
@group(0) @binding(30) var momentsTex : texture_2d<f32>;
@group(0) @binding(31) var integratedTex : texture_2d<f32>;

fn signalAt(t: vec2u) -> vec4f {
    if ((gp.flags & RT_GI_ACCUMULATE) != 0u) {
        let a = accum[t.y * u32(gp.traceSize.x) + t.x];
        return vec4f(a.rgb / max(a.a, 1.0), 1.0);
    }
    return textureLoad(signalTex, t, 0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.fullSize)) { return; }
    let px = gid.xy;
    let color = textureLoad(colorTex, px, 0);
    let depth = textureLoad(depthTex, px, 0);
    let raw = textureLoad(normalTex, px, 0);
    if (depth >= 1.0 || dot(raw.xyz, raw.xyz) < 1e-4) {
        textureStore(outTex, px, select(color, vec4f(0.0, 0.0, 0.0, 1.0), gp.view != 0u));
        return;
    }
    let n = normalize(raw.xyz * 2.0 - 1.0);
    let uv = giUv(px);
    let z = -giViewPos(uv, depth).z;
    var s = vec4f(0.0);
    var nearest = vec2u(0u);
    if (gp.downscale <= 1u) {
        s = signalAt(px);
        nearest = px;
    } else {
        // trace texel t is pixel t * downscale
        let tp = vec2f(px) / f32(gp.downscale);
        let t0 = vec2i(floor(tp));
        let f = tp - floor(tp);
        let last = vec2i(gp.traceSize) - 1;
        var wsum = 0.0;
        var best = -1.0;
        var fallback = vec4f(0.0);
        for (var k = 0; k < 4; k++) {
            let o = vec2i(k & 1, k >> 1u);
            let q = vec2u(clamp(t0 + o, vec2i(0), last));
            let g = giDecodeGuide(textureLoad(guideTex, q, 0));
            if (g.x <= 0.0) { continue; }
            let similar = exp(-abs(g.x - z) / max(0.02 * z, 1e-3)) * pow(max(dot(g.yzw, n), 0.0), 8.0);
            let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
            let w = (bilinear + 1e-3) * similar;
            let v = signalAt(q);
            s += w * v;
            wsum += w;
            if (similar > best) {
                best = similar;
                fallback = v;
                nearest = q;
            }
        }
        s = select(fallback, s / max(wsum, 1e-6), wsum > 1e-4);
    }
    let albedo = textureLoad(albedoTex, px, 0).rgb;
    let indirect = albedo * s.rgb * gp.intensity;
    var base = color.rgb;
    if ((gp.flags & RT_GI_HAS_SKY) != 0u) {
        base = max(base - albedo / RT_GI_PI * skyIrradiance(sky, n) * gp.ambient, vec3f(0.0));
    }
    let lit = base + indirect;
    var shown = vec4f(lit, color.a);
    switch (gp.view) {
        case 1u: { shown = vec4f(indirect, 1.0); }
        case 2u: { shown = vec4f(s.rgb * gp.intensity, 1.0); }
        case 3u: {
            // the variance the wavelet started from, relative to the signal's luminance squared
            let i = textureLoad(integratedTex, nearest, 0);
            let l = max(giLuminance(i.rgb), 1e-4);
            shown = vec4f(giHeat(sqrt(i.a) / l * 0.5) * gp.heatScale, 1.0);
        }
        case 4u: { shown = vec4f(giHeat(textureLoad(momentsTex, nearest, 0).z / gp.maxHistory) * gp.heatScale, 1.0); }
        case 5u: { shown = vec4f(giHeat((s.a - 1.0) / 400.0) * gp.heatScale, 1.0); }
        default: {}
    }
    textureStore(outTex, px, shown);
}
