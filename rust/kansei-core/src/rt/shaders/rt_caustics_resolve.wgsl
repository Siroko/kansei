// Caustics (rt::RtCausticsEffect), the map: `resolve` turns this frame's photons (the atomics,
// cleared for the next) into irradiance and blends them into the layers of the tiles that took
// photons (rgb lux, a the height the photons landed at over the tile's refY), by each tile's
// `blend`; `composite` adds the map's light to the image where the GBuffer shows an
// upward-facing surface under a tile, times its albedo / pi (not on glass or the sky, and with
// `receivers` 0 only on the surfaces RtShadowsEffect lights, whose direct light passes through
// glass), each light's `weighted` by RtShadowsEffect's visibility of it there. Prefixed with
// rt_caustics_lookup.wgsl.

const CAUSTICS_WEIGHT : f32 = 4096.0;

@group(0) @binding(0) var<uniform> caustics : KanseiRtCaustics;
@group(0) @binding(1) var<storage, read_write> acc : array<atomic<i32>>;
@group(0) @binding(2) var mapIn : texture_2d_array<f32>;
@group(0) @binding(3) var mapOut : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid : vec3u) {
    let layer = gid.z;
    let k = layer / max(caustics.lights, 1u);
    if (k >= min(caustics.count, KANSEI_RT_CAUSTIC_TILES)) { return; }
    let t = caustics.tiles[k];
    if (t.blend <= 0.0 || any(gid.xy >= t.dims)) { return; }
    let at = ((layer * caustics.texels + gid.y) * caustics.texels + gid.x) * 5u;
    let e = vec3f(f32(atomicExchange(&acc[at], 0)), f32(atomicExchange(&acc[at + 1u], 0)), f32(atomicExchange(&acc[at + 2u], 0))) / caustics.scale;
    let w = f32(atomicExchange(&acc[at + 3u], 0)) / CAUSTICS_WEIGHT;
    let wy = f32(atomicExchange(&acc[at + 4u], 0)) / CAUSTICS_WEIGHT;
    // a fresh tile (blend 1) starts from nothing: what its layer held was another placement's
    let previous = select(textureLoad(mapIn, gid.xy, layer, 0), vec4f(0.0), t.blend >= 1.0);
    var height = previous.a;
    if (w > 1e-3) {
        height = mix(previous.a, wy / w, select(0.5, 1.0, t.blend >= 1.0 || previous.a == 0.0));
    }
    textureStore(mapOut, gid.xy, layer, vec4f(mix(previous.rgb, e, t.blend), height));
}

struct CompositeParams {
    invProj   : mat4x4f,
    invView   : mat4x4f,
    size      : vec2f,
    // 0: lit; 1: the light the caustics add and take on a white surface, over a grey of 40 cd/m²
    view      : u32,
    // 0: only the surfaces RtShadowsEffect lights; 1: every surface
    receivers : u32,
    // the pixels the tiles may cover (min x, y, max x, y): the others pass through
    rect      : vec4f,
    // RtShadowsEffect's trace resolution and pixels a texel, and whether its visibility weighs
    // each light's change (visA, visB, visGuide)
    traceSize : vec2u,
    downscale : u32,
    weighted  : u32,
}

@group(0) @binding(0) var<uniform> cc : CompositeParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var emissiveTex : texture_2d<f32>;
@group(0) @binding(5) var inputTex : texture_2d<f32>;
@group(0) @binding(6) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var causticsMap : texture_2d_array<f32>;
@group(0) @binding(8) var causticsSampler : sampler;
@group(0) @binding(9) var<uniform> lookup : KanseiRtCaustics;
@group(0) @binding(10) var visA : texture_2d<f32>;
@group(0) @binding(11) var visB : texture_2d<f32>;
@group(0) @binding(12) var visGuide : texture_2d<f32>;

fn decodeNormal(p: vec2f) -> vec3f {
    var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    n = vec3f(n.xy + select(vec2f(t), vec2f(-t), n.xy >= vec2f(0.0)), n.z);
    return normalize(n);
}

// RtShadowsEffect's visibility of its lights at a pixel (view depth z, normal n), upsampled from
// its trace resolution as its composite does: the four texels round it, each by its bilinear
// weight and how alike its surface is.
fn visibility(px: vec2u, z: f32, n: vec3f) -> array<f32, 8> {
    let tp = (vec2f(px) + 0.5) / f32(cc.downscale) - 0.5;
    let t0 = vec2i(floor(tp));
    let f = tp - floor(tp);
    let last = vec2i(cc.traceSize) - 1;
    var a = vec4f(0.0);
    var b = vec4f(0.0);
    var wsum = 0.0;
    var fa = vec4f(1.0);
    var fb = vec4f(1.0);
    var best = -1.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1u);
        let q = clamp(t0 + o, vec2i(0), last);
        let g = textureLoad(visGuide, q, 0);
        if (g.z <= 0.0) { continue; }
        let similar = exp(-abs(g.z - z) / (0.02 * z + 1e-3)) * pow(max(dot(decodeNormal(g.xy), n), 0.0), 16.0);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * similar;
        let qa = textureLoad(visA, q, 0);
        let qb = textureLoad(visB, q, 0);
        a += w * qa;
        b += w * qb;
        wsum += w;
        if (similar > best) { best = similar; fa = qa; fb = qb; }
    }
    if (wsum > 1e-4) {
        a /= wsum;
        b /= wsum;
    } else {
        a = fa;
        b = fb;
    }
    return array<f32, 8>(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
}

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= cc.size)) { return; }
    let px = gid.xy;
    let color = textureLoad(inputTex, px, 0);
    if ((any(vec2f(px) < cc.rect.xy) || any(vec2f(px) > cc.rect.zw)) && cc.view == 0u) {
        textureStore(outputTex, px, color);
        return;
    }
    let depth = textureLoad(depthTex, px, 0);
    let albedo = textureLoad(albedoTex, px, 0);
    let ea = textureLoad(emissiveTex, px, 0).a;
    // the sky, glass (its albedo's alpha 0.05-0.45); surfaces RtShadowsEffect lights mark the
    // emissive alpha 0.05-0.45 (rt_shadows_gbuffer.wgsl), a fluid's screen-space surface above 0.5
    let rtLit = ea > 0.03 && ea < 0.47;
    let skip = depth >= 1.0 || (albedo.a > 0.025 && albedo.a < 0.475) || select(!rtLit, ea > 0.5, cc.receivers == 1u);
    var e = vec3f(0.0);
    var white = vec3f(0.0);
    if (!skip) {
        let uv = (vec2f(px) + 0.5) / cc.size;
        let view = cc.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        let world = (cc.invView * vec4f(view.xyz / view.w, 1.0)).xyz;
        let n = normalize(textureLoad(normalTex, px, 0).xyz * 2.0 - 1.0);
        // each light's change (a layer of each tile covering the point), where that light reaches:
        // RtShadowsEffect's visibility of it, read the first time a light changes anything
        var vis = array<f32, 8>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        var read = cc.weighted == 0u;
        if (lookup.enabled != 0u && n.y >= 0.3) {
            for (var k = 0u; k < min(lookup.count, KANSEI_RT_CAUSTIC_TILES); k++) {
                let t = lookup.tiles[k];
                if (t.live == 0u) { continue; }
                let uv = (world.xz - t.min) / t.size;
                if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { continue; }
                let d = vec2f(t.dims);
                let q = clamp(uv * d, vec2f(0.5), d - 0.5) / f32(lookup.texels);
                for (var l = 0u; l < lookup.lights; l++) {
                    let s = textureSampleLevel(causticsMap, causticsSampler, q, k * lookup.lights + l, 0.0);
                    if (abs(s.a + t.refY - world.y) > t.tolerance || all(s.rgb == vec3f(0.0))) { continue; }
                    if (!read) {
                        vis = visibility(px, -view.z / view.w, n);
                        read = true;
                    }
                    let slot = kansei_rt_caustics_shadow_slot(lookup, l);
                    white += s.rgb * select(1.0, saturate(vis[min(slot, 7u)]), slot < 8u);
                }
            }
        }
        white *= lookup.intensity;
        white /= 3.14159265;
        e = white * albedo.rgb;
    }
    if (cc.view == 1u) {
        textureStore(outputTex, px, vec4f(max(vec3f(40.0) + white, vec3f(0.0)), color.a));
        return;
    }
    textureStore(outputTex, px, vec4f(max(color.rgb + e, vec3f(0.0)), color.a));
}
