// Ray-traced glass (rt::RtReflectionsEffect with `set_glass`), at full resolution: on the pixels
// whose material wrote glass (GBUFFER_OUT_WGSL's kansei_gbuffer_out_glass: its tint in the albedo,
// its roughness in the albedo's alpha, 1 / its index of refraction in the normal's), the
// reflection (exact dielectric Fresnel at the surface) plus the light refracted through the glass:
// a ray through the grid of triangles, bent at every glass surface it crosses (RtSurface::glass;
// Snell's law, reflected where the light can't leave: total internal reflection), dimmed inside by
// the tint (Beer-Lambert, the tint the light left after a metre) and by each surface's Fresnel,
// until it leaves and hits the room (rt_reflect_hit.wgsl: the lit image where the camera sees the
// point, else the lights and the voxels), or the sky. The glass keeps the index, tint and
// roughness of the pixel it entered at. One path a sample: the light reflected inside at a surface
// crossed is lost. Frosted glass jitters each surface's normal over its lobe, rp.glassSamples
// paths a pixel a frame, blended with the pixel's history reprojected by the surface.
// Other pixels pass through. Prefixed as the trace (rt_reflect_common.wgsl, RT_GRID_WGSL, the
// voxel source, rt_reflect_hit.wgsl).

@group(0) @binding(0) var<uniform> rp : RtReflectParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> sky : SkyLighting;
// rtStats[5]: glass pixels; [6]: cells and triangles their rays visited
@group(1) @binding(5) var<storage, read_write> rtStats : array<atomic<u32>, 8>;
// frosted glass's history: last frame's (read through the sampler), and this frame's
@group(0) @binding(12) var glassHistory : texture_2d<f32>;
@group(0) @binding(13) var glassHistoryOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var glassSampler : sampler;
// the image the hits are looked up in is this pass's input: its glass isn't drawn yet
const RT_SCREEN_HAS_GLASS : bool = false;

// The reflectance of a smooth dielectric surface for unpolarized light, from a medium of index
// etaI into one of etaT at the incidence cosine cosI (1 past the critical angle).
fn rtFresnelDielectric(cosI: f32, etaI: f32, etaT: f32) -> f32 {
    let sinT = etaI / etaT * sqrt(max(1.0 - cosI * cosI, 0.0));
    if (sinT >= 1.0) {
        return 1.0;
    }
    let cosT = sqrt(max(1.0 - sinT * sinT, 0.0));
    let rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    let rpar = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return 0.5 * (rs * rs + rpar * rpar);
}

// The glass a pixel shows, as kansei_gbuffer_out_glass wrote it.
struct Glass {
    ior    : f32,
    // Beer-Lambert's absorption a metre (-log of the tint)
    absorb : vec3f,
    // the tan of its surfaces' lobe (rtLobeTan of its roughness; 0: clear)
    lobe   : f32,
}

struct GlassPath {
    radiance : vec3f,
    cost     : u32,
}

// The normal a ray meets at a surface of normal n: n itself, or on frosted glass a microfacet's,
// jittered over the lobe (the same cone the reflections' rough surfaces sample), on n's side.
fn glassFacet(g: Glass, n: vec3f, px: vec2u, seed: u32) -> vec3f {
    if (g.lobe <= 1e-3) {
        return n;
    }
    return rtGlossy(n, g.lobe, px, seed);
}

// One path into the glass at p (view direction v, surface normal n facing the eye): the
// reflection off it plus the light through it.
fn glassPath(g: Glass, p: vec3f, v: vec3f, n: vec3f, px: vec2u, k: u32, samples: u32, flags: u32) -> GlassPath {
    var out : GlassPath;
    let eye = rp.invView[3].xyz;
    let ior = g.ior;
    let lift = 0.05 * kansei_rt_grid.cell + 1e-3 * distance(eye, p);
    let seed = (rp.frame * samples + k) * 32u;
    var m = glassFacet(g, n, px, seed);
    if (dot(m, v) > -1e-3) {
        m = n;
    }
    let fr = rtFresnelDielectric(clamp(-dot(v, m), 0.0, 1.0), 1.0, ior);
    // the reflection off the outside (above the surface)
    var r = reflect(v, m);
    let rise = dot(r, n);
    if (rise < 0.02) {
        r = normalize(r + n * (0.02 - rise));
    }
    let mirror = rtSceneRay(p + n * lift, r, n, 0.0, flags);
    out.cost = mirror.cost.x + mirror.cost.y;
    // the light through it, meeting the glass's surfaces (and only those of other glass in its way)
    var throughput = vec3f(1.0 - fr);
    var d = refract(v, m, 1.0 / ior);
    if (dot(d, d) < 0.5) {
        d = refract(v, n, 1.0 / ior);
    }
    var o = p - n * lift;
    var inside = true;
    var through = vec3f(0.0);
    let step = 0.05 * kansei_rt_grid.cell;
    for (var i = 0u; i < rp.glassInterfaces; i++) {
        if ((rp.flags & RT_REFLECT_GRID) == 0u || !kansei_rt_contains(o)) {
            through = throughput * rtFarField(o, d, d, 0.0, 0.0);
            break;
        }
        let hit = kansei_rt_trace(o, d, 0.0, rp.maxDistance, flags | KANSEI_RT_GLASS_HITS);
        out.cost += hit.cells + hit.tests;
        if (!hit.found) {
            through = throughput * rtFarField(o, d, d, 0.0, kansei_rt_exit(o, d));
            break;
        }
        let h = o + d * hit.t;
        if (inside) {
            throughput *= exp(-g.absorb * hit.t);
        }
        if (!rtIsGlass(hit.triangle)) {
            through = throughput * rtHitRadiance(h, hit.normal, hit.triangle, hit.bary);
            break;
        }
        // a glass surface, bent by its interpolated normal (RtSurface::glass has smooth normals),
        // facing where the ray came from as hit.normal does; frosted, by a microfacet round it
        let ns = kansei_rt_shading_normal(hit.triangle, hit.bary, hit.normal);
        var mi = glassFacet(g, ns, px, seed + 1u + i);
        if (dot(mi, d) > -1e-3) {
            mi = ns;
        }
        let c = clamp(-dot(d, mi), 0.0, 1.0);
        let etaI = select(1.0, ior, inside);
        let etaT = select(ior, 1.0, inside);
        let f = rtFresnelDielectric(c, etaI, etaT);
        if (f >= 1.0) {
            d = reflect(d, mi);
            o = h + hit.normal * step;
            continue;
        }
        d = refract(d, mi, etaI / etaT);
        throughput *= 1.0 - f;
        o = h - hit.normal * step;
        inside = !inside;
    }
    out.radiance = fr * mirror.radiance + through;
    return out;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= rp.fullSize)) { return; }
    let px = gid.xy;
    let color = textureLoad(screenTex, px, 0);
    let depth = textureLoad(depthTex, px, 0);
    let albedo = textureLoad(albedoTex, px, 0);
    if (depth >= 1.0 || !rtIsGlassAlbedo(albedo.a)) {
        textureStore(outTex, px, color);
        textureStore(glassHistoryOut, px, vec4f(0.0));
        return;
    }
    let raw = textureLoad(normalTex, px, 0);
    var g : Glass;
    g.ior = 1.0 / clamp(raw.w, 0.05, 1.0);
    g.absorb = -log(clamp(albedo.rgb, vec3f(1e-4), vec3f(1.0)));
    let roughness = clamp((0.45 - albedo.a) / 0.4, 0.0, 1.0);
    // (8-bit: a clear glass's alpha reads a hair off 0.45)
    g.lobe = select(rtLobeTan(roughness), 0.0, roughness < 0.01);
    let eye = rp.invView[3].xyz;
    let p = rtWorldPos((vec2f(px) + 0.5) / rp.fullSize, depth);
    let v = normalize(p - eye);
    var n = normalize(raw.xyz * 2.0 - 1.0);
    if (dot(n, v) > 0.0) {
        n = -n;
    }
    var flags = KANSEI_RT_SOLID;
    if ((rp.flags & RT_REFLECT_ALPHA) != 0u) {
        flags = 0u;
    }
    let frosted = g.lobe > 0.0;
    let samples = select(1u, max(rp.glassSamples, 1u), frosted);
    var sum = vec3f(0.0);
    var cost = 0u;
    for (var k = 0u; k < samples; k++) {
        let path = glassPath(g, p, v, n, px, k, samples, flags);
        sum += path.radiance;
        cost += path.cost;
    }
    var result = min(sum / f32(samples), vec3f(60000.0));
    // frosted: blended with the history where the surface was last frame
    if (frosted && (rp.flags & RT_REFLECT_HISTORY) != 0u) {
        let clip = rp.prevViewProj * vec4f(p, 1.0);
        let uv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
        if (clip.w > 0.0 && all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
            let history = textureSampleLevel(glassHistory, glassSampler, uv, 0.0);
            if (history.a > 0.99) {
                result = mix(history.rgb, result, rp.blend);
            }
        }
    }
    textureStore(glassHistoryOut, px, vec4f(result, 1.0));
    if ((rp.flags & RT_REFLECT_STATS) != 0u) {
        atomicAdd(&rtStats[5], 1u);
        atomicAdd(&rtStats[6], cost);
    }
    textureStore(outTex, px, vec4f(result, color.a));
}
