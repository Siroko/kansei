// Ray-traced reflections and glass (rt::RtReflectionsEffect), how a ray's hit is lit, shared by
// the trace and the glass pass: with RT_REFLECT_SCREEN, the lit image where the camera sees the
// same point (sharp, with the direct light and the GI); elsewhere, with RT_REFLECT_DIRECT, the
// lights (the spot lights and the directional and point ones of `update_lights`, shadowed by rays
// through the grid) and the voxels' irradiance, else the voxels (srcHitRadiance); past the grid or on a
// miss, a voxel cone then the sky. Glass (RtSurface::glass) shows where the camera sees it (the
// glass pass's image) and is stepped through elsewhere. Included after the trace's bindings,
// RT_GRID_WGSL and the voxel source.

// the image the hits are looked up in (the lit colour before the reflections)
@group(0) @binding(10) var screenTex : texture_2d<f32>;
// the renderer's spot lights (RT_REFLECT_DIRECT: the hits the camera doesn't see are lit by them)
@group(0) @binding(11) var<storage, read> rtSpots : KanseiSpotLights;

// The directional and point lights (RtReflectionsEffect::update_lights): `numDir` directional ones
// (a: the direction it travels, b: its colour), then `numPoint` point ones (a: position and
// radius, b: colour, falling off as (1 - d / radius)^2).
struct RtReflectLight {
    a : vec4f,
    b : vec4f,
}
struct RtReflectLights {
    numDir   : u32,
    numPoint : u32,
    _pad0    : u32,
    _pad1    : u32,
    lights   : array<RtReflectLight>,
}
@group(0) @binding(15) var<storage, read> rtLights : RtReflectLights;

// Whether nothing in the grid lies within tMax along unit d from o (glass lets the light through).
fn rtVisible(o: vec3f, d: vec3f, tMax: f32) -> f32 {
    if (!kansei_rt_contains(o)) {
        return 1.0;
    }
    let hit = kansei_rt_trace(o, d, 0.0, tMax, KANSEI_RT_ANY_HIT | KANSEI_RT_SOLID);
    return select(1.0, 0.0, hit.found);
}

// Whether nothing in the grid lies between o and `to`.
fn rtLit(o: vec3f, to: vec3f) -> f32 {
    let dist = distance(o, to);
    return rtVisible(o, (to - o) / dist, dist - 1e-3);
}

// The light leaving a hit the camera doesn't see (a Lambertian surface of `albedo`, the
// includer's kansei_rt_hit_albedo: the triangle's, or its texture's where it has one; plus the
// includer's kansei_rt_hit_caustics):
// the lights' direct light, shadowed by rays through the grid, plus the irradiance the voxels hold
// round it (srcIrradiance, voxel GI's own hemisphere of cones).
fn rtLitHit(p: vec3f, ng: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
    let o = p + ng * (0.05 * kansei_rt_grid.cell);
    var e = vec3f(0.0);
    for (var i = 0u; i < rtLights.numDir; i++) {
        let light = rtLights.lights[i];
        let l = -normalize(light.a.xyz);
        let ndl = dot(n, l);
        if (ndl > 0.0) {
            e += light.b.rgb * ndl * rtVisible(o, l, rp.maxDistance);
        }
    }
    for (var i = 0u; i < rtLights.numPoint; i++) {
        let light = rtLights.lights[rtLights.numDir + i];
        let d = light.a.xyz - p;
        let dist = length(d);
        if (dist > light.a.w || dist < 1e-4) {
            continue;
        }
        let ndl = dot(n, d / dist);
        if (ndl > 0.0) {
            let falloff = (1.0 - dist / light.a.w) * (1.0 - dist / light.a.w);
            e += light.b.rgb * ndl * falloff * rtLit(o, light.a.xyz);
        }
    }
    for (var i = 0u; i < rtSpots.count; i++) {
        let light = rtSpots.lights[i];
        let s = kansei_spot_sample(light, p);
        let ndl = dot(n, s.toLight);
        if (ndl <= 0.0 || max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) {
            continue;
        }
        e += s.illuminance * ndl * rtLit(o, light.position);
    }
    if ((rp.flags & RT_REFLECT_NO_INDIRECT) == 0u) {
        e += srcIrradiance(o, n) * rp.indirectScale;
    }
    // the light glass and water focus there (and take away), from the includer: none, or a caustics
    // map (RtCausticsEffect)
    e += kansei_rt_hit_caustics(p, n);
    return albedo / 3.14159265 * max(e, vec3f(0.0));
}

fn rtHash(p: vec2u, frame: u32) -> vec2f {
    var h = p.x * 1973u + p.y * 9277u + frame * 26699u;
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12u)) * 0x297a2d39u;
    let a = h ^ (h >> 15u);
    let b = a * 0x9e3779b9u;
    return vec2f(f32(a & 0xffffu), f32(b >> 16u)) / 65536.0;
}

// A direction round `r`, uniform over the cone of half-angle atan(tanHalf).
fn rtGlossy(r: vec3f, tanHalf: f32, p: vec2u, frame: u32) -> vec3f {
    let xi = rtHash(p, frame);
    let phi = 6.2831853 * xi.x;
    let cosMax = 1.0 / sqrt(1.0 + tanHalf * tanHalf);
    let cosT = 1.0 - xi.y * (1.0 - cosMax);
    let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
    // an orthonormal basis round r (Duff et al. 2017)
    let s = select(-1.0, 1.0, r.z >= 0.0);
    let a = -1.0 / (s + r.z);
    let b = r.x * r.y * a;
    let t = vec3f(1.0 + s * r.x * r.x * a, s * b, -s * r.x);
    let bt = vec3f(b, s + r.y * r.y * a, -r.y);
    return normalize(r * cosT + (t * cos(phi) + bt * sin(phi)) * sinT);
}

fn rtIsGlass(id: u32) -> bool {
    return (bitcast<u32>(kansei_rt_triangles[id * 4u].w) & KANSEI_RT_GLASS) != 0u;
}

// How far off p's plane, along its normal, a point the camera sees may lie and still be p's
// surface: a proxy's few centimetres, not the whole lateral tolerance (a grid cell and a half),
// which took a parallel surface a bay away for it (a shelf's arms, 13 cm apart, each showing the
// one above it in its reflection).
fn rtScreenPlaneTolerance(p: vec3f, tolerance: f32) -> f32 {
    return min(tolerance, max(0.05, 0.004 * distance(rp.invView[3].xyz, p)));
}

// The lit image at pixel q where it shows the surface at p (within `tolerance`, facing `n`): rgb
// and 1, or 0.
fn rtScreenAt(q: vec2u, p: vec3f, n: vec3f, tolerance: f32) -> vec4f {
    let depth = textureLoad(depthTex, q, 0);
    if (depth >= 1.0) {
        return vec4f(0.0);
    }
    // (the glass pass's own image has no glass drawn yet: RT_SCREEN_HAS_GLASS, which the pass sets)
    let glass = rtIsGlassAlbedo(textureLoad(albedoTex, q, 0).a);
    if (!RT_SCREEN_HAS_GLASS && glass) {
        return vec4f(0.0);
    }
    let seen = rtWorldPos((vec2f(q) + 0.5) / rp.fullSize, depth);
    if (distance(seen, p) > tolerance || abs(dot(seen - p, n)) > rtScreenPlaneTolerance(p, tolerance)) {
        return vec4f(0.0);
    }
    // the same surface, not another within the tolerance (a foot's edge over the floor beside
    // it, whose dark pixels would smear along the floor): its normal too
    let raw = textureLoad(normalTex, q, 0);
    let seenN = raw.xyz * 2.0 - 1.0;
    if (dot(seenN, seenN) < 1e-4 || dot(normalize(seenN), n) < 0.8) {
        return vec4f(0.0);
    }
    // nor a metal (its F0, stored as 1 - F0 in the normal's alpha, a quarter or more): the image
    // holds it before its reflection is added, nearly black, and a shelf of polished sheets showed
    // each arm black in the next one's reflection
    if (!glass && raw.w < 0.75) {
        return vec4f(0.0);
    }
    return vec4f(textureLoad(screenTex, q, 0).rgb, 1.0);
}

// The lit image at p, when the camera sees p from the side `n` faces (rgb, and 1; else 0). Where
// something thin in front hides p (a leg over the floor), the pixels beside it on the same surface
// stand in: a hidden strip lit by the traced fallback instead (another estimate of the indirect
// light) shows in a mirror as a streak behind each leg.
fn rtScreenRadiance(p: vec3f, n: vec3f) -> vec4f {
    return rtScreenRadianceAt(p, n, true);
}

// The same, beside thin occluders only with `search`.
fn rtScreenRadianceAt(p: vec3f, n: vec3f, search: bool) -> vec4f {
    let eye = rp.invView[3].xyz;
    if (dot(n, eye - p) <= 0.0) {
        return vec4f(0.0);
    }
    let clip = rp.viewProj * vec4f(p, 1.0);
    if (clip.w <= 0.0) {
        return vec4f(0.0);
    }
    let uv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) {
        return vec4f(0.0);
    }
    let px = vec2u(uv * rp.fullSize);
    let tolerance = max(1.5 * kansei_rt_grid.cell, 0.01 * distance(eye, p));
    let direct = rtScreenAt(px, p, n, tolerance);
    if (direct.a > 0.5 || !search) {
        return direct;
    }
    // hidden by something nearer? then the same surface a few pixels either side
    let depth = textureLoad(depthTex, px, 0);
    // (by a few centimetres: a toe just in front of the floor counts)
    if (depth >= 1.0 || distance(eye, rtWorldPos((vec2f(px) + 0.5) / rp.fullSize, depth)) > distance(eye, p) - 0.03) {
        return vec4f(0.0);
    }
    let last = vec2i(rp.fullSize) - 1;
    for (var step = 1; step <= 8; step++) {
        let d = step * step * 2;
        for (var side = 0; side < 4; side++) {
            let o = select(vec2i(select(-d, d, side == 1), 0), vec2i(0, select(-d, d, side == 3)), side >= 2);
            let q = vec2u(clamp(vec2i(px) + o, vec2i(0), last));
            // its own point need only lie on p's plane, near p
            let dq = textureLoad(depthTex, q, 0);
            if (dq >= 1.0) {
                continue;
            }
            let seen = rtWorldPos((vec2f(q) + 0.5) / rp.fullSize, dq);
            if (abs(dot(seen - p, n)) > rtScreenPlaneTolerance(p, tolerance) || distance(seen, p) > 1.0) {
                continue;
            }
            let s = rtScreenAt(q, seen, n, tolerance);
            if (s.a > 0.5) {
                return s;
            }
        }
    }
    return vec4f(0.0);
}

// Light the hits the camera doesn't see from the voxels' surface cache alone (srcHitRadiance), not
// the lights: for a ray whose light counts for little (a weak Fresnel reflection, set by the glass
// pass around its reflection ray).
var<private> rtCheapHits : bool = false;

// A hit at p on `triangle` (geometric normal ng facing the ray's origin, barycentrics bary).
fn rtHitRadiance(p: vec3f, ng: vec3f, triangle: u32, bary: vec2f) -> vec3f {
    if ((rp.flags & RT_REFLECT_SCREEN) != 0u) {
        let s = rtScreenRadianceAt(p, ng, false);
        if (s.a > 0.5) {
            return s.rgb;
        }
        // a surface that lights itself, seen only through the refractive pass (water): the
        // includer's background image (none in TS); then the screen beside thin occluders
        let b = kansei_rt_hit_background(p, ng, triangle, bary);
        if (b.a > 0.5) {
            return b.rgb;
        }
        let beside = rtScreenRadianceAt(p, ng, true);
        if (beside.a > 0.5) {
            return beside.rgb;
        }
    }
    if ((rp.flags & RT_REFLECT_DIRECT) != 0u && !rtCheapHits) {
        return rtLitHit(p, ng, kansei_rt_shading_normal(triangle, bary, ng), kansei_rt_hit_albedo(triangle, bary));
    }
    return srcHitRadiance(p, ng);
}

// Past the grid (or outside it): the voxels from where the ray left it, then the sky.
fn rtFarField(origin: vec3f, d: vec3f, n: vec3f, tanHalf: f32, exit: f32) -> vec3f {
    let size = srcVoxelSize(origin);
    let c = srcCone(origin, d, n, max(rp.coneTan, tanHalf), max(exit, size), rp.maxDistance, rp.coneSteps);
    return c.rgb + c.a * rp.skyScale * skyRadiance(sky, d);
}

struct RtRay {
    radiance : vec3f,
    found    : bool,
    cost     : vec2u,   // cells, triangles
}

// The light a ray from `origin` (off surface normal `n`) brings back. Glass the camera doesn't see
// is stepped through (straight: what it refracts is the glass pass's job).
fn rtSceneRay(origin: vec3f, d: vec3f, n: vec3f, tanHalf: f32, traceFlags: u32) -> RtRay {
    var out : RtRay;
    if ((rp.flags & RT_REFLECT_GRID) == 0u || !kansei_rt_contains(origin)) {
        out.radiance = rtFarField(origin, d, n, tanHalf, 0.0);
        return out;
    }
    let glass = (rp.flags & RT_REFLECT_GLASS) != 0u;
    let flags = traceFlags | select(0u, KANSEI_RT_GLASS_HITS, glass);
    var o = origin;
    for (var k = 0u; k < 4u; k++) {
        let hit = kansei_rt_trace(o, d, 0.0, rp.maxDistance, flags);
        out.cost += vec2u(hit.cells, hit.tests);
        if (!hit.found) {
            out.radiance = rtFarField(o, d, n, tanHalf, kansei_rt_exit(o, d));
            return out;
        }
        let p = o + d * hit.t;
        if (glass && rtIsGlass(hit.triangle)) {
            // (not the coarse copy of the surface the ray left, RtSurface::with_approximate)
            let copy = (bitcast<u32>(kansei_rt_triangles[hit.triangle * 4u].w) & KANSEI_RT_APPROXIMATE) != 0u && distance(p, origin) < 0.3;
            if (!copy) {
                let s = rtScreenRadiance(p, hit.normal);
                if (s.a > 0.5) {
                    out.radiance = s.rgb;
                    out.found = true;
                    return out;
                }
            }
            o = p + d * (0.05 * kansei_rt_grid.cell);
            continue;
        }
        out.radiance = rtHitRadiance(p, hit.normal, hit.triangle, hit.bary);
        out.found = true;
        return out;
    }
    out.radiance = rtFarField(o, d, n, tanHalf, kansei_rt_exit(o, d));
    return out;
}
