// The caustics map (rt::RtCausticsEffect) read at a surface: `kansei_rt_caustics_lookup`, the
// irradiance the glass and the water add to (or take from) the direct light at world point p of an
// upward-facing surface (normal n) within the map's rectangle, at the height the photons landed
// at (within `heightTolerance`); and `kansei_rt_caustics_focus`, the ripples' focus of the direct
// light on a surface under the water (the water's caustics without photons: `focus` on). Shared by
// the effect's composite and RtReflectionsEffect's hits; prefixed with WATER_RIPPLES_WGSL.

struct KanseiRtCaustics {
    mapMin          : vec2f,
    mapSize         : vec2f,
    enabled         : u32,
    heightTolerance : f32,
    intensity       : f32,
    focus           : u32,
    waterMin        : vec2f,
    waterSize       : vec2f,
    waterLevel      : f32,
    ripples         : f32,
    time            : f32,
    ior             : f32,
}

// How much more than physical still-water optics the ripples focus at intensity 1 (a game's
// pool, its caustic lines plain to see).
const WATER_FOCUS_GAIN : f32 = 2.5;

// The share of the direct light a surface at p under the water receives through its ripples (1:
// as through still water; 1 out of the water or with `focus` off).
fn kansei_rt_caustics_focus(c: KanseiRtCaustics, p: vec3f) -> f32 {
    let uv = (p.xz - c.waterMin) / c.waterSize;
    if (c.enabled == 0u || c.focus == 0u || p.y >= c.waterLevel || any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
        return 1.0;
    }
    // (the intensity sharpens the focus, as deeper water would: its lines narrower and brighter)
    return kansei_water_focus(p.xz, c.time, c.ripples * c.intensity * WATER_FOCUS_GAIN, c.waterLevel - p.y, c.ior);
}

fn kansei_rt_caustics_lookup(map: texture_2d<f32>, smp: sampler, c: KanseiRtCaustics, p: vec3f, n: vec3f) -> vec3f {
    if (c.enabled == 0u || n.y < 0.3) {
        return vec3f(0.0);
    }
    let uv = (p.xz - c.mapMin) / c.mapSize;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
        return vec3f(0.0);
    }
    let s = textureSampleLevel(map, smp, uv, 0.0);
    if (abs(s.a - p.y) > c.heightTolerance) {
        return vec3f(0.0);
    }
    return s.rgb * c.intensity;
}
