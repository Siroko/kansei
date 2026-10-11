// The caustics map (rt::RtCausticsEffect) read at a surface: `kansei_rt_caustics_lookup`, the
// irradiance the glass and the water add to (or take from) the direct light at world point p of an
// upward-facing surface (normal n) within the map's rectangle, at the height the photons landed
// at (within `heightTolerance`); and `kansei_rt_caustics_focus`, the caustic lines of the direct
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

// The water's caustics at intensity 1 (`kansei_water_caustic_lines`): the share of the light
// between the lines, and on them at their brightest (a game's pool: lines plain to see, the
// light's mean about kept).
const WATER_CAUSTIC_BASE : f32 = 0.45;
const WATER_CAUSTIC_PEAK : f32 = 4.0;

// The share of the direct light a surface at p under the water receives through its ripples (1:
// as through still water; 1 out of the water or with `focus` off): its caustics, faded in over
// the first 10 cm of depth.
fn kansei_rt_caustics_focus(c: KanseiRtCaustics, p: vec3f) -> f32 {
    let uv = (p.xz - c.waterMin) / c.waterSize;
    if (c.enabled == 0u || c.focus == 0u || p.y >= c.waterLevel || any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
        return 1.0;
    }
    let lines = kansei_water_caustic_lines(p.xz, c.time);
    let amount = clamp(c.ripples, 0.0, 1.0) * c.intensity * smoothstep(0.0, 0.1, c.waterLevel - p.y);
    return max(1.0 + amount * (mix(WATER_CAUSTIC_BASE, WATER_CAUSTIC_PEAK, lines) - 1.0), 0.0);
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
