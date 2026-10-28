// The caustics map (rt::RtCausticsEffect) read at a surface: `kansei_rt_caustics_lookup`, the
// irradiance the glass adds to (or takes from) the direct light at world point p of an
// upward-facing surface (normal n), summed over the tiles round the glass casters that cover p and
// over the lights, each at the height its photons landed at (within its tolerance);
// `kansei_rt_caustics_light`, one light's, and `kansei_rt_caustics_light_parts`, the light it adds
// and the light it takes away apart (to weigh what it takes by that light's visibility, its slot
// in RtShadowsEffect's `kansei_rt_caustics_shadow_slot`: the light it adds is its photons', which
// find their own occluders). Shared by the effect's resolve and composite and by passes that light
// surfaces themselves; bind the map as a texture_2d_array (two layers a light at each tile, the
// light added and the light taken away: (tile * lights + light) * 2 and the next) with a
// filtering sampler.

// A caster's tile: a horizontal rectangle of `dims` texels from world xz `min`, `size` metres a
// side; its texels' height is relative to `refY`, and a surface takes its light within
// `tolerance` of it. `box` is the caster's (no deposit lands inside a caster). `live`: it is read;
// `blend`: this frame's photons' weight in it, 0 when it takes none.
struct KanseiRtCausticTile {
    min       : vec2f,
    size      : vec2f,
    dims      : vec2u,
    refY      : f32,
    tolerance : f32,
    boxMin    : vec3f,
    live      : u32,
    boxMax    : vec3f,
    blend     : f32,
}

const KANSEI_RT_CAUSTIC_TILES : u32 = 8u;

struct KanseiRtCaustics {
    enabled   : u32,
    count     : u32,
    intensity : f32,
    // the map's side (texels) a layer
    texels    : u32,
    // irradiance (lux) times this is the photons' fixed point
    scale     : f32,
    // the lights sending photons: a tile's layers are twice as many
    lights    : u32,
    _pad0     : u32,
    _pad1     : u32,
    // each light's slot in RtShadowsEffect's visibility, four a vector
    shadow    : array<vec4u, 2>,
    tiles     : array<KanseiRtCausticTile, 8>,
}

fn kansei_rt_caustics_shadow_slot(c: KanseiRtCaustics, light: u32) -> u32 {
    return c.shadow[light / 4u][light % 4u];
}

// The `light`th light's change at p (n its normal), over the tiles that cover p: the light its
// photons bring through the glass (0) and the light the glass keeps from where it would have passed
// straight on (1, negative).
fn kansei_rt_caustics_light_parts(map: texture_2d_array<f32>, smp: sampler, c: KanseiRtCaustics, p: vec3f, n: vec3f, light: u32) -> array<vec3f, 2> {
    var e = array<vec3f, 2>(vec3f(0.0), vec3f(0.0));
    if (c.enabled == 0u || n.y < 0.3 || light >= c.lights) {
        return e;
    }
    for (var k = 0u; k < min(c.count, KANSEI_RT_CAUSTIC_TILES); k++) {
        let t = c.tiles[k];
        if (t.live == 0u) { continue; }
        let uv = (p.xz - t.min) / t.size;
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { continue; }
        // within the tile's texels of the light's layers, half a texel in from their edge
        let d = vec2f(t.dims);
        let q = clamp(uv * d, vec2f(0.5), d - 0.5) / f32(c.texels);
        for (var part = 0u; part < 2u; part++) {
            let s = textureSampleLevel(map, smp, q, (k * c.lights + light) * 2u + part, 0.0);
            if (abs(s.a + t.refY - p.y) > t.tolerance) { continue; }
            e[part] += s.rgb * c.intensity;
        }
    }
    return e;
}

// The `light`th light's change at p (n its normal), over the tiles that cover p.
fn kansei_rt_caustics_light(map: texture_2d_array<f32>, smp: sampler, c: KanseiRtCaustics, p: vec3f, n: vec3f, light: u32) -> vec3f {
    let e = kansei_rt_caustics_light_parts(map, smp, c, p, n, light);
    return e[0] + e[1];
}

fn kansei_rt_caustics_lookup(map: texture_2d_array<f32>, smp: sampler, c: KanseiRtCaustics, p: vec3f, n: vec3f) -> vec3f {
    var e = vec3f(0.0);
    for (var l = 0u; l < c.lights; l++) {
        e += kansei_rt_caustics_light(map, smp, c, p, n, l);
    }
    return e;
}
