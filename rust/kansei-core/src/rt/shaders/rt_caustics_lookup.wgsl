// The caustics map (rt::RtCausticsEffect) read at a surface: `kansei_rt_caustics_lookup`, the
// irradiance the glass adds to (or takes from) the direct light at world point p of an
// upward-facing surface (normal n), summed over the tiles round the glass casters that cover p,
// each at the height its photons landed at (within its tolerance). Shared by the effect's resolve
// and composite and by passes that light surfaces themselves; bind the map as a
// texture_2d_array (a layer a tile) with a filtering sampler.

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
    _pad0     : u32,
    _pad1     : u32,
    _pad2     : u32,
    tiles     : array<KanseiRtCausticTile, 8>,
}

fn kansei_rt_caustics_lookup(map: texture_2d_array<f32>, smp: sampler, c: KanseiRtCaustics, p: vec3f, n: vec3f) -> vec3f {
    if (c.enabled == 0u || n.y < 0.3) {
        return vec3f(0.0);
    }
    var e = vec3f(0.0);
    for (var k = 0u; k < min(c.count, KANSEI_RT_CAUSTIC_TILES); k++) {
        let t = c.tiles[k];
        if (t.live == 0u) { continue; }
        let uv = (p.xz - t.min) / t.size;
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { continue; }
        // within the tile's texels of its layer, half a texel in from their edge
        let d = vec2f(t.dims);
        let q = clamp(uv * d, vec2f(0.5), d - 0.5) / f32(c.texels);
        let s = textureSampleLevel(map, smp, q, k, 0.0);
        if (abs(s.a + t.refY - p.y) > t.tolerance) { continue; }
        e += s.rgb;
    }
    return e * c.intensity;
}
