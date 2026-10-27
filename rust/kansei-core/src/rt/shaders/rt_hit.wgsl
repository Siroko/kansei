// What a trace through an `RtGrid` returns and how to read its hit, whichever backend traces it
// (rt_grid.wgsl's grid, rt_query.wgsl's hardware ray queries): `KanseiRtHit`, the trace flags, the
// grid's box (`kansei_rt_exit`, `kansei_rt_contains`) and a hit triangle's surface by its id
// (`kansei_rt_albedo`, `kansei_rt_uv`, `kansei_rt_shading_normal`...). Included after
// rt_types.wgsl, before the backend's `kansei_rt_trace`.

struct KanseiRtHit {
    // distance along `dir` (its length the unit); tMax on a miss
    t        : f32,
    found    : bool,
    triangle : u32,
    // the unit geometric normal, facing the ray's origin
    normal   : vec3f,
    // barycentrics of v1 and v2
    bary     : vec2f,
    // cost: cells visited, triangles tested
    cells    : u32,
    tests    : u32,
}

// stop at the first hit, not the closest
const KANSEI_RT_ANY_HIT : u32 = 1u;
// alpha-tested triangles count as solid (no kansei_rt_covered)
const KANSEI_RT_SOLID : u32 = 2u;
// glass triangles (KANSEI_RT_GLASS) are hit; without it rays pass straight through them
const KANSEI_RT_GLASS_HITS : u32 = 4u;
const KANSEI_RT_BARY_EPSILON : f32 = 1e-6;

fn kansei_rt_bounds_min() -> vec3f {
    return kansei_rt_grid.origin;
}

fn kansei_rt_bounds_max() -> vec3f {
    return kansei_rt_grid.origin + vec3f(kansei_rt_grid.dims) * kansei_rt_grid.cell;
}

fn kansei_rt_contains(p: vec3f) -> bool {
    return all(p >= kansei_rt_bounds_min()) && all(p <= kansei_rt_bounds_max());
}

fn kansei_rt_inv_dir(d: vec3f) -> vec3f {
    let eps = 1e-12;
    return 1.0 / select(d, select(vec3f(-eps), vec3f(eps), d >= vec3f(0.0)), abs(d) < vec3f(eps));
}

// The slab test: where the ray enters and leaves a box (enter > leave: it misses).
fn kansei_rt_slab(o: vec3f, invD: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
    let a = (bmin - o) * invD;
    let b = (bmax - o) * invD;
    let lo = min(a, b);
    let hi = max(a, b);
    return vec2f(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

// Where the ray leaves the grid's box (0 when it never meets it ahead).
fn kansei_rt_exit(o: vec3f, d: vec3f) -> f32 {
    let s = kansei_rt_slab(o, kansei_rt_inv_dir(d), kansei_rt_bounds_min(), kansei_rt_bounds_max());
    return select(0.0, max(s.y, 0.0), s.x <= s.y);
}

// Triangle `id`'s surface word: flags (KANSEI_RT_ALPHA) in the low byte, the alpha layer in the next.
fn kansei_rt_surface(id: u32) -> u32 {
    return bitcast<u32>(kansei_rt_triangles[id * 4u].w);
}

fn kansei_rt_albedo(id: u32) -> vec3f {
    return unpack4x8unorm(bitcast<u32>(kansei_rt_triangles[id * 4u + 1u].w)).rgb;
}

// Triangle `id`'s reflectance at normal incidence (RtSurface::specular: 0 for a matte one), kept in
// the albedo's alpha as 1 - F0, as the GBuffer keeps it.
fn kansei_rt_specular(id: u32) -> f32 {
    return 1.0 - unpack4x8unorm(bitcast<u32>(kansei_rt_triangles[id * 4u + 1u].w)).a;
}

// The light triangle `id` emits (cd/m²; RtSurface::record_emission), RGBE as rt_emission.wgsl
// packs it.
fn kansei_rt_emission(id: u32) -> vec3f {
    let w = bitcast<u32>(kansei_rt_triangles[id * 4u + 3u].w);
    let e = w >> 24u;
    if (e == 0u) {
        return vec3f(0.0);
    }
    return (vec3f(f32(w & 255u), f32((w >> 8u) & 255u), f32((w >> 16u) & 255u)) + 0.5) * exp2(f32(e) - 136.0);
}

// The id of the source triangle `id` came from (the gather's `source`).
fn kansei_rt_source(id: u32) -> u32 {
    return bitcast<u32>(kansei_rt_triangles[id * 4u + 2u].w) >> 20u;
}

// The record (instance) of its source it came from.
fn kansei_rt_record(id: u32) -> u32 {
    return bitcast<u32>(kansei_rt_triangles[id * 4u + 2u].w) & 0xfffffu;
}

fn kansei_rt_uv(id: u32, bary: vec2f) -> vec2f {
    let w = bitcast<vec4u>(kansei_rt_triangles[id * 4u + 3u]);
    return unpack2x16float(w.x) * (1.0 - bary.x - bary.y) + unpack2x16float(w.y) * bary.x + unpack2x16float(w.z) * bary.y;
}

// The normal to shade a hit with: its triangle's vertex normals interpolated where it carries
// them (KANSEI_RT_SMOOTH), else the geometric one; on the side `ng` (the hit's normal) faces.
fn kansei_rt_shading_normal(id: u32, bary: vec2f, ng: vec3f) -> vec3f {
    let surface = bitcast<u32>(kansei_rt_triangles[id * 4u].w);
    if ((surface & KANSEI_RT_SMOOTH) == 0u) {
        return ng;
    }
    let w = bitcast<vec4u>(kansei_rt_triangles[id * 4u + 3u]);
    let n = normalize(kansei_rt_unpack_normal(w.x) * (1.0 - bary.x - bary.y) + kansei_rt_unpack_normal(w.y) * bary.x + kansei_rt_unpack_normal(w.z) * bary.y);
    return select(-n, n, dot(n, ng) >= 0.0);
}

// The hit's unit geometric normal, facing the ray's origin.
fn kansei_rt_finish(hit: ptr<function, KanseiRtHit>, d: vec3f) {
    if ((*hit).found) {
        let id = (*hit).triangle;
        var n = normalize(cross(kansei_rt_triangles[id * 4u + 1u].xyz, kansei_rt_triangles[id * 4u + 2u].xyz));
        if (dot(n, d) > 0.0) {
            n = -n;
        }
        (*hit).normal = n;
    }
}

