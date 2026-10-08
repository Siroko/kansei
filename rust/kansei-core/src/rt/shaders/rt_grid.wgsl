// Ray tracing through an `RtGrid` (rt/grid.rs), for any compute pass. Declare the grid's buffers
// with `RtGrid::bindings_wgsl(group, first)` (`kansei_rt_grid`, `kansei_rt_triangles`,
// `kansei_rt_cells`), and define `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, whether an
// alpha-tested triangle's surface is there at `uv` (rt::RT_OPAQUE_WGSL: always). Included after
// rt_types.wgsl.
//
// `kansei_rt_trace(origin, dir, tMin, tMax, flags)` tests the big triangles, then walks the cells
// the ray crosses within the grid's box (Amanatides & Woo's 3D DDA, leaving empty 4^3 macro cells
// in one step), testing the triangles listed in each (Moller & Trumbore, both sides), and stops at
// the first cell whose exit lies past the closest hit; with KANSEI_RT_ANY_HIT at the first hit.

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

// Moller-Trumbore, both sides: (t, u, v), t < 0 for a miss.
fn kansei_rt_intersect(o: vec3f, d: vec3f, v0: vec3f, e1: vec3f, e2: vec3f, tMin: f32, tMax: f32) -> vec3f {
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-20) { return vec3f(-1.0); }
    let inv = 1.0 / det;
    let s = o - v0;
    let u = dot(s, p) * inv;
    if (u < -KANSEI_RT_BARY_EPSILON || u > 1.0 + KANSEI_RT_BARY_EPSILON) { return vec3f(-1.0); }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if (v < -KANSEI_RT_BARY_EPSILON || u + v > 1.0 + KANSEI_RT_BARY_EPSILON) { return vec3f(-1.0); }
    let t = dot(e2, q) * inv;
    if (t <= tMin || t >= tMax) { return vec3f(-1.0); }
    return vec3f(t, u, v);
}

// Test triangle `id`, closer than the hit so far; alpha-tested ones where kansei_rt_covered says,
// glass only with KANSEI_RT_GLASS_HITS.
fn kansei_rt_test(hit: ptr<function, KanseiRtHit>, o: vec3f, d: vec3f, tMin: f32, id: u32, flags: u32) -> bool {
    let a = kansei_rt_triangles[id * 4u];
    let surface = bitcast<u32>(a.w);
    if ((surface & KANSEI_RT_GLASS) != 0u && (flags & KANSEI_RT_GLASS_HITS) == 0u) {
        return false;
    }
    let r = kansei_rt_intersect(o, d, a.xyz, kansei_rt_triangles[id * 4u + 1u].xyz, kansei_rt_triangles[id * 4u + 2u].xyz, tMin, (*hit).t);
    (*hit).tests += 1u;
    if (r.x < 0.0) {
        return false;
    }
    if ((surface & KANSEI_RT_ALPHA) != 0u && (flags & KANSEI_RT_SOLID) == 0u) {
        if (!kansei_rt_covered((surface >> 8u) & 255u, kansei_rt_uv(id, r.yz))) {
            return false;
        }
    }
    (*hit).t = r.x;
    (*hit).found = true;
    (*hit).triangle = id;
    (*hit).bary = r.yz;
    return true;
}

fn kansei_rt_cell_index(c: vec3i) -> u32 {
    let d = vec3i(kansei_rt_grid.dims);
    return u32((c.z * d.y + c.y) * d.x + c.x);
}

fn kansei_rt_macro_index(c: vec3i) -> u32 {
    let m = vec3u(c) >> vec3u(KANSEI_RT_MACRO_SHIFT);
    let d = kansei_rt_grid.macroDims;
    return (m.z * d.y + m.y) * d.x + m.x;
}

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

// The closest hit (or with KANSEI_RT_ANY_HIT any hit) of the ray between tMin and tMax, among the
// grid's triangles (those within its box).
fn kansei_rt_trace(o: vec3f, d: vec3f, tMin: f32, tMax: f32, flags: u32) -> KanseiRtHit {
    var hit : KanseiRtHit;
    hit.t = tMax;
    hit.found = false;
    let g = kansei_rt_grid;
    let anyHit = (flags & KANSEI_RT_ANY_HIT) != 0u;
    // the big triangles, every ray
    let big = min(kansei_rt_cells[g.bigBase], g.bigCapacity);
    for (var k = 0u; k < big; k++) {
        if (kansei_rt_test(&hit, o, d, tMin, kansei_rt_cells[g.bigBase + 1u + k], flags) && anyHit) {
            kansei_rt_finish(&hit, d);
            return hit;
        }
    }
    let invD = kansei_rt_inv_dir(d);
    let gmin = g.origin;
    let box = kansei_rt_slab(o, invD, gmin, kansei_rt_bounds_max());
    var t = max(box.x, tMin);
    let tEnd = box.y;
    if (box.x > box.y || t >= min(tEnd, hit.t)) {
        kansei_rt_finish(&hit, d);
        return hit;
    }
    let dims = vec3i(g.dims);
    let stepI = select(vec3i(-1), vec3i(1), d >= vec3f(0.0));
    let tDelta = abs(invD) * g.cell;
    let block = g.cell * f32(1u << KANSEI_RT_MACRO_SHIFT);
    var restart = true;
    var cell = vec3i(0);
    var tNext = vec3f(0.0);
    let limit = 2u * (g.dims.x + g.dims.y + g.dims.z) + 16u;
    for (var iter = 0u; iter < limit; iter++) {
        if (restart) {
            let p = (o + d * t - gmin) / g.cell;
            cell = clamp(vec3i(floor(p)), vec3i(0), dims - 1);
            tNext = (vec3f(cell + max(stepI, vec3i(0))) * g.cell + gmin - o) * invD;
            restart = false;
        }
        hit.cells += 1u;
        if ((g.flags & KANSEI_RT_MACRO_SKIP) != 0u && kansei_rt_cells[g.macroBase + kansei_rt_macro_index(cell)] == 0u) {
            // leave the empty macro cell in one step
            let lo = gmin + vec3f(cell >> vec3u(KANSEI_RT_MACRO_SHIFT)) * block;
            let m = kansei_rt_slab(o, invD, lo, lo + vec3f(block));
            t = max(m.y, t) + 0.5 * g.epsilon;
            if (t >= min(tEnd, hit.t)) {
                break;
            }
            restart = true;
            continue;
        }
        let c = kansei_rt_cell_index(cell);
        var start = 0u;
        if (c > 0u) {
            start = kansei_rt_cells[c - 1u];
        }
        let end = min(kansei_rt_cells[c], g.refCapacity);
        for (var k = start; k < end; k++) {
            if (kansei_rt_test(&hit, o, d, tMin, kansei_rt_cells[g.refsBase + k], flags) && anyHit) {
                kansei_rt_finish(&hit, d);
                return hit;
            }
        }
        let tCell = min(min(tNext.x, tNext.y), tNext.z);
        if (tCell >= min(tEnd, hit.t)) {
            break;
        }
        t = tCell;
        if (tNext.x <= tNext.y && tNext.x <= tNext.z) {
            cell.x += stepI.x;
            tNext.x += tDelta.x;
        } else if (tNext.y <= tNext.z) {
            cell.y += stepI.y;
            tNext.y += tDelta.y;
        } else {
            cell.z += stepI.z;
            tNext.z += tDelta.z;
        }
        if (any(cell < vec3i(0)) || any(cell >= dims)) {
            break;
        }
    }
    kansei_rt_finish(&hit, d);
    return hit;
}
