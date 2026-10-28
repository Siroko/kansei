// Ray tracing through an `RtGrid` with hardware ray queries (rt/query.rs; wgpu's
// EXPERIMENTAL_RAY_QUERY, native only): the same `kansei_rt_trace` as rt_grid.wgsl, returning the
// same `KanseiRtHit`, over an acceleration structure of the grid's gathered triangles. Declare the
// grid's bindings with `RtBackend::Hardware.bindings_wgsl(group, first)` (`kansei_rt_grid`,
// `kansei_rt_triangles`, `kansei_rt_tlas`), and define `kansei_rt_covered` as for the grid.
// Included after rt_types.wgsl and rt_hit.wgsl; the module's `enable` goes first
// (`rt::hoist_wgsl_enables`).
//
// The BLAS holds triangle `id` as primitive `id` (of its opaque geometry, or of its second,
// non-opaque one when alpha-tested or glass), so a hit names the same triangle as the grid's
// would. Only non-opaque candidates come back to the shader, for `kansei_rt_covered` and glass.
// `cells` is 0 and `tests` counts those candidates.

enable wgpu_ray_query;

// The closest hit (or with KANSEI_RT_ANY_HIT any hit) of the ray between tMin and tMax, among the
// grid's triangles, within its box.
fn kansei_rt_trace(o: vec3f, d: vec3f, tMin: f32, tMax: f32, flags: u32) -> KanseiRtHit {
    var hit : KanseiRtHit;
    hit.t = tMax;
    hit.found = false;
    // the triangles meet the box; what lies past it is the caller's (kansei_rt_exit), and the
    // BLAS's filler lies outside it
    let box = kansei_rt_slab(o, kansei_rt_inv_dir(d), kansei_rt_bounds_min(), kansei_rt_bounds_max());
    let tStart = max(max(tMin, box.x - kansei_rt_grid.epsilon), 0.0);
    let tEnd = min(tMax, box.y + kansei_rt_grid.epsilon);
    if (box.x > box.y || tEnd <= tStart) {
        return hit;
    }
    let solid = (flags & KANSEI_RT_SOLID) != 0u;
    let glassHits = (flags & KANSEI_RT_GLASS_HITS) != 0u;
    let noApproximate = (flags & KANSEI_RT_NO_APPROXIMATE) != 0u;
    var rayFlags = RAY_FLAG_NONE;
    if ((flags & KANSEI_RT_ANY_HIT) != 0u) {
        rayFlags |= RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    }
    if (solid && glassHits && !noApproximate) {
        // nothing to decide per candidate
        rayFlags |= RAY_FLAG_FORCE_OPAQUE;
    }
    var rq : ray_query;
    rayQueryInitialize(&rq, kansei_rt_tlas, RayDesc(rayFlags, 0xffu, tStart, tEnd, o, d));
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        hit.tests += 1u;
        let id = c.primitive_index;
        let surface = kansei_rt_surface(id);
        if ((surface & KANSEI_RT_GLASS) != 0u && !glassHits) {
            continue;
        }
        if ((surface & KANSEI_RT_APPROXIMATE) != 0u && noApproximate) {
            continue;
        }
        if ((surface & KANSEI_RT_ALPHA) != 0u && !solid) {
            if (!kansei_rt_covered((surface >> 8u) & 255u, kansei_rt_uv(id, c.barycentrics))) {
                continue;
            }
        }
        rayQueryConfirmIntersection(&rq);
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind != RAY_QUERY_INTERSECTION_NONE) {
        hit.t = h.t;
        hit.found = true;
        hit.triangle = h.primitive_index;
        hit.bary = h.barycentrics;
    }
    kansei_rt_finish(&hit, d);
    return hit;
}
