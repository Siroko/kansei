// ── Hardware ray query traversal (TLAS + BLAS) ──────────────────────────────
//
// traversal.wgsl's `trace_bvh_internal` with hardware ray queries (RtBackend::Hardware, native
// only; pathtracer/accel.rs): the same HitInfo for the same triangle, over an acceleration
// structure of the scene. A BLAS a mesh (primitive i = the mesh's triangle i, in `triangles`'
// order from the instance's `blas_tri_offset`) and a TLAS instance a scene instance, in order.
//
// Requires: intersection.wgsl (Ray, HitInfo), instances.wgsl (Instance and its transforms)
// Requires: external bindings for triangles, instances and `scene_tlas` (acceleration_structure).
// The module's `enable` goes first (`rt::hoist_wgsl_enables`).

enable wgpu_ray_query;

/// ray_triangle's nearest distance, along the ray as given
const RQ_T_MIN = 1e-5;

fn trace_bvh_internal(ray: Ray, any_hit: bool, max_dist: f32) -> HitInfo {
    pt_count_ray();
    var hit: HitInfo;
    hit.t = max_dist;
    hit.hit = false;

    // every triangle is opaque and two-sided, as ray_triangle's are
    var flags = RAY_FLAG_FORCE_OPAQUE;
    if (any_hit) { flags |= RAY_FLAG_TERMINATE_ON_FIRST_HIT; }
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_tlas, RayDesc(flags, 0xffu, RQ_T_MIN, max_dist, ray.origin, ray.dir));
    while (rayQueryProceed(&rq)) {}
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind == RAY_QUERY_INTERSECTION_NONE) { return hit; }

    let inst_idx = h.instance_custom_data;
    let inst = instances[inst_idx];
    hit.t = h.t;
    hit.u = h.barycentrics.x;
    hit.v = h.barycentrics.y;
    hit.tri_index = inst.blas_tri_offset + h.primitive_index;
    hit.instance_id = inst_idx;
    hit.mat_index = inst.material_index;
    hit.hit = true;
    if (any_hit) { return hit; }

    let tbase = hit.tri_index * TRI_STRIDE;
    let n0 = vec3f(triangles[tbase + 9u],  triangles[tbase + 10u], triangles[tbase + 11u]);
    let n1 = vec3f(triangles[tbase + 12u], triangles[tbase + 13u], triangles[tbase + 14u]);
    let n2 = vec3f(triangles[tbase + 15u], triangles[tbase + 16u], triangles[tbase + 17u]);
    let w = 1.0 - hit.u - hit.v;
    hit.world_norm = transform_normal_to_world(normalize(n0 * w + n1 * hit.u + n2 * hit.v), inst);
    hit.world_pos = ray.origin + ray.dir * hit.t;
    return hit;
}
