// ── The path tracer's scene: triangles and instances ──────────────────────────
//
// Shared by both traversals (traversal.wgsl, the BVH4 in compute; traversal_query.wgsl, hardware
// ray queries), which define `trace_bvh_internal(ray, any_hit, max_dist)`. Requires external
// bindings `triangles` (array<f32>, TRI_STRIDE floats a triangle) and `instances`
// (array<Instance>), and `fn pt_count_ray()`, called once a ray (empty but when counting rays).

const TRI_STRIDE = 24u; // 24 floats per triangle (pos + normal + uv per vertex)

struct Instance {
    transform0     : vec4f, // row 0 of 3x4 affine (object-to-world)
    transform1     : vec4f, // row 1
    transform2     : vec4f, // row 2
    inv_transform0 : vec4f, // row 0 of inverse 3x4 (world-to-object)
    inv_transform1 : vec4f, // row 1
    inv_transform2 : vec4f, // row 2
    blas_node_offset : u32,
    blas_tri_offset  : u32,
    blas_tri_count   : u32,
    material_index   : u32,
}

// ── Instance transforms ──────────────────────────────────────────────────────

fn transform_ray_to_local(ray: Ray, inst: Instance) -> Ray {
    var local: Ray;
    // Apply inverse affine transform
    local.origin = vec3f(
        dot(inst.inv_transform0.xyz, ray.origin) + inst.inv_transform0.w,
        dot(inst.inv_transform1.xyz, ray.origin) + inst.inv_transform1.w,
        dot(inst.inv_transform2.xyz, ray.origin) + inst.inv_transform2.w,
    );
    local.dir = vec3f(
        dot(inst.inv_transform0.xyz, ray.dir),
        dot(inst.inv_transform1.xyz, ray.dir),
        dot(inst.inv_transform2.xyz, ray.dir),
    );
    return local;
}

fn transform_point_to_world(p: vec3f, inst: Instance) -> vec3f {
    return vec3f(
        dot(inst.transform0.xyz, p) + inst.transform0.w,
        dot(inst.transform1.xyz, p) + inst.transform1.w,
        dot(inst.transform2.xyz, p) + inst.transform2.w,
    );
}

/// Normal transform = (M^-1)^T * n, i.e. columns of the inverse matrix.
fn transform_normal_to_world(n: vec3f, inst: Instance) -> vec3f {
    return normalize(vec3f(
        inst.inv_transform0.x * n.x + inst.inv_transform1.x * n.y + inst.inv_transform2.x * n.z,
        inst.inv_transform0.y * n.x + inst.inv_transform1.y * n.y + inst.inv_transform2.y * n.z,
        inst.inv_transform0.z * n.x + inst.inv_transform1.z * n.y + inst.inv_transform2.z * n.z,
    ));
}

/// World-space triangle area for MIS on emissive geometry hits.
fn get_world_tri_area(tri_idx: u32, inst_id: u32) -> f32 {
    let base = tri_idx * TRI_STRIDE;
    let v0 = vec3f(triangles[base], triangles[base + 1u], triangles[base + 2u]);
    let v1 = vec3f(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]);
    let v2 = vec3f(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u]);
    let inst = instances[inst_id];
    let w0 = transform_point_to_world(v0, inst);
    let w1 = transform_point_to_world(v1, inst);
    let w2 = transform_point_to_world(v2, inst);
    return 0.5 * length(cross(w1 - w0, w2 - w0));
}

/// Trace a ray against the full scene (TLAS -> BLAS). Returns closest hit.
fn trace_bvh(ray: Ray) -> HitInfo {
    return trace_bvh_internal(ray, false, 1e30);
}

/// Shadow ray test. Returns true if any geometry is hit within max_dist.
fn trace_bvh_shadow(ray: Ray, max_dist: f32) -> bool {
    let hit = trace_bvh_internal(ray, true, max_dist);
    return hit.hit;
}
