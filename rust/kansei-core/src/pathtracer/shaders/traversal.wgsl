// ── BVH4 traversal (TLAS + BLAS) ─────────────────────────────────────────────
//
// Requires: intersection.wgsl (Ray, HitInfo, ray_aabb, ray_triangle)
// Requires: external bindings for triangles, bvh4_nodes, tlas_bvh4_nodes, instances
//
// BVH4 node layout (8 x vec4f = 128 bytes):
//   [0] child_min_x  [1] child_max_x
//   [2] child_min_y  [3] child_max_y
//   [4] child_min_z  [5] child_max_z
//   [6] children  (vec4i: negative = leaf -(idx+1), positive = internal node)
//   [7] counts    (vec4u: triangle count for leaves, 0 for internal/sentinel)

const TLAS_STACK_SIZE = 16u;
const BLAS_STACK_SIZE = 64u;
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

// ── BLAS traversal ───────────────────────────────────────────────────────────

fn traverse_blas(
    ray: Ray,
    bvh4_root: u32,
    tri_offset: u32,
    tri_count: u32,
    t_max: f32,
    any_hit: bool,
) -> HitInfo {
    var hit: HitInfo;
    hit.t = t_max;
    hit.hit = false;

    if (tri_count == 0u) { return hit; }

    let inv_dir = 1.0 / ray.dir;
    let ox = vec4f(ray.origin.x);
    let oy = vec4f(ray.origin.y);
    let oz = vec4f(ray.origin.z);
    let idx4 = vec4f(inv_dir.x);
    let idy4 = vec4f(inv_dir.y);
    let idz4 = vec4f(inv_dir.z);

    var stack: array<u32, BLAS_STACK_SIZE>;
    var stack_ptr = 0u;
    stack[0] = bvh4_root;
    stack_ptr = 1u;

    var iter = 0u;
    while (stack_ptr > 0u && iter < 16384u) {
        iter += 1u;
        stack_ptr -= 1u;
        let base = stack[stack_ptr] * 8u;

        // Read BVH4 node (8 vec4f = 128 bytes)
        let child_min_x = bvh4_nodes[base + 0u];
        let child_max_x = bvh4_nodes[base + 1u];
        let child_min_y = bvh4_nodes[base + 2u];
        let child_max_y = bvh4_nodes[base + 3u];
        let child_min_z = bvh4_nodes[base + 4u];
        let child_max_z = bvh4_nodes[base + 5u];
        let children    = bitcast<vec4i>(bvh4_nodes[base + 6u]);
        let tri_counts  = bitcast<vec4u>(bvh4_nodes[base + 7u]);

        // Test all 4 child AABBs simultaneously
        let t1x = (child_min_x - ox) * idx4;
        let t2x = (child_max_x - ox) * idx4;
        let t1y = (child_min_y - oy) * idy4;
        let t2y = (child_max_y - oy) * idy4;
        let t1z = (child_min_z - oz) * idz4;
        let t2z = (child_max_z - oz) * idz4;

        let tmin_v = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
        let tmax_v = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));

        // Collect and sort hit children by distance (nearest first on stack top)
        var hit_dist: array<f32, 4>;
        var hit_idx: array<u32, 4>;
        var hit_count = 0u;

        for (var ci = 0u; ci < 4u; ci++) {
            let tmin_c = tmin_v[ci];
            let tmax_c = tmax_v[ci];
            if (tmax_c >= 0.0 && tmin_c <= tmax_c && tmin_c <= hit.t) {
                let child_idx = children[ci];

                if (child_idx < 0) {
                    // Leaf: test triangles immediately
                    let tri_start = u32(-(child_idx + 1));
                    let tri_cnt = tri_counts[ci];
                    for (var ti = 0u; ti < tri_cnt; ti++) {
                        let tri_idx = tri_start + ti;
                        let tbase = (tri_offset + tri_idx) * TRI_STRIDE;
                        let v0 = vec3f(triangles[tbase + 0u], triangles[tbase + 1u], triangles[tbase + 2u]);
                        let v1 = vec3f(triangles[tbase + 3u], triangles[tbase + 4u], triangles[tbase + 5u]);
                        let v2 = vec3f(triangles[tbase + 6u], triangles[tbase + 7u], triangles[tbase + 8u]);
                        let tuv = ray_triangle(ray, v0, v1, v2);
                        if (tuv.x > 0.0 && tuv.x < hit.t) {
                            hit.t = tuv.x;
                            hit.u = tuv.y;
                            hit.v = tuv.z;
                            hit.tri_index = tri_offset + tri_idx;
                            hit.hit = true;
                            if (any_hit) { return hit; }
                            let n0 = vec3f(triangles[tbase + 9u],  triangles[tbase + 10u], triangles[tbase + 11u]);
                            let n1 = vec3f(triangles[tbase + 12u], triangles[tbase + 13u], triangles[tbase + 14u]);
                            let n2 = vec3f(triangles[tbase + 15u], triangles[tbase + 16u], triangles[tbase + 17u]);
                            let w = 1.0 - tuv.y - tuv.z;
                            hit.world_norm = normalize(n0 * w + n1 * tuv.y + n2 * tuv.z);
                            hit.world_pos = ray.origin + ray.dir * tuv.x;
                        }
                    }
                } else {
                    // Internal: collect for sorted push
                    hit_dist[hit_count] = max(tmin_c, 0.0);
                    hit_idx[hit_count] = u32(child_idx);
                    hit_count += 1u;
                }
            }
        }

        // Push internal children sorted farthest-first (nearest popped first)
        if (hit_count > 0u) {
            // Simple insertion sort for up to 4 elements (ascending = farthest first)
            for (var i = 1u; i < hit_count; i++) {
                let kd = hit_dist[i];
                let ki = hit_idx[i];
                var j = i;
                while (j > 0u && hit_dist[j - 1u] < kd) {
                    hit_dist[j] = hit_dist[j - 1u];
                    hit_idx[j] = hit_idx[j - 1u];
                    j -= 1u;
                }
                hit_dist[j] = kd;
                hit_idx[j] = ki;
            }
            // Push as many as fit (sorted farthest-first, so nearest are pushed last = popped first)
            let push_count = min(hit_count, BLAS_STACK_SIZE - stack_ptr);
            for (var i = 0u; i < push_count; i++) {
                stack[stack_ptr] = hit_idx[i];
                stack_ptr += 1u;
            }
        }
    }
    return hit;
}

// ── TLAS traversal ───────────────────────────────────────────────────────────

fn trace_bvh_internal(ray: Ray, any_hit: bool, max_dist: f32) -> HitInfo {
    var hit: HitInfo;
    hit.t = max_dist;
    hit.hit = false;

    let inv_dir = 1.0 / ray.dir;
    let ox = vec4f(ray.origin.x);
    let oy = vec4f(ray.origin.y);
    let oz = vec4f(ray.origin.z);
    let idx4 = vec4f(inv_dir.x);
    let idy4 = vec4f(inv_dir.y);
    let idz4 = vec4f(inv_dir.z);

    var stack: array<u32, TLAS_STACK_SIZE>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    var iter = 0u;
    while (stack_ptr > 0u && iter < 16384u) {
        iter += 1u;
        stack_ptr -= 1u;
        let base = stack[stack_ptr] * 8u;

        // Read TLAS BVH4 node (8 vec4f = 128 bytes, 1 cache line)
        let child_min_x = tlas_bvh4_nodes[base + 0u];
        let child_max_x = tlas_bvh4_nodes[base + 1u];
        let child_min_y = tlas_bvh4_nodes[base + 2u];
        let child_max_y = tlas_bvh4_nodes[base + 3u];
        let child_min_z = tlas_bvh4_nodes[base + 4u];
        let child_max_z = tlas_bvh4_nodes[base + 5u];
        let children    = bitcast<vec4i>(tlas_bvh4_nodes[base + 6u]);
        let child_cnts  = bitcast<vec4u>(tlas_bvh4_nodes[base + 7u]);

        // Test all 4 child AABBs simultaneously
        let t1x = (child_min_x - ox) * idx4;
        let t2x = (child_max_x - ox) * idx4;
        let t1y = (child_min_y - oy) * idy4;
        let t2y = (child_max_y - oy) * idy4;
        let t1z = (child_min_z - oz) * idz4;
        let t2z = (child_max_z - oz) * idz4;

        let tmin_v = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
        let tmax_v = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));

        var hit_dist: array<f32, 4>;
        var hit_idx: array<u32, 4>;
        var hit_count = 0u;

        for (var ci = 0u; ci < 4u; ci++) {
            let tmin_c = tmin_v[ci];
            let tmax_c = tmax_v[ci];
            if (tmax_c >= 0.0 && tmin_c <= tmax_c && tmin_c <= hit.t) {
                let child_idx = children[ci];

                if (child_idx < 0) {
                    // Leaf: instance intersection (skip sentinels with count 0)
                    if (child_cnts[ci] > 0u) {
                        let inst_idx = u32(-(child_idx + 1));
                        let inst = instances[inst_idx];
                        let local_ray = transform_ray_to_local(ray, inst);

                        let blas_hit = traverse_blas(
                            local_ray,
                            inst.blas_node_offset,
                            inst.blas_tri_offset,
                            inst.blas_tri_count,
                            hit.t,
                            any_hit,
                        );

                        if (blas_hit.hit && blas_hit.t < hit.t) {
                            hit = blas_hit;
                            hit.instance_id = inst_idx;
                            hit.mat_index = inst.material_index;
                            if (any_hit) { return hit; }
                            hit.world_pos = transform_point_to_world(blas_hit.world_pos, inst);
                            hit.world_norm = transform_normal_to_world(blas_hit.world_norm, inst);
                        }
                    }
                } else {
                    // Internal: collect for sorted push
                    hit_dist[hit_count] = max(tmin_c, 0.0);
                    hit_idx[hit_count] = u32(child_idx);
                    hit_count += 1u;
                }
            }
        }

        // Push internal children sorted farthest-first (nearest popped first)
        if (hit_count > 0u) {
            for (var i = 1u; i < hit_count; i++) {
                let kd = hit_dist[i];
                let ki = hit_idx[i];
                var j = i;
                while (j > 0u && hit_dist[j - 1u] < kd) {
                    hit_dist[j] = hit_dist[j - 1u];
                    hit_idx[j] = hit_idx[j - 1u];
                    j -= 1u;
                }
                hit_dist[j] = kd;
                hit_idx[j] = ki;
            }
            let push_count = min(hit_count, TLAS_STACK_SIZE - stack_ptr);
            for (var i = 0u; i < push_count; i++) {
                stack[stack_ptr] = hit_idx[i];
                stack_ptr += 1u;
            }
        }
    }
    return hit;
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
