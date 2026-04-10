// ── Ray-primitive intersection utilities ─────────────────────────────────────
//
// Provides:
//   Ray        — ray with origin + direction
//   HitInfo    — closest-hit result
//   ray_aabb   — slab-based AABB test (returns tmin or -1)
//   ray_triangle — Moller-Trumbore triangle test (returns (t,u,v) or (-1,0,0))

struct Ray {
    origin : vec3f,
    dir    : vec3f,
}

struct HitInfo {
    t           : f32,
    u           : f32,
    v           : f32,
    tri_index   : u32,
    instance_id : u32,
    mat_index   : u32,
    world_pos   : vec3f,
    world_norm  : vec3f,
    hit         : bool,
}

/// Slab-based ray-AABB intersection.
/// Returns the entry distance (>= 0) on hit, or -1.0 on miss.
/// `inv_dir` must be 1.0 / ray.dir (precomputed by caller).
fn ray_aabb(ray: Ray, bmin: vec3f, bmax: vec3f, t_max: f32, inv_dir: vec3f) -> f32 {
    let t1 = (bmin - ray.origin) * inv_dir;
    let t2 = (bmax - ray.origin) * inv_dir;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmax < 0.0 || tmin > tmax || tmin > t_max) { return -1.0; }
    return max(tmin, 0.0);
}

/// Branchless Moller-Trumbore ray-triangle intersection.
/// Returns (t, u, v) with t < 0 on miss.
fn ray_triangle(ray: Ray, v0: vec3f, v1: vec3f, v2: vec3f) -> vec3f {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(ray.dir, e2);
    let a = dot(e1, h);
    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * dot(s, h);
    let q = cross(s, e1);
    let v = f * dot(ray.dir, q);
    let t = f * dot(e2, q);
    let valid = abs(a) >= 1e-8 && u >= 0.0 && u <= 1.0 && v >= 0.0 && u + v <= 1.0 && t >= 1e-5;
    return select(vec3f(-1.0, 0.0, 0.0), vec3f(t, u, v), valid);
}
