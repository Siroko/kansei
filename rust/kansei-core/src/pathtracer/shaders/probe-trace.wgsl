// ── Probe ray tracing ────────────────────────────────────────────────────────
//
// Traces rays from irradiance probe positions using spherical Fibonacci
// sampling. Writes ray results (hit radiance + direction) to a buffer.
//
// Requires: intersection.wgsl, traversal.wgsl concatenated before this file,
//           plus BVH storage binding preamble.

struct ProbeTraceParams {
    grid_min       : vec3f,
    grid_step_x    : f32,
    grid_dims      : vec3u,
    grid_step_y    : f32,
    grid_step_z    : f32,
    rays_per_probe : u32,
    frame_index    : u32,
    light_count    : u32,
    max_distance   : f32,
    _pad0          : f32,
    _pad1          : f32,
    _pad2          : f32,
}

const LIGHT_DIRECTIONAL = 1u;
const LIGHT_AREA        = 2u;
const LIGHT_POINT       = 3u;

struct LightData {
    position   : vec3f,
    light_type : u32,
    color      : vec3f,
    intensity  : f32,
    normal     : vec3f,
    _pad       : f32,
    extra      : vec4f,
}

struct MaterialData {
    albedo             : vec3f,
    roughness          : f32,
    metallic           : f32,
    ior                : f32,
    max_bounces        : f32,
    transmission       : f32,
    absorption_color   : vec3f,
    absorption_density : f32,
    emissive           : vec3f,
    emissive_intensity : f32,
}

@group(0) @binding(0) var<uniform> params           : ProbeTraceParams;
// BVH bindings 1-4 injected by preamble: triangles, bvh4_nodes, tlas_bvh4_nodes, instances
@group(0) @binding(5) var<storage, read> materials   : array<MaterialData>;
@group(0) @binding(6) var<storage, read> scene_lights : array<LightData>;
@group(0) @binding(7) var<storage, read_write> ray_results : array<vec4f>;
@group(0) @binding(8) var<storage, read> probe_sh_prev : array<vec4f>;

// ── PCG random ───────────────────────────────────────────────────────────────
var<private> rng_state: u32;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn next_random() -> f32 {
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967295.0;
}

// ── Spherical Fibonacci for uniform sphere sampling ──────────────────────────
const GOLDEN_RATIO = 1.6180339887;

fn spherical_fibonacci(idx: u32, total: u32) -> vec3f {
    let i = f32(idx) + 0.5;
    let n = f32(total);
    let phi = 2.0 * 3.14159265 * (i / GOLDEN_RATIO - floor(i / GOLDEN_RATIO));
    let cos_theta = 1.0 - 2.0 * i / n;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn probe_world_pos(ix: u32, iy: u32, iz: u32) -> vec3f {
    return params.grid_min + vec3f(
        f32(ix) * params.grid_step_x,
        f32(iy) * params.grid_step_y,
        f32(iz) * params.grid_step_z,
    );
}

fn probe_index_to_coord(idx: u32) -> vec3u {
    let x = idx % params.grid_dims.x;
    let y = (idx / params.grid_dims.x) % params.grid_dims.y;
    let z = idx / (params.grid_dims.x * params.grid_dims.y);
    return vec3u(x, y, z);
}

// ── Probe grid sampling (previous frame's SH for multi-bounce) ──────────────
const SH_STRIDE = 9u;

fn compute_sh_basis_probe(d: vec3f) -> array<f32, 9> {
    let x = d.x; let y = d.y; let z = d.z;
    var sh: array<f32, 9>;
    sh[0] = 0.282095;
    sh[1] = 0.488603 * y;
    sh[2] = 0.488603 * z;
    sh[3] = 0.488603 * x;
    sh[4] = 1.092548 * x * y;
    sh[5] = 1.092548 * y * z;
    sh[6] = 0.315392 * (3.0 * z * z - 1.0);
    sh[7] = 1.092548 * x * z;
    sh[8] = 0.546274 * (x * x - y * y);
    return sh;
}

fn evaluate_probe_sh_prev(probe_idx: u32, dir: vec3f) -> vec3f {
    let base = probe_idx * SH_STRIDE;
    let sh = compute_sh_basis_probe(dir);
    var result = vec3f(0.0);
    for (var c = 0u; c < 9u; c++) {
        result += probe_sh_prev[base + c].xyz * sh[c];
    }
    return max(result, vec3f(0.0));
}

fn sample_probe_grid_prev(world_pos: vec3f, normal: vec3f) -> vec3f {
    let dims = params.grid_dims;
    let grid_step = vec3f(params.grid_step_x, params.grid_step_y, params.grid_step_z);
    let local_pos = (world_pos - params.grid_min) / grid_step;
    let base_coord = vec3i(floor(local_pos));
    let frac = local_pos - vec3f(base_coord);

    var total_irradiance = vec3f(0.0);
    var total_weight = 0.0;

    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let probe_coord = base_coord + vec3i(dx, dy, dz);
                let clamped = clamp(probe_coord, vec3i(0), vec3i(dims) - vec3i(1));
                let probe_idx = u32(clamped.z) * dims.x * dims.y + u32(clamped.y) * dims.x + u32(clamped.x);

                let t = vec3f(f32(dx), f32(dy), f32(dz));
                let trilinear = (1.0 - abs(t.x - frac.x)) * (1.0 - abs(t.y - frac.y)) * (1.0 - abs(t.z - frac.z));

                if (trilinear <= 0.0) { continue; }

                let irradiance = evaluate_probe_sh_prev(probe_idx, normal);
                total_irradiance += irradiance * trilinear;
                total_weight += trilinear;
            }
        }
    }

    if (total_weight > 0.0) {
        return total_irradiance / total_weight;
    }
    return vec3f(0.0);
}

// ── Simple direct lighting evaluation for probe rays ─────────────────────────
fn evaluate_direct_lighting(hit_pos: vec3f, hit_norm: vec3f) -> vec3f {
    var total = vec3f(0.0);
    for (var i = 0u; i < params.light_count; i++) {
        let light = scene_lights[i];
        var light_dir: vec3f;
        var dist: f32;
        var intensity: vec3f;

        if (light.light_type == LIGHT_DIRECTIONAL) {
            light_dir = normalize(-light.position);
            dist = 1e30;
            intensity = light.color * light.intensity;
        } else if (light.light_type == LIGHT_AREA) {
            let to_light = light.position - hit_pos;
            dist = length(to_light);
            light_dir = to_light / dist;
            let cos_light = max(dot(-light_dir, light.normal), 0.0);
            let area = light.extra.x * light.extra.y;
            intensity = light.color * light.intensity * cos_light * area / (dist * dist);
        } else {
            let to_light = light.position - hit_pos;
            dist = length(to_light);
            light_dir = to_light / dist;
            intensity = light.color * light.intensity / (dist * dist);
        }

        let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
        if (n_dot_l <= 0.0) { continue; }

        // Shadow test
        var shadow_ray: Ray;
        shadow_ray.origin = hit_pos + hit_norm * 0.001;
        shadow_ray.dir = light_dir;
        let shadow_hit = trace_bvh(shadow_ray);
        if (shadow_hit.hit && shadow_hit.t < dist - 0.01) { continue; }

        total += intensity * n_dot_l;
    }
    return total;
}

@compute @workgroup_size(64)
fn probe_trace_main(@builtin(global_invocation_id) gid: vec3u) {
    let total_probes = params.grid_dims.x * params.grid_dims.y * params.grid_dims.z;
    let total_rays = total_probes * params.rays_per_probe;
    let ray_idx = gid.x;
    if (ray_idx >= total_rays) { return; }

    let probe_idx = ray_idx / params.rays_per_probe;
    let local_ray = ray_idx % params.rays_per_probe;

    // Initialize RNG
    rng_state = pcg_hash(ray_idx + params.frame_index * 1000003u);

    let coord = probe_index_to_coord(probe_idx);
    let probe_pos = probe_world_pos(coord.x, coord.y, coord.z);

    // Randomized spherical Fibonacci: rotate base direction by frame-varying random
    let base_dir = spherical_fibonacci(local_ray, params.rays_per_probe);
    let angle = f32(params.frame_index) * GOLDEN_RATIO * 2.0 * 3.14159265;
    let ca = cos(angle);
    let sa = sin(angle);
    let rot_dir = vec3f(
        base_dir.x * ca - base_dir.z * sa,
        base_dir.y,
        base_dir.x * sa + base_dir.z * ca,
    );

    var ray: Ray;
    ray.origin = probe_pos;
    ray.dir = normalize(rot_dir);

    // Trace with transparency skip
    var radiance = vec3f(0.0);
    var distance = params.max_distance;
    var current_ray = ray;
    var bounces = 0u;

    loop {
        if (bounces >= 4u) { break; }
        let hit = trace_bvh(current_ray);
        if (!hit.hit) { break; }

        let mat = materials[hit.mat_index];

        // Probe-invisible or transmissive: skip through
        if (mat.max_bounces < 0.0 || mat.transmission >= 0.5) {
            current_ray.origin = hit.world_pos + current_ray.dir * 0.01;
            bounces++;
            continue;
        }

        distance = hit.t;

        // Emissive surfaces
        if (mat.emissive_intensity > 0.0) {
            radiance = mat.emissive * mat.emissive_intensity;
        }

        // Direct lighting at hit point (Lambertian)
        let direct = evaluate_direct_lighting(hit.world_pos, hit.world_norm);

        // Multi-bounce: sample previous frame's probe grid for indirect
        let indirect = sample_probe_grid_prev(hit.world_pos + hit.world_norm * 0.01, hit.world_norm);

        radiance += (direct + indirect) * mat.albedo;
        break;
    }

    // Store: [radiance.rgb, distance], [rayDir.xyz, 0]
    let out_base = ray_idx * 2u;
    ray_results[out_base + 0u] = vec4f(radiance, distance);
    ray_results[out_base + 1u] = vec4f(ray.dir, 0.0);
}
