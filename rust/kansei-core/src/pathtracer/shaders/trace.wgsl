// ── Path tracing compute shader ─────────────────────────────────────────────
//
// Requires: intersection.wgsl (Ray, HitInfo, ray_aabb, ray_triangle)
//           traversal.wgsl   (Instance, trace_bvh, trace_bvh_shadow, get_world_tri_area,
//                              triangles, bvh4_nodes, tlas_bvh4_nodes, instances)
//
// This file is CONCATENATED after intersection.wgsl + traversal.wgsl at pipeline
// creation time. Do NOT re-declare Ray, HitInfo, Instance, or traversal functions.
//
// Features:
//   - Primary ray generation from inverse view-projection
//   - Multi-bounce path tracing with russian roulette
//   - GGX microfacet specular + cosine-weighted diffuse BSDF
//   - Fresnel-weighted lobe selection (specular vs diffuse)
//   - Next Event Estimation (NEE) with directional, area, and point lights
//   - Multiple Importance Sampling (MIS) for area lights + emissive geometry
//   - Refraction / transmission with Beer's law absorption
//   - Blue noise + PCG hybrid random number generation

// ── Structures ──────────────────────────────────────────────────────────────

struct MaterialData {
    albedo             : vec3f,
    roughness          : f32,
    metallic           : f32,
    ior                : f32,
    max_bounces        : f32,
    transmission       : f32,   // 0.0 = opaque, 1.0 = fully transmissive
    absorption_color   : vec3f,
    absorption_density : f32,
    emissive           : vec3f,
    emissive_intensity : f32,
}

struct TraceParams {
    inv_view_proj   : mat4x4f,  // bytes 0-63
    camera_pos      : vec3f,    // 64-75
    frame_index     : u32,      // 76-79
    width           : u32,      // 80-83
    height          : u32,      // 84-87
    light_count     : u32,      // 88-91
    spp             : u32,      // 92-95
    use_blue_noise  : u32,      // 96-99
    fixed_seed      : u32,      // 100-103
    max_bounces     : u32,      // 104-107
    _pad0           : u32,      // 108-111
    ambient_color   : vec3f,    // 112-123
    _pad1           : u32,      // 124-127
    _pad2           : vec4f,    // 128-143
    _pad3           : vec4f,    // 144-159
    _pad4           : vec4f,    // 160-175
    _pad5           : vec4f,    // 176-191
}

const LIGHT_DIRECTIONAL = 1u;
const LIGHT_AREA        = 2u;
const LIGHT_POINT       = 3u;

struct LightData {
    position   : vec3f,     // world pos (area/point) or direction (directional)
    light_type : u32,       // 1=directional, 2=area, 3=point
    color      : vec3f,
    intensity  : f32,
    normal     : vec3f,     // area light facing direction
    _pad       : f32,
    extra      : vec4f,     // area: (sizeX, sizeZ, 0, 0), point: (radius, 0, 0, 0)
}

// ── Bindings ────────────────────────────────────────────────────────────────
// Bindings 0-4 are declared here; bindings 5-8 are declared by traversal.wgsl
// (triangles, bvh4_nodes, tlas_bvh4_nodes, instances).

@group(0) @binding(0) var<uniform> params : TraceParams;
@group(0) @binding(1) var output_tex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read> materials   : array<MaterialData>;
@group(0) @binding(3) var<storage, read> scene_lights : array<LightData>;
@group(0) @binding(4) var<storage, read> blue_noise   : array<f32>;

// ── RNG: Blue noise + PCG hybrid sampler ────────────────────────────────────

const BN_SIZE      : u32 = 128u;
const GOLDEN_RATIO : f32 = 0.6180339887;
const PI           : f32 = 3.14159265;

var<private> _bn_pixel  : vec2u;
var<private> _bn_frame  : u32;
var<private> _bn_dim    : u32;
var<private> _rng_state : u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn init_sampler(pixel: vec2u, frame: u32, sample_idx: u32) {
    _bn_pixel = pixel;
    _bn_frame = select(frame * 16u + sample_idx, sample_idx, params.fixed_seed != 0u);
    _bn_dim = 0u;
    _rng_state = pcg_hash(pixel.x + pixel.y * 9999u + _bn_frame * 1000003u);
}

fn next_random() -> f32 {
    let dim = _bn_dim;
    _bn_dim += 1u;

    if (dim < 8u && params.use_blue_noise != 0u) {
        let px = (_bn_pixel.x + dim * 17u) % BN_SIZE;
        let py = (_bn_pixel.y + dim * 31u) % BN_SIZE;
        let bn = blue_noise[py * BN_SIZE + px];
        return fract(bn + f32(_bn_frame) * GOLDEN_RATIO);
    }

    // PCG fallback for higher dimensions
    _rng_state = pcg_hash(_rng_state);
    return f32(_rng_state) / 4294967295.0;
}

// ── Orthonormal basis ───────────────────────────────────────────────────────
// Robust ONB from normal (Duff et al. 2017, Frisvad 2012 revised).
// Branchless, no cross products, numerically stable for all directions.

fn build_onb(n: vec3f) -> array<vec3f, 2> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return array<vec3f, 2>(
        vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3f(b, s + n.y * n.y * a, -n.y),
    );
}

// ── Sampling functions ──────────────────────────────────────────────────────

fn cosine_sample_hemisphere(n: vec3f, r1: f32, r2: f32) -> vec3f {
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);

    let tb = build_onb(n);
    let tangent = tb[0];
    let bitangent = tb[1];

    return normalize(
        tangent * cos(phi) * sin_theta +
        bitangent * sin(phi) * sin_theta +
        n * cos_theta
    );
}

fn sample_ggx(n: vec3f, roughness: f32, r1: f32, r2: f32) -> vec3f {
    let a = roughness * roughness;
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let tb = build_onb(n);
    let tangent = tb[0];
    let bitangent = tb[1];

    let h = normalize(
        tangent * cos(phi) * sin_theta +
        bitangent * sin(phi) * sin_theta +
        n * cos_theta
    );
    return h;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ── MIS (Multiple Importance Sampling) helpers ──────────────────────────────

fn d_ggx(ndot_h: f32, a2: f32) -> f32 {
    let denom = ndot_h * ndot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    return a2 / max(a2 + pdf_b * pdf_b, 1e-10);
}

/// Combined BSDF pdf: specular GGX + diffuse cosine hemisphere
fn pdf_bsdf(n: vec3f, wo: vec3f, wi: vec3f, roughness: f32, metallic: f32, surf_albedo: vec3f) -> f32 {
    let cos_theta = max(dot(n, wi), 0.0);
    if (cos_theta <= 0.0) { return 0.0; }

    let ndot_v = max(dot(n, wo), 0.001);
    let f0 = mix(vec3f(0.04), surf_albedo, metallic);
    let f = fresnel_schlick(ndot_v, f0);
    let spec_avg = (f.r + f.g + f.b) / 3.0;
    let diff_w = (1.0 - spec_avg) * (1.0 - metallic);
    let total_w = spec_avg + diff_w;
    let spec_prob = clamp(spec_avg / max(total_w, 0.001), 0.05, 0.95);

    let pdf_diff = cos_theta / PI;

    let h = normalize(wo + wi);
    let ndot_h = max(dot(n, h), 0.0);
    let vdot_h = max(dot(wo, h), 0.001);
    let alpha = max(roughness, 0.02) * max(roughness, 0.02);
    let alpha2 = alpha * alpha;
    let pdf_spec = d_ggx(ndot_h, alpha2) * ndot_h / (4.0 * vdot_h);

    return spec_prob * pdf_spec + (1.0 - spec_prob) * pdf_diff;
}

// ── Shadow test that skips refractive (transparent) surfaces ────────────────

fn trace_shadow(origin: vec3f, dir: vec3f, max_dist: f32) -> bool {
    var ro = origin;
    var remaining = max_dist;
    for (var i = 0u; i < 4u; i++) {
        var sray: Ray;
        sray.origin = ro;
        sray.dir = dir;
        let hit = trace_bvh_internal(sray, false, remaining);
        if (!hit.hit) { return false; }
        let hit_mat = materials[hit.mat_index];
        if (hit_mat.transmission < 0.5) { return true; }
        // Transmissive surface: advance past it
        ro = ro + dir * (hit.t + 0.002);
        remaining -= hit.t + 0.002;
        if (remaining <= 0.0) { return false; }
    }
    return false;
}

// ── Per-light-type evaluation ───────────────────────────────────────────────

fn evaluate_directional_light(hit_pos: vec3f, hit_norm: vec3f, light: LightData) -> vec3f {
    let light_dir = normalize(-light.position); // position stores direction for directional
    let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
    if (n_dot_l <= 0.0) { return vec3f(0.0); }

    if (trace_shadow(hit_pos + hit_norm * 0.001, light_dir, 1e30)) { return vec3f(0.0); }

    return light.color * light.intensity * n_dot_l;
}

fn evaluate_area_light(hit_pos: vec3f, hit_norm: vec3f, light: LightData,
                       wo: vec3f, roughness: f32, metallic: f32, surf_albedo: vec3f) -> vec3f {
    let lr1 = next_random();
    let lr2 = next_random();
    let size_x = light.extra.x;
    let size_z = light.extra.y;
    let area = size_x * size_z;

    // Build tangent frame from light normal
    let ln = light.normal;
    let lup = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(ln.y) < 0.9);
    let lt = normalize(cross(lup, ln));
    let lb = cross(ln, lt);

    let light_pos = light.position
        + lt * (lr1 - 0.5) * size_x
        + lb * (lr2 - 0.5) * size_z;

    let to_light = light_pos - hit_pos;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let light_dir = to_light / dist;

    let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
    if (n_dot_l <= 0.0) { return vec3f(0.0); }

    let light_cos = max(dot(ln, -light_dir), 0.0);
    if (light_cos <= 0.0) { return vec3f(0.0); }

    if (trace_shadow(hit_pos + hit_norm * 0.001, light_dir, dist - 0.01)) { return vec3f(0.0); }

    // MIS: weight NEE against BSDF probability for this direction
    let pdf_light = dist2 / max(light_cos * area, 1e-8);
    let pdf_bsdf_val = pdf_bsdf(hit_norm, wo, light_dir, roughness, metallic, surf_albedo);
    let mis_w = power_heuristic(pdf_light, pdf_bsdf_val);

    return light.color * light.intensity * n_dot_l * light_cos * area / dist2 * mis_w;
}

fn evaluate_point_light(hit_pos: vec3f, hit_norm: vec3f, light: LightData) -> vec3f {
    let to_light = light.position - hit_pos;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let light_dir = to_light / dist;

    let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
    if (n_dot_l <= 0.0) { return vec3f(0.0); }

    if (trace_shadow(hit_pos + hit_norm * 0.001, light_dir, dist - 0.01)) { return vec3f(0.0); }

    return light.color * light.intensity * n_dot_l / dist2;
}

// ── Evaluate all scene lights at a hit point ────────────────────────────────

fn evaluate_lighting(hit_pos: vec3f, hit_norm: vec3f, wo: vec3f,
                     roughness: f32, metallic: f32, surf_albedo: vec3f) -> vec3f {
    var total = vec3f(0.0);
    let count = params.light_count;
    for (var i = 0u; i < count; i++) {
        let light = scene_lights[i];
        if (light.light_type == LIGHT_DIRECTIONAL) {
            total += evaluate_directional_light(hit_pos, hit_norm, light);
        } else if (light.light_type == LIGHT_AREA) {
            total += evaluate_area_light(hit_pos, hit_norm, light, wo, roughness, metallic, surf_albedo);
        } else if (light.light_type == LIGHT_POINT) {
            total += evaluate_point_light(hit_pos, hit_norm, light);
        }
    }
    return total;
}

// ── Multi-bounce path tracer with NEE ───────────────────────────────────────
// Returns irradiance at start_pos. At each vertex: NEE for direct light,
// then stochastic PBR bounce (specular or diffuse based on metallic/Fresnel).

fn trace_path(start_pos: vec3f, start_norm: vec3f, skip_first_nee: u32,
              start_wo: vec3f, start_roughness: f32, start_metallic: f32,
              start_albedo: vec3f) -> vec3f {
    var accumulated = vec3f(0.0);
    if (skip_first_nee == 0u) {
        accumulated = evaluate_lighting(start_pos, start_norm, start_wo,
                                        start_roughness, start_metallic, start_albedo);
    }
    var throughput = vec3f(1.0);
    var pos = start_pos;
    var norm = start_norm;
    var sample_spec = false;
    var spec_roughness = 1.0f;
    var incoming_dir = vec3f(0.0);
    // Track previous-vertex material for BSDF pdf at emissive hits
    var prev_wo = start_wo;
    var prev_roughness = start_roughness;
    var prev_metallic = start_metallic;
    var prev_albedo = start_albedo;
    let max_bounces = params.max_bounces;

    for (var bounce = 0u; bounce < max_bounces; bounce++) {
        var bounce_ray: Ray;
        bounce_ray.origin = pos + norm * 0.001;

        if (sample_spec) {
            let h = sample_ggx(norm, max(spec_roughness, 0.02), next_random(), next_random());
            bounce_ray.dir = reflect(incoming_dir, h);
            if (dot(bounce_ray.dir, norm) <= 0.0) { break; }
        } else {
            bounce_ray.dir = cosine_sample_hemisphere(norm, next_random(), next_random());
        }

        let hit = trace_bvh(bounce_ray);
        if (!hit.hit) {
            accumulated += throughput * params.ambient_color;
            break;
        }

        let mat = materials[hit.mat_index];

        // MIS: when BSDF hits emissive geometry, evaluate emission with MIS weight
        if (mat.emissive_intensity > 0.0) {
            let emission = mat.emissive * mat.emissive_intensity;
            let hit_cos = abs(dot(hit.world_norm, -bounce_ray.dir));
            let dist2 = hit.t * hit.t;
            let tri_area = get_world_tri_area(hit.tri_index, hit.instance_id);
            let pdf_l = dist2 / max(hit_cos * tri_area, 1e-8);
            let pdf_b = pdf_bsdf(norm, prev_wo, bounce_ray.dir, prev_roughness, prev_metallic, prev_albedo);
            let w = power_heuristic(pdf_b, pdf_l);
            accumulated += throughput * emission * w;
        }

        if (mat.emissive_intensity > 0.0 || mat.transmission > 0.5) { break; }

        // PBR: decide next bounce from this surface
        let cos_theta = abs(dot(hit.world_norm, -bounce_ray.dir));
        let f0 = mix(vec3f(0.04), mat.albedo, mat.metallic);
        let f = fresnel_schlick(cos_theta, f0);
        let spec_avg = (f.r + f.g + f.b) / 3.0;
        let diff_w = (1.0 - spec_avg) * (1.0 - mat.metallic);
        let total_w = spec_avg + diff_w;
        let spec_prob = clamp(spec_avg / max(total_w, 0.001), 0.05, 0.95);

        let hit_wo = -bounce_ray.dir;

        if (next_random() < spec_prob) {
            throughput *= f / spec_prob;
            sample_spec = true;
            spec_roughness = mat.roughness;
        } else {
            throughput *= mat.albedo * (vec3f(1.0) - f) * (1.0 - mat.metallic) / (1.0 - spec_prob);
            sample_spec = false;
        }

        accumulated += throughput * evaluate_lighting(hit.world_pos, hit.world_norm, hit_wo,
                                                       mat.roughness, mat.metallic, mat.albedo);

        // Russian roulette after first bounce
        if (bounce > 0u) {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if (p < 0.01 || next_random() > p) { break; }
            throughput /= p;
        }

        incoming_dir = bounce_ray.dir;
        pos = hit.world_pos;
        norm = hit.world_norm;
        prev_wo = hit_wo;
        prev_roughness = mat.roughness;
        prev_metallic = mat.metallic;
        prev_albedo = mat.albedo;
    }

    return accumulated;
}

// ── Refraction tracing ──────────────────────────────────────────────────────

fn trace_refraction(in_ray: Ray, first_hit: HitInfo, mat: MaterialData) -> vec3f {
    var throughput = vec3f(1.0);
    var ray = in_ray;
    var current_hit = first_hit;
    let max_bounces_r = u32(abs(mat.max_bounces));
    var inside_medium = false;

    for (var bounce = 0u; bounce < max_bounces_r; bounce++) {
        let n = current_hit.world_norm;
        // Geometric entering: ensures face_norm always faces toward incoming ray
        let geom_entering = dot(ray.dir, n) < 0.0;
        let face_norm = select(-n, n, geom_entering);
        // Medium tracking for eta
        let media_entering = !inside_medium;
        let eta = select(mat.ior, 1.0 / mat.ior, media_entering);

        // Fresnel at each interface
        let cos_i = abs(dot(ray.dir, face_norm));
        let f0_scalar = pow((1.0 - mat.ior) / (1.0 + mat.ior), 2.0);
        let fresnel_val = f0_scalar + (1.0 - f0_scalar) * pow(1.0 - cos_i, 5.0);

        let refracted = refract(ray.dir, face_norm, eta);
        let tir = length(refracted) < 0.001;

        if (tir || next_random() < fresnel_val) {
            // Total internal reflection or Fresnel reflection
            let rh = sample_ggx(face_norm, max(mat.roughness, 0.02), next_random(), next_random());
            ray.dir = reflect(ray.dir, rh);
            if (dot(ray.dir, face_norm) <= 0.0) {
                ray.dir = reflect(in_ray.dir, face_norm); // fallback to perfect reflect
            }
            ray.origin = current_hit.world_pos + face_norm * 0.01;
        } else {
            // Refraction: toggle medium state
            inside_medium = !inside_medium;
            if (mat.roughness > 0.01) {
                let rh = sample_ggx(face_norm, mat.roughness, next_random(), next_random());
                let perturbed_refract = refract(ray.dir, rh, eta);
                if (length(perturbed_refract) > 0.001) {
                    ray.dir = perturbed_refract;
                } else {
                    ray.dir = refracted;
                }
            } else {
                ray.dir = refracted;
            }
            ray.origin = current_hit.world_pos - face_norm * 0.01;
        }

        let next_hit = trace_bvh(ray);
        if (!next_hit.hit) {
            return throughput * params.ambient_color;
        }

        let dist = next_hit.t;
        if (inside_medium) {
            throughput *= exp(-mat.absorption_color * mat.absorption_density * dist);
        }

        let next_mat = materials[next_hit.mat_index];
        if (next_mat.transmission < 0.5) {
            // Exited transmissive medium -- PBR shading at the exit surface
            let exit_norm = next_hit.world_norm;
            let exit_cos = abs(dot(ray.dir, exit_norm));
            let exit_f0 = mix(vec3f(0.04), next_mat.albedo, next_mat.metallic);
            let exit_f = fresnel_schlick(exit_cos, exit_f0);
            let exit_spec_avg = (exit_f.r + exit_f.g + exit_f.b) / 3.0;
            let exit_diff_w = (1.0 - exit_spec_avg) * (1.0 - next_mat.metallic);
            let exit_total_w = exit_spec_avg + exit_diff_w;
            let exit_spec_prob = clamp(exit_spec_avg / max(exit_total_w, 0.001), 0.05, 0.95);

            if (next_random() < exit_spec_prob) {
                // Specular reflection off the exit surface (GGX)
                var m_ray: Ray;
                let m_h = sample_ggx(exit_norm, max(next_mat.roughness, 0.02), next_random(), next_random());
                m_ray.dir = reflect(ray.dir, m_h);
                if (dot(m_ray.dir, exit_norm) <= 0.0) {
                    return throughput * trace_path(next_hit.world_pos, exit_norm, 0u,
                        -ray.dir, next_mat.roughness, next_mat.metallic, next_mat.albedo) * next_mat.albedo;
                }
                m_ray.origin = next_hit.world_pos + exit_norm * 0.001;
                var m_hit = trace_bvh(m_ray);
                // Follow specular bounces
                for (var mb = 0u; mb < 4u; mb++) {
                    if (!m_hit.hit) { break; }
                    let mb_mat = materials[m_hit.mat_index];
                    let mb_cos = abs(dot(m_ray.dir, m_hit.world_norm));
                    let mb_f0 = mix(vec3f(0.04), mb_mat.albedo, mb_mat.metallic);
                    let mb_spec_avg = (fresnel_schlick(mb_cos, mb_f0).r + fresnel_schlick(mb_cos, mb_f0).g + fresnel_schlick(mb_cos, mb_f0).b) / 3.0;
                    if (mb_spec_avg > 0.3 && mb_mat.transmission < 0.5) {
                        let mb_h = sample_ggx(m_hit.world_norm, max(mb_mat.roughness, 0.02), next_random(), next_random());
                        m_ray.dir = reflect(m_ray.dir, mb_h);
                        if (dot(m_ray.dir, m_hit.world_norm) <= 0.0) { break; }
                        m_ray.origin = m_hit.world_pos + m_hit.world_norm * 0.001;
                        m_hit = trace_bvh(m_ray);
                    } else {
                        break;
                    }
                }
                if (m_hit.hit) {
                    let m_mat = materials[m_hit.mat_index];
                    return throughput * exit_f / exit_spec_prob * trace_path(m_hit.world_pos, m_hit.world_norm, 0u,
                        -m_ray.dir, m_mat.roughness, m_mat.metallic, m_mat.albedo) * m_mat.albedo;
                }
                return throughput * exit_f / exit_spec_prob * params.ambient_color;
            } else {
                // Diffuse at exit surface
                let exit_kd = (vec3f(1.0) - exit_f) * (1.0 - next_mat.metallic);
                return throughput * trace_path(next_hit.world_pos, exit_norm, 0u,
                    -ray.dir, next_mat.roughness, next_mat.metallic, next_mat.albedo) * next_mat.albedo * exit_kd / (1.0 - exit_spec_prob);
            }
        }
        current_hit = next_hit;
    }
    _ = next_random();
    return throughput * params.ambient_color;
}

// ── Main compute entry point ────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn trace_main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.x, gid.y);

    // UV in [0,1] with half-pixel offset for pixel center
    let uv = (vec2f(coord) + 0.5) / vec2f(f32(params.width), f32(params.height));

    // Reconstruct world-space ray from inverse view-projection
    let ndc_near = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0, 1.0);
    let ndc_far  = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let near_wp = params.inv_view_proj * ndc_near;
    let far_wp  = params.inv_view_proj * ndc_far;
    let near_pos = near_wp.xyz / near_wp.w;
    let far_pos  = far_wp.xyz / far_wp.w;

    let spp = params.spp;
    var accumulated = vec3f(0.0);

    for (var s = 0u; s < spp; s++) {
        init_sampler(coord, params.frame_index, s);

        var primary_ray: Ray;
        primary_ray.origin = params.camera_pos;
        primary_ray.dir = normalize(far_pos - near_pos);

        let primary_hit = trace_bvh(primary_ray);

        if (!primary_hit.hit) {
            accumulated += params.ambient_color;
            continue;
        }

        let mat = materials[primary_hit.mat_index];

        // Add emissive contribution directly
        if (mat.emissive_intensity > 0.0) {
            accumulated += mat.emissive * mat.emissive_intensity;
            continue;
        }

        // Handle transmissive surfaces via dedicated refraction tracing
        if (mat.transmission > 0.5) {
            accumulated += trace_refraction(primary_ray, primary_hit, mat);
            continue;
        }

        let view_dir = -primary_ray.dir;
        let hit_norm = primary_hit.world_norm;
        let hit_pos = primary_hit.world_pos;
        let ndot_v = max(dot(hit_norm, view_dir), 0.001);
        let f0 = mix(vec3f(0.04), mat.albedo, mat.metallic);
        let f = fresnel_schlick(ndot_v, f0);
        let spec_avg = (f.r + f.g + f.b) / 3.0;

        // Energy partition: specular vs diffuse
        let spec_w = spec_avg;
        let diff_w = (1.0 - spec_avg) * (1.0 - mat.metallic);
        let total_w = spec_w + diff_w;
        let spec_prob = clamp(spec_w / max(total_w, 0.001), 0.05, 0.95);

        let r = next_random();
        if (r < spec_prob) {
            // -- Specular reflection (GGX) --
            let half_vec = sample_ggx(hit_norm, max(mat.roughness, 0.02), next_random(), next_random());
            let spec_dir = reflect(-view_dir, half_vec);

            if (dot(spec_dir, hit_norm) > 0.0) {
                var spec_ray: Ray;
                spec_ray.origin = hit_pos + hit_norm * 0.001;
                spec_ray.dir = spec_dir;
                var spec_hit = trace_bvh(spec_ray);

                // Follow metallic bounces
                for (var sk = 0u; sk < 4u; sk++) {
                    if (!spec_hit.hit) { break; }
                    let sk_mat = materials[spec_hit.mat_index];
                    if (sk_mat.metallic > 0.5 && sk_mat.transmission < 0.5) {
                        let h2 = sample_ggx(spec_hit.world_norm, max(sk_mat.roughness, 0.02), next_random(), next_random());
                        spec_ray.dir = reflect(spec_ray.dir, h2);
                        if (dot(spec_ray.dir, spec_hit.world_norm) <= 0.0) { break; }
                        spec_ray.origin = spec_hit.world_pos + spec_hit.world_norm * 0.001;
                        spec_hit = trace_bvh(spec_ray);
                    } else { break; }
                }

                if (spec_hit.hit) {
                    let spec_mat = materials[spec_hit.mat_index];
                    if (spec_mat.transmission > 0.5) {
                        accumulated += trace_refraction(spec_ray, spec_hit, spec_mat) * f / spec_prob;
                    } else {
                        accumulated += trace_path(spec_hit.world_pos, spec_hit.world_norm, 0u,
                            -spec_ray.dir, spec_mat.roughness, spec_mat.metallic, spec_mat.albedo) * spec_mat.albedo * f / spec_prob;
                    }
                } else {
                    accumulated += params.ambient_color * f / spec_prob;
                }
            }
        } else {
            // -- Diffuse: multi-bounce path tracing with NEE --
            let diff_prob = 1.0 - spec_prob;
            let kd = (vec3f(1.0) - f) * (1.0 - mat.metallic);
            accumulated += trace_path(hit_pos, hit_norm, 0u,
                view_dir, mat.roughness, mat.metallic, mat.albedo) * mat.albedo * kd / max(diff_prob, 0.01);
        }
    }

    textureStore(output_tex, coord, vec4f(accumulated / f32(spp), 1.0));
}
