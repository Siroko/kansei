// ── ReSTIR DI ────────────────────────────────────────────────────────────────
//
// Two entry points compiled into separate pipelines:
//   restir_generate — initial reservoir sampling + temporal reuse
//   restir_spatial  — spatial reuse from neighbor reservoirs + final shading
//
// Requires: intersection.wgsl, traversal.wgsl concatenated before this file,
//           plus BVH storage binding preamble (triangles, bvh4_nodes,
//           tlas_bvh4_nodes, instances).

// ── Reservoir layout ─────────────────────────────────────────────────────────
// 3 vec4f per pixel in a storage buffer:
//   [pixIdx*3 + 0]: xyz = sample position on light, w = bitcast<f32>(lightIdx)
//   [pixIdx*3 + 1]: xyz = sample normal,             w = targetPdf
//   [pixIdx*3 + 2]: x = W, y = wSum, z = bitcast<f32>(M), w = unused

const RESERVOIR_STRIDE = 3u;

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

struct ReSTIRParams {
    inv_view_proj  : mat4x4f,
    prev_view_proj : mat4x4f,
    camera_pos     : vec3f,
    frame_index    : u32,
    width          : u32,
    height         : u32,
    light_count    : u32,
    max_history    : u32,
}

// ── Bindings (generate pass) ─────────────────────────────────────────────────
// BVH bindings (4-8) are injected via the binding preamble before this file.

@group(0) @binding(0)  var depth_tex        : texture_depth_2d;
@group(0) @binding(1)  var normal_tex       : texture_2d<f32>;
@group(0) @binding(2)  var<uniform> params  : ReSTIRParams;
@group(0) @binding(3)  var<storage, read> scene_lights    : array<LightData>;
@group(0) @binding(9)  var<storage, read> materials       : array<MaterialData>;
@group(0) @binding(10) var<storage, read> reservoir_prev  : array<vec4f>;
@group(0) @binding(11) var<storage, read_write> reservoir_cur : array<vec4f>;
// binding 12: direct_light_out (only used in spatial pass, declared below)

// ── PCG RNG ──────────────────────────────────────────────────────────────────
var<private> _rng_state: u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn init_rng(pixel: vec2u, frame: u32) {
    _rng_state = pcg_hash(pixel.x + pixel.y * 9999u + frame * 1000003u);
}

fn next_random() -> f32 {
    _rng_state = pcg_hash(_rng_state);
    return f32(_rng_state) / 4294967295.0;
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ── Shadow test (skips transmissive surfaces) ────────────────────────────────
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
        ro = ro + dir * (hit.t + 0.002);
        remaining -= hit.t + 0.002;
        if (remaining <= 0.0) { return false; }
    }
    return false;
}

// ── Target PDF: unshadowed contribution estimate ─────────────────────────────
fn eval_target_pdf(hit_pos: vec3f, hit_norm: vec3f, sample_pos: vec3f, sample_norm: vec3f,
                   light_color: vec3f, light_intensity: f32, light_type: u32) -> f32 {
    if (light_type == LIGHT_DIRECTIONAL) {
        let light_dir = normalize(-sample_pos);
        let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
        return luminance(light_color * light_intensity * n_dot_l);
    }

    let to_light = sample_pos - hit_pos;
    let dist2 = dot(to_light, to_light);
    if (dist2 < 1e-8) { return 0.0; }
    let dist = sqrt(dist2);
    let light_dir = to_light / dist;
    let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
    if (n_dot_l <= 0.0) { return 0.0; }

    if (light_type == LIGHT_POINT) {
        return luminance(light_color * light_intensity * n_dot_l / dist2);
    }

    // Area light
    let light_cos = max(dot(sample_norm, -light_dir), 0.0);
    if (light_cos <= 0.0) { return 0.0; }
    return luminance(light_color * light_intensity * n_dot_l * light_cos / dist2);
}

fn write_reservoir(pix_idx: u32, pos: vec3f, norm: vec3f, light_idx: u32,
                   target_pdf: f32, w_capital: f32, w_sum: f32, m: u32) {
    let base = pix_idx * RESERVOIR_STRIDE;
    reservoir_cur[base + 0u] = vec4f(pos, bitcast<f32>(light_idx));
    reservoir_cur[base + 1u] = vec4f(norm, target_pdf);
    reservoir_cur[base + 2u] = vec4f(w_capital, w_sum, bitcast<f32>(m), 0.0);
}

// ── Full contribution evaluation (used in spatial pass) ──────────────────────
fn eval_contribution(hit_pos: vec3f, hit_norm: vec3f, sample_pos: vec3f, sample_norm: vec3f,
                     light_color: vec3f, light_intensity: f32, light_type: u32) -> vec3f {
    if (light_type == LIGHT_DIRECTIONAL) {
        let light_dir = normalize(-sample_pos);
        let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
        return light_color * light_intensity * n_dot_l;
    }
    let to_light = sample_pos - hit_pos;
    let dist2 = dot(to_light, to_light);
    if (dist2 < 1e-8) { return vec3f(0.0); }
    let dist = sqrt(dist2);
    let light_dir = to_light / dist;
    let n_dot_l = max(dot(hit_norm, light_dir), 0.0);
    if (n_dot_l <= 0.0) { return vec3f(0.0); }
    if (light_type == LIGHT_POINT) {
        return light_color * light_intensity * n_dot_l / dist2;
    }
    let light_cos = max(dot(sample_norm, -light_dir), 0.0);
    if (light_cos <= 0.0) { return vec3f(0.0); }
    return light_color * light_intensity * n_dot_l * light_cos / dist2;
}

// ═════════════════════════════════════════════════════════════════════════════
// Entry point 1: restir_generate
// ═════════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(8, 8)
fn restir_generate(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let pix_idx = coord.y * params.width + coord.x;
    init_rng(coord, params.frame_index);

    // Reconstruct world position from GBuffer
    let trace_size = vec2f(f32(params.width), f32(params.height));
    let uv = (vec2f(coord) + 0.5) / trace_size;
    let gbuf_dim = vec2u(textureDimensions(depth_tex));
    let gbuf_coord = min(vec2u(vec2f(gbuf_dim) * uv), gbuf_dim - vec2u(1u));
    let depth = textureLoad(depth_tex, gbuf_coord, 0);

    // Sky - write empty reservoir
    if (depth >= 1.0) {
        write_reservoir(pix_idx, vec3f(0.0), vec3f(0.0), 0u, 0.0, 0.0, 0.0, 0u);
        return;
    }

    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.inv_view_proj * ndc;
    let world_pos = wp.xyz / wp.w;
    let world_normal = normalize(textureLoad(normal_tex, vec2i(gbuf_coord), 0).xyz * 2.0 - 1.0);

    // ── Generate initial reservoir: stream all lights ──
    var r_pos = vec3f(0.0);
    var r_norm = vec3f(0.0);
    var r_light_idx = 0u;
    var r_target_pdf = 0.0;
    var r_w_sum = 0.0;
    var r_m = 0u;

    let light_count = params.light_count;
    for (var li = 0u; li < light_count; li++) {
        let light = scene_lights[li];
        var s_pos: vec3f;
        var s_norm: vec3f;
        var source_pdf: f32;

        if (light.light_type == LIGHT_AREA) {
            let lr1 = next_random();
            let lr2 = next_random();
            let size_x = light.extra.x;
            let size_z = light.extra.y;
            let ln = light.normal;
            let lup = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(ln.y) < 0.9);
            let lt = normalize(cross(lup, ln));
            let lb = cross(ln, lt);
            s_pos = light.position + lt * (lr1 - 0.5) * size_x + lb * (lr2 - 0.5) * size_z;
            s_norm = ln;
            source_pdf = 1.0 / (size_x * size_z * f32(light_count));
        } else if (light.light_type == LIGHT_POINT) {
            s_pos = light.position;
            s_norm = vec3f(0.0, 1.0, 0.0);
            source_pdf = 1.0 / f32(light_count);
        } else {
            // Directional
            s_pos = light.position;
            s_norm = vec3f(0.0, 1.0, 0.0);
            source_pdf = 1.0 / f32(light_count);
        }

        let p_hat = eval_target_pdf(world_pos, world_normal, s_pos, s_norm,
                                     light.color, light.intensity, light.light_type);
        let w = p_hat / max(source_pdf, 1e-20);
        r_w_sum += w;
        r_m += 1u;

        if (next_random() * r_w_sum < w) {
            r_pos = s_pos;
            r_norm = s_norm;
            r_light_idx = li;
            r_target_pdf = p_hat;
        }
    }

    // Compute W for initial reservoir
    var r_w = 0.0;
    if (r_target_pdf > 0.0 && r_m > 0u) {
        r_w = r_w_sum / (f32(r_m) * r_target_pdf);
    }

    // ── Temporal reuse: merge with previous frame's reservoir ──
    if (params.frame_index > 0u) {
        let prev_clip = params.prev_view_proj * vec4f(world_pos, 1.0);
        let prev_ndc = prev_clip.xyz / prev_clip.w;
        let prev_uv = vec2f(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));

        if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
            let prev_pix_coord = vec2u(trace_size * prev_uv);
            let prev_pix_idx = min(prev_pix_coord.y * params.width + prev_pix_coord.x,
                                   params.width * params.height - 1u);

            let p_base = prev_pix_idx * RESERVOIR_STRIDE;
            let pr0 = reservoir_prev[p_base + 0u];
            let pr1 = reservoir_prev[p_base + 1u];
            let pr2 = reservoir_prev[p_base + 2u];
            let p_pos = pr0.xyz;
            let p_norm = pr1.xyz;
            let p_light_idx = bitcast<u32>(pr0.w);
            let p_w = pr2.x;
            let p_m = bitcast<u32>(pr2.z);

            if (p_m > 0u && p_light_idx < light_count) {
                let capped_m = min(p_m, params.max_history);

                let prev_light = scene_lights[p_light_idx];
                let p_hat_at_cur = eval_target_pdf(world_pos, world_normal, p_pos, p_norm,
                                                    prev_light.color, prev_light.intensity, prev_light.light_type);

                let merge_w = p_hat_at_cur * p_w * f32(capped_m);
                r_w_sum += merge_w;
                r_m += capped_m;

                if (next_random() * r_w_sum < merge_w) {
                    r_pos = p_pos;
                    r_norm = p_norm;
                    r_light_idx = p_light_idx;
                    r_target_pdf = p_hat_at_cur;
                }

                if (r_target_pdf > 0.0 && r_m > 0u) {
                    r_w = r_w_sum / (f32(r_m) * r_target_pdf);
                }
            }
        }
    }

    write_reservoir(pix_idx, r_pos, r_norm, r_light_idx, r_target_pdf, r_w, r_w_sum, r_m);
}

// ═════════════════════════════════════════════════════════════════════════════
// Entry point 2: restir_spatial
// ═════════════════════════════════════════════════════════════════════════════
// NOTE: This entry point uses the same bindings as generate, but replaces
// reservoir_cur with a read-only reservoirIn, and adds the output texture.
// The Rust side builds a separate pipeline + bind group for this pass.
// In this single-file approach, both entry points share declarations.
// The spatial pass reads from reservoir_cur (which was just written by generate)
// and writes to direct_light_out.

@group(0) @binding(12) var direct_light_out : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn restir_spatial(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let pix_idx = coord.y * params.width + coord.x;
    // Offset RNG seed for spatial pass to avoid correlation with generate
    _rng_state = pcg_hash(coord.x + coord.y * 9999u + (params.frame_index + 7777u) * 1000003u);

    let trace_size = vec2f(f32(params.width), f32(params.height));
    let uv = (vec2f(coord) + 0.5) / trace_size;
    let gbuf_dim = vec2u(textureDimensions(depth_tex));
    let gbuf_coord = min(vec2u(vec2f(gbuf_dim) * uv), gbuf_dim - vec2u(1u));
    let depth = textureLoad(depth_tex, gbuf_coord, 0);

    if (depth >= 1.0) {
        textureStore(direct_light_out, coord, vec4f(0.0));
        return;
    }

    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.inv_view_proj * ndc;
    let world_pos = wp.xyz / wp.w;
    let world_normal = normalize(textureLoad(normal_tex, vec2i(gbuf_coord), 0).xyz * 2.0 - 1.0);

    // Read current reservoir (post-temporal, from generate pass)
    let base = pix_idx * RESERVOIR_STRIDE;
    let r0 = reservoir_cur[base + 0u];
    let r1 = reservoir_cur[base + 1u];
    let r2 = reservoir_cur[base + 2u];
    var r_pos = r0.xyz;
    var r_norm = r1.xyz;
    var r_light_idx = bitcast<u32>(r0.w);
    var r_target_pdf = r1.w;
    var r_w = r2.x;
    var r_w_sum = r2.y;
    var r_m = bitcast<u32>(r2.z);

    // ── Spatial reuse: merge with 5 random neighbors ──
    let spatial_radius = 10.0;
    let spatial_count = 5u;

    for (var si = 0u; si < spatial_count; si++) {
        let angle = next_random() * 6.2831853;
        let radius = next_random() * spatial_radius;
        let offset = vec2i(vec2f(cos(angle), sin(angle)) * radius);
        let neighbor_coord = vec2i(coord) + offset;

        if (neighbor_coord.x < 0 || neighbor_coord.x >= i32(params.width) ||
            neighbor_coord.y < 0 || neighbor_coord.y >= i32(params.height)) {
            continue;
        }

        // Validate neighbor similarity (depth + normal)
        let n_uv = (vec2f(vec2u(neighbor_coord)) + 0.5) / trace_size;
        let n_gbuf = min(vec2u(vec2f(gbuf_dim) * n_uv), gbuf_dim - vec2u(1u));
        let n_depth = textureLoad(depth_tex, n_gbuf, 0);
        if (abs(depth - n_depth) > 0.1) { continue; }
        let n_normal = normalize(textureLoad(normal_tex, vec2i(n_gbuf), 0).xyz * 2.0 - 1.0);
        if (dot(world_normal, n_normal) < 0.9) { continue; }

        let n_pix_idx = u32(neighbor_coord.y) * params.width + u32(neighbor_coord.x);
        let n_base = n_pix_idx * RESERVOIR_STRIDE;
        let nr0 = reservoir_cur[n_base + 0u];
        let nr1 = reservoir_cur[n_base + 1u];
        let nr2 = reservoir_cur[n_base + 2u];
        let n_pos = nr0.xyz;
        let n_norm2 = nr1.xyz;
        let n_light_idx = bitcast<u32>(nr0.w);
        let n_w = nr2.x;
        let n_m = min(bitcast<u32>(nr2.z), params.max_history);

        if (n_m == 0u || n_light_idx >= params.light_count) { continue; }

        let n_light = scene_lights[n_light_idx];
        let p_hat_at_me = eval_target_pdf(world_pos, world_normal, n_pos, n_norm2,
                                           n_light.color, n_light.intensity, n_light.light_type);

        let merge_w = p_hat_at_me * n_w * f32(n_m);
        r_w_sum += merge_w;
        r_m += n_m;

        if (next_random() * r_w_sum < merge_w) {
            r_pos = n_pos;
            r_norm = n_norm2;
            r_light_idx = n_light_idx;
            r_target_pdf = p_hat_at_me;
        }
    }

    // Final W
    if (r_target_pdf > 0.0 && r_m > 0u) {
        r_w = r_w_sum / (f32(r_m) * r_target_pdf);
    } else {
        r_w = 0.0;
    }

    // ── Final shading: visibility test + contribution ──
    var direct_light = vec3f(0.0);
    if (r_target_pdf > 0.0 && r_light_idx < params.light_count) {
        let light = scene_lights[r_light_idx];
        var light_dir: vec3f;
        var shadow_dist: f32;

        if (light.light_type == LIGHT_DIRECTIONAL) {
            light_dir = normalize(-r_pos);
            shadow_dist = 1e30;
        } else {
            let to_light = r_pos - world_pos;
            let dist = length(to_light);
            light_dir = to_light / dist;
            shadow_dist = dist - 0.01;
        }

        if (!trace_shadow(world_pos + world_normal * 0.001, light_dir, shadow_dist)) {
            direct_light = eval_contribution(world_pos, world_normal, r_pos, r_norm,
                                              light.color, light.intensity, light.light_type) * r_w;
        }
    }

    textureStore(direct_light_out, coord, vec4f(direct_light, 1.0));
}
