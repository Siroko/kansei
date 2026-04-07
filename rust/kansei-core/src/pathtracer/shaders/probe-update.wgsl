// ── Probe SH update ──────────────────────────────────────────────────────────
//
// Projects ray results onto L2 spherical harmonics (9 coefficients per probe,
// each storing RGB as vec4f.xyz). Temporal hysteresis blend with previous SH.
//
// One thread per probe. Each thread loops over all its rays.

struct ProbeUpdateParams {
    rays_per_probe : u32,
    total_probes   : u32,
    hysteresis     : f32,
    frame_index    : u32,
}

@group(0) @binding(0) var<uniform> params            : ProbeUpdateParams;
@group(0) @binding(1) var<storage, read> ray_results  : array<vec4f>;
@group(0) @binding(2) var<storage, read> sh_history   : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> sh_output : array<vec4f>;

// ── L2 Spherical Harmonics ──────────────────────────────────────────────────
// 9 basis functions for order-2 SH
// Convention: Y_lm indexed as [0..8] = Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22
//
// Each probe stores 9 vec4f (xyz = RGB coefficients) = 144 bytes
// SH_STRIDE = 9

const SH_STRIDE = 9u;
const PI = 3.14159265;

fn compute_sh_basis(d: vec3f) -> array<f32, 9> {
    let x = d.x; let y = d.y; let z = d.z;
    var sh: array<f32, 9>;
    sh[0] = 0.282095;                      // Y00  = 1/(2*sqrt(pi))
    sh[1] = 0.488603 * y;                  // Y1-1 = sqrt(3/(4pi)) * y
    sh[2] = 0.488603 * z;                  // Y10  = sqrt(3/(4pi)) * z
    sh[3] = 0.488603 * x;                  // Y11  = sqrt(3/(4pi)) * x
    sh[4] = 1.092548 * x * y;              // Y2-2 = sqrt(15/(4pi)) * xy
    sh[5] = 1.092548 * y * z;              // Y2-1 = sqrt(15/(4pi)) * yz
    sh[6] = 0.315392 * (3.0 * z * z - 1.0); // Y20  = sqrt(5/(16pi)) * (3z^2-1)
    sh[7] = 1.092548 * x * z;              // Y21  = sqrt(15/(4pi)) * xz
    sh[8] = 0.546274 * (x * x - y * y);    // Y22  = sqrt(15/(16pi)) * (x^2-y^2)
    return sh;
}

@compute @workgroup_size(64)
fn probe_update_main(@builtin(global_invocation_id) gid: vec3u) {
    let probe_idx = gid.x;
    if (probe_idx >= params.total_probes) { return; }

    // Accumulate SH coefficients from ray results
    var sh_accum: array<vec3f, 9>;
    for (var c = 0u; c < 9u; c++) {
        sh_accum[c] = vec3f(0.0);
    }

    let ray_base = probe_idx * params.rays_per_probe;
    var valid_rays = 0.0;

    for (var r = 0u; r < params.rays_per_probe; r++) {
        let result_base = (ray_base + r) * 2u;
        let radiance_dist = ray_results[result_base + 0u];
        let dir_pad = ray_results[result_base + 1u];

        let radiance = radiance_dist.xyz;
        let ray_dir = dir_pad.xyz;

        // Project radiance onto SH basis
        let sh = compute_sh_basis(ray_dir);

        // Monte Carlo integration: weight = 4*PI / N (uniform sphere sampling)
        for (var c = 0u; c < 9u; c++) {
            sh_accum[c] += radiance * sh[c];
        }
        valid_rays += 1.0;
    }

    // Normalize: Monte Carlo estimate of integral over sphere = (4*PI / N) * sum
    let weight = 4.0 * PI / max(valid_rays, 1.0);

    let sh_base = probe_idx * SH_STRIDE;

    // Blend with history
    let alpha = select(params.hysteresis, 0.0, params.frame_index == 0u);

    for (var c = 0u; c < 9u; c++) {
        let new_coeff = sh_accum[c] * weight;
        let prev_coeff = sh_history[sh_base + c].xyz;
        let blended = mix(new_coeff, prev_coeff, alpha);
        sh_output[sh_base + c] = vec4f(blended, 0.0);
    }
}
