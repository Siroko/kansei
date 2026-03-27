export const probeUpdateShader = /* wgsl */`
struct ProbeUpdateParams {
    raysPerProbe : u32,
    totalProbes  : u32,
    hysteresis   : f32,
    frameIndex   : u32,
}

@group(0) @binding(0) var<uniform> params : ProbeUpdateParams;
@group(0) @binding(1) var<storage, read> rayResults  : array<vec4f>;
@group(0) @binding(2) var<storage, read> shHistory   : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> shOutput : array<vec4f>;

// ── L2 Spherical Harmonics ──────────────────────────────────────────
// 9 basis functions for order-2 SH
// Convention: Y_lm indexed as [0..8] = Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22
//
// Each probe stores 9 vec4f (xyz = RGB coefficients) = 144 bytes
// SH_STRIDE = 9

const SH_STRIDE = 9u;
const PI = 3.14159265;

fn computeSHBasis(d: vec3f) -> array<f32, 9> {
    // Real spherical harmonics basis functions (orthonormal)
    let x = d.x; let y = d.y; let z = d.z;
    var sh: array<f32, 9>;
    sh[0] = 0.282095;                    // Y00  = 1/(2*sqrt(pi))
    sh[1] = 0.488603 * y;                // Y1-1 = sqrt(3/(4pi)) * y
    sh[2] = 0.488603 * z;                // Y10  = sqrt(3/(4pi)) * z
    sh[3] = 0.488603 * x;                // Y11  = sqrt(3/(4pi)) * x
    sh[4] = 1.092548 * x * y;            // Y2-2 = sqrt(15/(4pi)) * xy
    sh[5] = 1.092548 * y * z;            // Y2-1 = sqrt(15/(4pi)) * yz
    sh[6] = 0.315392 * (3.0*z*z - 1.0);  // Y20  = sqrt(5/(16pi)) * (3z²-1)
    sh[7] = 1.092548 * x * z;            // Y21  = sqrt(15/(4pi)) * xz
    sh[8] = 0.546274 * (x*x - y*y);      // Y22  = sqrt(15/(16pi)) * (x²-y²)
    return sh;
}

// One thread per probe. Each thread loops over all rays and projects onto SH.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let probeIdx = gid.x;
    if (probeIdx >= params.totalProbes) { return; }

    // Accumulate SH coefficients from ray results
    var shAccum: array<vec3f, 9>;
    for (var c = 0u; c < 9u; c++) {
        shAccum[c] = vec3f(0.0);
    }

    let rayBase = probeIdx * params.raysPerProbe;
    var validRays = 0.0;

    for (var r = 0u; r < params.raysPerProbe; r++) {
        let resultBase = (rayBase + r) * 2u;
        let radianceDist = rayResults[resultBase + 0u];
        let dirPad = rayResults[resultBase + 1u];

        let radiance = radianceDist.xyz;
        let rayDir = dirPad.xyz;

        // Project radiance onto SH basis
        let sh = computeSHBasis(rayDir);

        // Monte Carlo integration: weight = 4*PI / N (uniform sphere sampling)
        for (var c = 0u; c < 9u; c++) {
            shAccum[c] += radiance * sh[c];
        }
        validRays += 1.0;
    }

    // Normalize: Monte Carlo estimate of integral over sphere = (4*PI / N) * sum
    let weight = 4.0 * PI / max(validRays, 1.0);

    let shBase = probeIdx * SH_STRIDE;

    // Blend with history
    let alpha = select(params.hysteresis, 0.0, params.frameIndex == 0u);

    for (var c = 0u; c < 9u; c++) {
        let newCoeff = shAccum[c] * weight;
        let prevCoeff = shHistory[shBase + c].xyz;
        let blended = mix(newCoeff, prevCoeff, alpha);
        shOutput[shBase + c] = vec4f(blended, 0.0);
    }
}
`;
