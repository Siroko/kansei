export const shaderCode = /* wgsl */`

#include <fbm>

/**
 * Modulo 289 operation for a vec4.
 * @param x The input vector.
 * @return The result of the modulo operation.
 */
fn mod289_vec4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

/**
 * Modulo 289 operation for a float.
 * @param x The input float.
 * @return The result of the modulo operation.
 */
fn mod289_f32(x: f32) -> f32 {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

/**
 * Permute operation for a vec4.
 * @param x The input vector.
 * @return The permuted vector.
 */
fn permute_vec4(x: vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((x * 34.0) + 1.0) * x);
}

/**
 * Permute operation for a float.
 * @param x The input float.
 * @return The permuted float.
 */
fn permute_f32(x: f32) -> f32 {
    return mod289_f32(((x * 34.0) + 1.0) * x);
}

/**
 * Taylor inverse square root approximation for a vec4.
 * @param r The input vector.
 * @return The approximated inverse square root.
 */
fn taylorInvSqrt_vec4(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

/**
 * Taylor inverse square root approximation for a float.
 * @param r The input float.
 * @return The approximated inverse square root.
 */
fn taylorInvSqrt_f32(r: f32) -> f32 {
    return 1.79284291400159 - 0.85373472095314 * r;
}

/**
 * Computes a gradient vector for 4D noise.
 * @param j The input float.
 * @param ip The input permutation vector.
 * @return The gradient vector.
 */
fn grad4(j: f32, ip: vec4<f32>) -> vec4<f32> {
    let ones = vec4<f32>(1.0, 1.0, 1.0, -1.0);
    var p = vec4<f32>(floor(fract(vec3<f32>(j) * ip.xyz) * 7.0) * ip.z - 1.0, 0.0);
    let w = 1.5 - dot(abs(p.xyz), ones.xyz);
    p.w = w;
    
    let s = vec4<f32>(
        select(0.0, 1.0, p.x < 0.0),
        select(0.0, 1.0, p.y < 0.0),
        select(0.0, 1.0, p.z < 0.0),
        select(0.0, 1.0, w < 0.0)
    );
    
    p = vec4<f32>(p.xyz + (s.xyz * 2.0 - 1.0) * s.www, p.w);
    return p;
}

/**
 * Computes simplex noise derivatives.
 * @param v The input vector.
 * @return The noise derivatives.
 */
fn simplexNoiseDerivatives(v: vec4<f32>) -> vec4<f32> {
    let C = vec4<f32>(0.138196601125011, 0.276393202250021, 0.414589803375032, -0.447213595499958);

    let i = floor(v + dot(v, vec4<f32>(0.309016994374947451)));
    let x0 = v - i + dot(i, C.xxxx);

    var i0 = vec4<f32>(0.0);
    let isX = step(x0.yzw, x0.xxx);
    let isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0 = vec4<f32>(i0.x, 1.0 - isX);
    i0.y += isYZ.x + isYZ.y;
    i0 = vec4<f32>(i0.x, i0.y, i0.z + (1.0 - isYZ.x), i0.w + (1.0 - isYZ.y));
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;

    let i3 = clamp(i0, vec4<f32>(0.0), vec4<f32>(1.0));
    let i2 = clamp(i0 - 1.0, vec4<f32>(0.0), vec4<f32>(1.0));
    let i1 = clamp(i0 - 2.0, vec4<f32>(0.0), vec4<f32>(1.0));

    let x1 = x0 - i1 + C.xxxx;
    let x2 = x0 - i2 + C.yyyy;
    let x3 = x0 - i3 + C.zzzz;
    let x4 = x0 + C.wwww;

    let i_mod = mod289_vec4(i);
    let j0 = permute_f32(permute_f32(permute_f32(permute_f32(i_mod.w) + i_mod.z) + i_mod.y) + i_mod.x);
    let j1 = permute_vec4(permute_vec4(permute_vec4(permute_vec4(
        i_mod.w + vec4<f32>(i1.w, i2.w, i3.w, 1.0))
        + i_mod.z + vec4<f32>(i1.z, i2.z, i3.z, 1.0))
        + i_mod.y + vec4<f32>(i1.y, i2.y, i3.y, 1.0))
        + i_mod.x + vec4<f32>(i1.x, i2.x, i3.x, 1.0));

    let ip = vec4<f32>(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0);

    let p0 = grad4(j0, ip);
    let p1 = grad4(j1.x, ip);
    let p2 = grad4(j1.y, ip);
    let p3 = grad4(j1.z, ip);
    let p4 = grad4(j1.w, ip);

    let norm = taylorInvSqrt_vec4(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    let p0_norm = p0 * norm.x;
    let p1_norm = p1 * norm.y;
    let p2_norm = p2 * norm.z;
    let p3_norm = p3 * norm.w;
    let p4_norm = p4 * taylorInvSqrt_f32(dot(p4, p4));

    let values0 = vec3<f32>(dot(p0_norm, x0), dot(p1_norm, x1), dot(p2_norm, x2));
    let values1 = vec2<f32>(dot(p3_norm, x3), dot(p4_norm, x4));

    let m0 = max(0.5 - vec3<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3<f32>(0.0));
    let m1 = max(0.5 - vec2<f32>(dot(x3, x3), dot(x4, x4)), vec2<f32>(0.0));

    let temp0 = -6.0 * m0 * m0 * values0;
    let temp1 = -6.0 * m1 * m1 * values1;

    let mmm0 = m0 * m0 * m0;
    let mmm1 = m1 * m1 * m1;

    let dx = temp0[0] * x0.x + temp0[1] * x1.x + temp0[2] * x2.x + temp1[0] * x3.x + temp1[1] * x4.x + mmm0[0] * p0.x + mmm0[1] * p1.x + mmm0[2] * p2.x + mmm1[0] * p3.x + mmm1[1] * p4.x;
    let dy = temp0[0] * x0.y + temp0[1] * x1.y + temp0[2] * x2.y + temp1[0] * x3.y + temp1[1] * x4.y + mmm0[0] * p0.y + mmm0[1] * p1.y + mmm0[2] * p2.y + mmm1[0] * p3.y + mmm1[1] * p4.y;
    let dz = temp0[0] * x0.z + temp0[1] * x1.z + temp0[2] * x2.z + temp1[0] * x3.z + temp1[1] * x4.z + mmm0[0] * p0.z + mmm0[1] * p1.z + mmm0[2] * p2.z + mmm1[0] * p3.z + mmm1[1] * p4.z;
    let dw = temp0[0] * x0.w + temp0[1] * x1.w + temp0[2] * x2.w + temp1[0] * x3.w + temp1[1] * x4.w + mmm0[0] * p0.w + mmm0[1] * p1.w + mmm0[2] * p2.w + mmm1[0] * p3.w + mmm1[1] * p4.w;

    return vec4<f32>(dx, dy, dz, dw) * 49.0;
}

/**
 * Computes the curl velocity based on a position.
 * @param position The input position vector.
 * @return The curl velocity vector.
 */
fn getCurlVelocity(position: vec4<f32>) -> vec3<f32> {
    let NOISE_TIME_SCALE = 1.6146594;
    let NOISE_SCALE = 1.3043478;
    let NOISE_POSITION_SCALE = 0.0022283;

    let oldPosition = position.xyz;
    let noisePosition = oldPosition * NOISE_POSITION_SCALE;

    let noiseTime = NOISE_TIME_SCALE;

    var xNoisePotentialDerivatives = vec4<f32>(0.0);
    var yNoisePotentialDerivatives = vec4<f32>(0.0);
    var zNoisePotentialDerivatives = vec4<f32>(0.0);

    let persistence = 0.277174;

    for (var i = 0; i < OCTAVES; i++) {
        let scale = (1.0 / 2.0) * pow(2.0, f32(i));
        let noiseScale = pow(persistence, f32(i));

        xNoisePotentialDerivatives += simplexNoiseDerivatives(vec4<f32>(noisePosition * pow(2.0, f32(i)), noiseTime)) * noiseScale * scale;
        yNoisePotentialDerivatives += simplexNoiseDerivatives(vec4<f32>((noisePosition + vec3<f32>(123.4, 129845.6, -1239.1)) * pow(2.0, f32(i)), noiseTime)) * noiseScale * scale;
        zNoisePotentialDerivatives += simplexNoiseDerivatives(vec4<f32>((noisePosition + vec3<f32>(-9519.0, 9051.0, -123.0)) * pow(2.0, f32(i)), noiseTime)) * noiseScale * scale;
    }

    let noiseVelocity = vec3<f32>(
        zNoisePotentialDerivatives.y - yNoisePotentialDerivatives.z,
        xNoisePotentialDerivatives.z - zNoisePotentialDerivatives.x,
        yNoisePotentialDerivatives.x - xNoisePotentialDerivatives.y
    ) * NOISE_SCALE;

    return noiseVelocity;
}
`;
