export const shaderCode = /* wgsl */`
// Constants

/**
 * The number of octaves used in the Fractal Brownian Motion (FBM) calculation.
 */
const OCTAVES: i32 = 8;

// Utility functions

/**
 * Generates a pseudo-random number based on a 2D vector input.
 * 
 * @param co A 2D vector used as input for the random number generation.
 * @return A pseudo-random float value between 0.0 and 1.0.
 */
fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

/**
 * Generates a pseudo-random number based on a 2D vector input.
 * 
 * @param st A 2D vector used as input for the random number generation.
 * @return A pseudo-random float value between 0.0 and 1.0.
 */
fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

// Noise function

/**
 * Computes a 2D noise value based on a 2D vector input.
 * 
 * @param st A 2D vector representing the input coordinates.
 * @return A float value representing the noise at the given coordinates.
 */
fn noise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);

    let a = random(i);
    let b = random(i + vec2<f32>(1.0, 0.0));
    let c = random(i + vec2<f32>(0.0, 1.0));
    let d = random(i + vec2<f32>(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
           (c - a) * u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}

// Fractal Brownian Motion

/**
 * Computes the Fractal Brownian Motion (FBM) value for a given 2D vector input.
 * 
 * @param st_input A 2D vector representing the input coordinates.
 * @return A float value representing the FBM at the given coordinates.
 */
fn fbm(st_input: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var st = st_input;

    for (var i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}
`;
