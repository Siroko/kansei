/// Blue noise table dimensions.
pub const BLUE_NOISE_SIZE: u32 = 128;

/// Generate a 128x128 blue noise lookup table.
/// Returns 128*128 = 16384 float values in [0, 1).
///
/// Uses the R2 quasi-random (Kronecker) sequence, which provides
/// good low-discrepancy coverage and decorrelated samples across
/// the 2D domain. This is not true void-and-cluster blue noise but
/// gives adequate spectral distribution for path tracer jitter.
pub fn generate_blue_noise() -> Vec<f32> {
    let size = (BLUE_NOISE_SIZE * BLUE_NOISE_SIZE) as usize;
    let mut noise = vec![0.0f32; size];

    // Plastic constant reciprocal — optimal 2D Kronecker sequence base
    let phi2: f64 = 1.0 / 1.3247179572447460259609088; // 1/plastic_constant
    let alpha1 = phi2;
    let alpha2 = phi2 * phi2;

    for i in 0..size {
        let x = (i % BLUE_NOISE_SIZE as usize) as f64;
        let y = (i / BLUE_NOISE_SIZE as usize) as f64;
        noise[i] = ((0.5 + alpha1 * x + alpha2 * y) % 1.0).abs() as f32;
    }

    noise
}
