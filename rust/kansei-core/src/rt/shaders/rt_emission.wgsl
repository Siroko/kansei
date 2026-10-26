// Emitted light packed in one word for the ray tracing grid (RtSurface::record_emission): RGBE
// (Ward's shared exponent), r | g << 8 | b << 16 | e << 24, the value (m + 0.5) x 2^(e - 136);
// 0 is black. Its 8-bit mantissas keep about 1% of the brightest channel; radiances from 2^-127
// to 2^127. rt_grid.wgsl's kansei_rt_emission reads it back from a hit.

fn kansei_rt_pack_emission(radiance: vec3f) -> u32 {
    let c = max(radiance, vec3f(0.0));
    let peak = max(c.r, max(c.g, c.b));
    if (peak < 1e-30) {
        return 0u;
    }
    // 2^(x - 1) <= peak < 2^x, and the mantissas in [0, 255]
    let x = clamp(i32(floor(log2(peak))) + 1, -127, 127);
    let m = min(vec3u(c * exp2(f32(8 - x))), vec3u(255u));
    return m.r | (m.g << 8u) | (m.b << 16u) | (u32(x + 128) << 24u);
}

fn kansei_rt_unpack_emission(w: u32) -> vec3f {
    let e = w >> 24u;
    if (e == 0u) {
        return vec3f(0.0);
    }
    return (vec3f(f32(w & 255u), f32((w >> 8u) & 255u), f32((w >> 16u) & 255u)) + 0.5) * exp2(f32(e) - 136.0);
}
