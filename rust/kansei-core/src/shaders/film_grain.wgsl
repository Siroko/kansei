// Film grain and dither on the sRGB-encoded display signal: the tonemapper's (tonemap.wgsl), and
// SmaaEffect's after its blend when the tonemapper leaves them to it (ToneMapOptions::defer_film).
// Self-contained: it calls nothing of its includers.

// PCG-based 3D hash (Jarzynski & Olano, "Hash Functions for GPU Rendering", 2020).
fn pcg3d(v0: vec3u) -> vec3u {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn hash3(pix: vec2i, frame: u32) -> vec3f {
    return vec3f(pcg3d(vec3u(bitcast<vec2u>(pix), frame))) * (1.0 / 4294967296.0);
}

// Film grain: value noise on a lattice of `grainSize` pixels, a new pattern every frame.
// Each lattice value is triangular in [-1, 1] (two uniforms), which reads like film, not TV static.
fn grainNoise(pix: vec2f, grainSize: f32, frame: u32) -> f32 {
    let q = pix / max(grainSize, 1.0);
    let i = vec2i(floor(q));
    let f = fract(q);
    let s = f * f * (3.0 - 2.0 * f);
    let n00 = hash3(i, frame);
    let n10 = hash3(i + vec2i(1, 0), frame);
    let n01 = hash3(i + vec2i(0, 1), frame);
    let n11 = hash3(i + vec2i(1, 1), frame);
    let v = mix(mix(n00.x + n00.y, n10.x + n10.y, s.x), mix(n01.x + n01.y, n11.x + n11.y, s.x), s.y);
    return v - 1.0;
}

// The film on the encoded colour of `pixel`: grain, strongest in the midtones of the encoded
// signal (where film shows it), then a triangular dither of one 8-bit step, so dusk gradients
// don't band on 8-bit swapchains; clamped to [0, 1].
fn filmApply(encoded: vec3f, pixel: vec2u, grain: f32, grainSize: f32, frame: u32, dither: bool) -> vec3f {
    var c = encoded;
    if (grain > 0.0) {
        let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
        c += grain * 0.12 * grainNoise(vec2f(pixel) + 0.5, grainSize, frame) * (4.0 * l * (1.0 - l));
    }
    if (dither) {
        let h = hash3(vec2i(pixel), frame ^ 0x9e3779b9u);
        c += (h.x + h.y - 1.0) / 255.0;
    }
    return saturate(c);
}
