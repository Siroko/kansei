// Scrolling elevation map: 256x256 grid of f32 values.
// Each frame:
//   - Rows 1..255: copy from previous frame's row below (scroll up)
//   - Row 0 (bottom): write current FFT waveform + simplex noise
//
// Dispatched as (256/16, 256/16) = (16, 16) workgroups of 16x16 threads.

struct ScrollParams {
    time: f32,
    noiseScale: f32,
    noiseStrength: f32,
    noiseSpeed: f32,
    noiseZoom: f32,
    noiseMix: f32,  // 0 = FFT only, 1 = noise only
    peakExponent: f32,
    temporalBlend: f32,
    fftPow: f32,
    peakMin: f32, // minimum peak value at edges (0 = fully faded, 1 = no fade)
};

@group(0) @binding(0) var elevRead: texture_2d<f32>;                      // previous frame
@group(0) @binding(1) var elevWrite: texture_storage_2d<r32float, write>;  // current frame
@group(0) @binding(2) var<storage, read> fftRow: array<u32>;               // 256 FFT values (0-255)
@group(0) @binding(3) var<uniform> params: ScrollParams;

// ── Simple hash-based noise (no lookup tables = Safari safe) ──
fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn valueNoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep
    let a = hash2d(i);
    let b = hash2d(i + vec2<f32>(1.0, 0.0));
    let c = hash2d(i + vec2<f32>(0.0, 1.0));
    let d = hash2d(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var val: f32 = 0.0;
    var amp: f32 = 0.5;
    var freq: f32 = 1.0;
    var pos = p;
    for (var i = 0; i < octaves; i++) {
        val += amp * (valueNoise(pos * freq) * 2.0 - 1.0);
        amp *= 0.5;
        freq *= 2.0;
    }
    return val;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= 256u || y >= 256u) { return; }

    if (y > 0u) {
        // Scroll: copy from the row below in the previous frame
        let val = textureLoad(elevRead, vec2<u32>(x, y - 1u), 0).r;
        textureStore(elevWrite, vec2<u32>(x, y), vec4<f32>(val, 0.0, 0.0, 0.0));
    } else {
        // Row 0: FFT waveform + noise, temporally smoothed.
        let rawFft = (f32(fftRow[x]) - 128.0) / 128.0; // -1..+1
        // Apply power curve then normalize to 0..1 (always positive Z)
        let powered = sign(rawFft) * pow(abs(rawFft), params.fftPow);
        let fftVal = (powered + 1.0) * 0.5;

        // Pure FFT only — noise is applied per-vertex in the displacement shaders
        // for spatial coherence across the elevation surface.
        let mixed = fftVal;

        // Symmetric exponential peak: full amplitude at center (x=128),
        // decays toward edges (x=0 and x=255). exp(-exponent * dist^2).
        let t = f32(x) / 255.0;
        let dist = (t - 0.5) * 2.0; // -1..+1
        let peak = max(params.peakMin, exp(-params.peakExponent * dist * dist));

        let newVal = mixed * peak;

        let prevVal = textureLoad(elevRead, vec2<u32>(x, 0u), 0).r;
        textureStore(elevWrite, vec2<u32>(x, y), vec4<f32>(mix(prevVal, newVal, params.temporalBlend), 0.0, 0.0, 0.0));
    }
}
