// What the sky lighting and the environment cubemap capture besides the sky, after Unreal's
// real-time sky-light capture: the scene's exponential height fog composited at infinite distance
// over the sky and the clouds (SkyAtmosphere::capture_fog), and what lies below the horizon
// (SkyAtmosphere::lower_hemisphere). Bound as `capture` by the passes that include it.

struct SkyCapture {
    layer0        : vec4f,   // density (per m at height), height falloff (per m), height (m)
    layer1        : vec4f,
    inscattering  : vec3f,   // the fog's colour at full opacity
    fogOn         : u32,
    lowerColor    : vec3f,   // the radiance below the horizon, with lowerMode 1
    lowerMode     : u32,     // 0: the ground (under the fog, if any); 1: lowerColor
    captureHeight : f32,     // world height the sky is captured from (m)
    maxOpacity    : f32,
    skyAmbientScale : f32,   // how much of the sky's distant light the fog adds to its colour
    _pad1         : f32,
}

// Optical depth of one exponential layer from height z to infinity along a direction whose up
// component is mu, in closed form: level or downward, the fog thickens without end.
fn captureLayerDepth(layer: vec4f, z: f32, mu: f32) -> f32 {
    if (layer.x <= 0.0) { return 0.0; }
    let k = layer.y * mu;
    if (k <= 1e-7) { return 1e30; }
    return layer.x * exp(min(-layer.y * (z - layer.z), 80.0)) / k;
}

// Whether the capture replaces the lit ground below the horizon (the fog, or a set colour).
fn captureCoversGround() -> bool {
    return capture.fogOn != 0u || capture.lowerMode == 1u;
}

// The radiance the capture sees from world direction d, given `sky`, the sky's (and the clouds')
// radiance there without the fog, and `distant`, the sky's distant light, which the fog adds to
// its colour as Unreal's does.
fn capturedSky(d: vec3f, sky: vec3f, distant: vec3f) -> vec3f {
    var lum = sky;
    if (capture.fogOn != 0u) {
        let tau = captureLayerDepth(capture.layer0, capture.captureHeight, d.y) + captureLayerDepth(capture.layer1, capture.captureHeight, d.y);
        let t = max(exp(-min(tau, 80.0)), 1.0 - capture.maxOpacity);
        lum = lum * t + (capture.inscattering + capture.skyAmbientScale * distant) * (1.0 - t);
    }
    if (capture.lowerMode == 1u && d.y < 0.0) {
        lum = capture.lowerColor;
    }
    return lum;
}
