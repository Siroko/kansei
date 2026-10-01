// Cinematic depth of field: a thin-lens circle of confusion from the camera's focal length,
// f-stop, focus distance and filmback, gathered as scattered bokeh in two layers (after Jimenez
// 2014, "Next-Generation Post-Processing in Call of Duty: Advanced Warfare", and Unreal's
// DiaphragmDOF): the near field, in front of the focus plane, which spills over everything
// behind it with coverage; and the background (in focus and behind it), which never bleeds over
// a sharper surface in front of it. Shared by every pass.
//
// The half-resolution layers are rgba32float: rgb the layer's colour, a the layer's CoC radius
// (half-resolution pixels) and coverage of the texel packed together (packLayer).

struct DofParams {
    cocScale      : f32,   // CoC radius (full-resolution pixels) of a point at infinity
    focusDistance : f32,   // metres of view depth
    maxCoc        : f32,   // largest CoC radius, full-resolution pixels
    cameraNear    : f32,
    cameraFar     : f32,
    width         : u32,   // full resolution
    height        : u32,
    sampleCount   : u32,   // gather samples per half-resolution pixel
    bladeCount    : u32,   // aperture blades; below 3 a round aperture
    bladeRotation : f32,   // radians
    frame         : u32,   // 0: a static sample pattern; otherwise it rotates every frame
    debugView     : u32,   // 0 the image; 1 background layer, 2 near layer, 3 near alpha, 4 CoC
}

const PI : f32 = 3.14159265358979;

fn viewDepth(d: f32) -> f32 {
    // glam::Mat4::perspective_rh: [0, 1] depth
    return p.cameraNear * p.cameraFar / (p.cameraFar - d * (p.cameraFar - p.cameraNear));
}

// Signed CoC radius in full-resolution pixels: negative in front of the focus plane, positive
// behind it, cocScale at infinity (the sky). Not clamped.
fn cocFromDepth(d: f32) -> f32 {
    if (d >= 1.0) { return p.cocScale; }
    return p.cocScale * (1.0 - p.focusDistance / max(viewDepth(d), 1e-4));
}

// How much of a full-resolution sample belongs to the near field: nothing within a pixel and a
// half of CoC in front of the focus plane (it renders sharp), all of it beyond two and a half.
fn nearWeight(coc: f32) -> f32 {
    return smoothstep(1.5, 2.5, -coc);
}

// How far apart two CoCs (pixels) must be to count as different depth layers: a pixel for sharp
// surfaces, a fifth of the blur for blurred ones, which cannot show a silhouette finer than that.
fn layerTolerance(coc: f32) -> f32 {
    return 0.5 + 0.2 * abs(coc);
}

// The depth under full-resolution pixel `px`. The depth buffer is at the scene's render size,
// below this pass's when a temporal upscaler before it reconstructs the display size.
fn loadDepth(tex: texture_depth_2d, px: vec2i) -> f32 {
    let dims = vec2i(textureDimensions(tex));
    let q = vec2i((vec2f(px) + 0.5) * vec2f(dims) / vec2f(f32(p.width), f32(p.height)));
    return textureLoad(tex, min(q, dims - 1), 0);
}

// A layer's CoC (to 1/16 pixel) and coverage (to about 1/1000) in one f32.
fn packLayer(coc: f32, coverage: f32) -> f32 {
    return round(coc * 16.0) + min(saturate(coverage), 0.999);
}

fn layerCoc(a: f32) -> f32 {
    return floor(a) / 16.0;
}

fn layerCoverage(a: f32) -> f32 {
    return a - floor(a);
}

fn halfSize() -> vec2u {
    return vec2u((p.width + 1u) / 2u, (p.height + 1u) / 2u);
}

fn cover(distance: f32, coc: f32) -> f32 {
    return saturate(coc - distance + 0.5);
}

// The unit aperture's radius along angle theta: 1 for a round aperture, the polygon's edge for
// `bladeCount` blades.
fn apertureRadius(theta: f32) -> f32 {
    if (p.bladeCount < 3u) { return 1.0; }
    let seg = 2.0 * PI / f32(p.bladeCount);
    let a = theta - p.bladeRotation;
    let local = a - seg * floor(a / seg) - seg * 0.5;
    return cos(seg * 0.5) / cos(local);
}

// The unit aperture's area.
fn apertureArea() -> f32 {
    if (p.bladeCount < 3u) { return PI; }
    let n = f32(p.bladeCount);
    return 0.5 * n * sin(2.0 * PI / n);
}
