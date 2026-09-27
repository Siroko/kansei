// Cinematic depth of field: a thin-lens circle of confusion from the camera's focal length,
// f-stop, focus distance and filmback, gathered as scattered bokeh (after Jimenez 2014,
// "Next-Generation Post-Processing in Call of Duty: Advanced Warfare"). Shared by every pass.

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
    _pad          : u32,
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

fn halfSize() -> vec2u {
    return vec2u((p.width + 1u) / 2u, (p.height + 1u) / 2u);
}
