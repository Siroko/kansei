// Screen-space global illumination, shared by its passes. View space is right-handed, the camera
// looking down -z; the depth buffer is [0, 1] perspective.

struct SsgiParams {
    proj         : mat4x4f,
    invProj      : mat4x4f,
    view         : mat4x4f,   // world -> view
    invView      : mat4x4f,
    prevViewProj : mat4x4f,   // world -> the previous frame's clip space
    fullSize     : vec2f,     // the image (and depth buffer)
    traceSize    : vec2f,     // the traced, reduced-resolution target
    radius       : f32,       // metres searched around each point
    thickness    : f32,       // metres assumed behind each depth sample
    intensity    : f32,       // of the bounce
    aoStrength   : f32,       // how much of the sky's ambient light occlusion removes
    slices       : u32,
    steps        : u32,       // per side of each slice
    frame        : u32,
    historyValid : u32,
    maxRadiusPx  : f32,
    blend        : f32,       // weight of the new frame in the history
    hasSky       : u32,
    _pad         : f32,
}

const SSGI_PI : f32 = 3.14159265358979;

fn ssgiViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = sp.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

fn ssgiDepth(px: vec2i) -> f32 {
    return textureLoad(depthTex, clamp(px, vec2i(0), vec2i(sp.fullSize) - 1), 0);
}

// The depth-buffer pixel under a uv.
fn ssgiPixel(uv: vec2f) -> vec2i {
    return clamp(vec2i(uv * sp.fullSize), vec2i(0), vec2i(sp.fullSize) - 1);
}
