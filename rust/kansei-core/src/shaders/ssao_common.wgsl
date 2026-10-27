// Screen-space ambient occlusion, shared by its passes (postprocessing/effects/ssao.rs). View
// space is right-handed, the camera looking down -z; the depth buffer is [0, 1] perspective and
// uv runs down the screen. The GBuffer's depth and normal are bindings 1 and 2 of every pass.

struct SSAOParams {
    proj        : mat4x4f,
    invProj     : mat4x4f,
    view        : mat4x4f,   // world -> view
    invView     : mat4x4f,
    skyUp       : vec4f,     // the hemisphere ambient (ambient 0): radiance from straight up
    skyDown     : vec4f,     // and from straight down
    size        : vec2f,     // the output (and the AO targets)
    radius      : f32,       // metres around each point
    bias        : f32,       // metres per metre of distance from the camera
    kernelSize  : u32,
    strength    : f32,       // how much of the occluded ambient is taken out
    maxRadiusPx : f32,       // the radius on screen at most
    ambient     : u32,       // 0 the hemisphere, 1 the sky's SH, 2 the colour less its emissive
    debug       : u32,       // 1: output the occlusion (1 open, 0 occluded) in place of the image
    _pad0       : u32,
    _pad1       : u32,
    _pad2       : u32,
}

@group(0) @binding(0) var<uniform> params : SSAOParams;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;

const SSAO_PI : f32 = 3.14159265358979;

// The GBuffer's texel under output pixel `px` (the GBuffer is at the render size, which is the
// output's unless the effect runs after a temporal upscaler).
fn ssaoGBufferPixel(px: vec2i) -> vec2i {
    let dims = vec2f(textureDimensions(depthTex));
    return clamp(vec2i((vec2f(px) + 0.5) * dims / params.size), vec2i(0), vec2i(dims) - 1);
}

fn ssaoDepth(px: vec2i) -> f32 {
    return textureLoad(depthTex, ssaoGBufferPixel(clamp(px, vec2i(0), vec2i(params.size) - 1)), 0);
}

fn ssaoViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = params.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

// The view-space point at the centre of pixel `px`.
fn ssaoViewPosAt(px: vec2i) -> vec3f {
    return ssaoViewPos((vec2f(px) + 0.5) / params.size, ssaoDepth(px));
}

// The view-space normal at pixel `px` (whose point is `p`), toward the camera: the GBuffer's
// world normal where the material wrote one, else one rebuilt from the depth around the pixel.
fn ssaoViewNormal(px: vec2i, p: vec3f) -> vec3f {
    let raw = textureLoad(normalTex, ssaoGBufferPixel(px), 0).xyz;
    if (dot(raw, raw) > 1e-4) {
        let n = (params.view * vec4f(normalize(raw * 2.0 - 1.0), 0.0)).xyz;
        if (dot(n, n) > 1e-8) { return normalize(n); }
    }
    let r = ssaoViewPosAt(px + vec2i(1, 0));
    let l = ssaoViewPosAt(px - vec2i(1, 0));
    let d = ssaoViewPosAt(px + vec2i(0, 1));
    let u = ssaoViewPosAt(px - vec2i(0, 1));
    // the side with the smaller depth step, so edges don't bend the normal; uv runs down the
    // screen, so (right) x (up) faces the camera, and the sign check holds it there
    let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
    let dy = select(p - d, u - p, abs(u.z - p.z) < abs(p.z - d.z));
    var n = normalize(cross(dx, dy));
    if (dot(n, -p) < 0.0) { n = -n; }
    return n;
}

// Interleaved gradient noise (Jimenez 2014): a per-pixel value in [0, 1) whose neighbourhoods
// spread evenly, so a small blur averages it out.
fn ssaoNoise(px: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
}
