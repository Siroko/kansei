// Spot-light shafts (VolumetricFogEffect, SpotScattering::Raymarched), shared by their passes.

struct ShaftParams {
    invViewProj  : mat4x4f,
    prevViewProj : mat4x4f,   // world to the previous frame's clip space
    cameraPos    : vec3f,
    frame        : u32,
    viewForward  : vec3f,     // the camera's forward axis, for linear depth
    steps        : u32,       // samples per light along each ray
    size         : vec2u,     // the traced target, half the image
    fullSize     : vec2u,
    gridNear     : f32,
    gridFar      : f32,
    gridD        : f32,
    blend        : f32,       // weight of the new frame in the history
    cameraNear   : f32,
    cameraFar    : f32,
    historyValid : u32,
    _pad         : f32,
}

// The world-space ray through a point of the image (uv in [0, 1], y down).
fn shaftRay(uv: vec2f) -> vec3f {
    let p = sp.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    return normalize(p.xyz / p.w - sp.cameraPos);
}
