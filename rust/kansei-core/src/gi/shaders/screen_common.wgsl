// Voxel GI on screen (gi::VoxelGIEffect), shared by its passes. View space is right-handed, the
// camera looking down -z; the depth buffer is [0, 1] perspective.

struct VoxelGiParams {
    invProj      : mat4x4f,
    invView      : mat4x4f,
    view         : mat4x4f,   // world -> view
    prevViewProj : mat4x4f,   // world -> the previous frame's clip space
    fullSize     : vec2f,     // the image (and depth buffer)
    traceSize    : vec2f,     // the traced, reduced-resolution target
    nearSize     : vec2f,     // the near field's (screen-space GI's) traced target, if any
    startVoxels  : f32,       // voxels out along the normal the cones start from
    maxDistance  : f32,       // metres a cone looks
    maxSteps     : u32,       // per cone
    frame        : u32,
    intensity    : f32,       // of the bounce
    ambient      : f32,       // how much of the material's own sky ambient the GI replaces
    blend        : f32,       // weight of the new frame in the history
    historyValid : u32,
    hasSky       : u32,
    debug        : u32,       // 1: output only the light the GI adds; 2: the volume's voxels; 3: a slice of the distance field; 4: the probes
    nearField    : u32,       // 1: screen-space GI in front, the voxels past it
    skyScale     : f32,       // of the sky past the volume
    sdfAo        : f32,       // strength of the distance field's AO on the GI (0: none)
    sdfSlice     : f32,       // height of the debug slice, metres
    hasSdf       : u32,
    _pad2        : u32,
    _pad0        : u32,
    _pad1        : u32,
}

fn gpViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = gp.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

fn gpDepth(px: vec2i) -> f32 {
    return textureLoad(depthTex, clamp(px, vec2i(0), vec2i(gp.fullSize) - 1), 0);
}

// The depth-buffer pixel under a uv.
fn gpPixel(uv: vec2f) -> vec2i {
    return clamp(vec2i(uv * gp.fullSize), vec2i(0), vec2i(gp.fullSize) - 1);
}
