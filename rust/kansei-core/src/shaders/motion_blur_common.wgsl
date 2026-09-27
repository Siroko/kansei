// Motion blur (McGuire et al. 2012, Jimenez 2014): per-pixel blur vectors and their per-tile
// maximum, the maximum over the tiles whose blur reaches each tile, then a gather along that
// dominant vector with depth-aware weights. Blur vectors are radii in pixels: a pixel's streak
// runs from -v to +v around it.

struct MotionBlurParams {
    invViewProj  : mat4x4f,   // inverse of this frame's jittered view-projection (the depth's)
    viewProj     : mat4x4f,   // this frame, unjittered
    prevViewProj : mat4x4f,   // last frame, unjittered
    size         : vec2f,     // pixels
    scale        : f32,       // per-frame motion in pixels -> blur radius in pixels
    maxRadius    : f32,       // largest blur radius, pixels
    steps        : u32,       // gather samples on each side of the pixel
    frame        : u32,       // noise seed
    enabled      : u32,       // 0: copy the input (camera cuts, amount 0)
    _pad         : u32,
}

// pixels per tile side (the prepare pass' workgroup size)
const TILE : u32 = 16u;

fn tileCount() -> vec2u {
    return (vec2u(p.size) + TILE - 1u) / TILE;
}
