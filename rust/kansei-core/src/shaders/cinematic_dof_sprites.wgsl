// Scattered bokeh, shared by the extraction (which appends the sprites and bins them) and the
// post-filter (which adds them to the layers). Each sprite is a disc of its CoC in the aperture's
// shape with its energy spread evenly over it. The half-resolution image is cut into BIN x BIN
// bins, each listing the sprites that reach it, so a pixel only visits the sprites that can
// light it, and a frame without highlights costs one read per pixel.

struct Sprite {
    center : vec2f,   // half-resolution pixels
    coc    : f32,     // radius, half-resolution pixels
    layer  : f32,     // 1: near field, 0: background
    color  : vec4f,   // the scattered energy (colour x coverage) of the texel
}

struct HighlightParams {
    contrast   : f32,   // scatter above this multiple of the surroundings' luminance
    minCoc     : f32,   // half-resolution pixels
    maxSprites : u32,
    enabled    : u32,
}

const BIN : u32 = 16u;
// sprites listed per bin; a highlight that would overflow one is gathered instead
const BIN_CAPACITY : u32 = 64u;

fn binGrid() -> vec2u {
    return (halfSize() + BIN - 1u) / BIN;
}

// A sprite's light at a half-resolution pixel. A background sprite may only spread over a
// surface in front of it as far as that surface's CoC (`surfaceCoc`), as in the gather, so it
// never shines over a sharper object in front of it.
fn spriteLight(s: Sprite, pixel: vec2f, surfaceCoc: f32) -> vec3f {
    let offset = pixel - s.center;
    let r = max(s.coc, 0.5);
    let distance = length(offset) / apertureRadius(atan2(offset.y, offset.x));
    let energy = s.color.rgb / (apertureArea() * r * r);
    if (s.layer > 0.5) { return energy * cover(distance, r); }
    let tol = layerTolerance(min(surfaceCoc, r));
    let front = saturate((r - surfaceCoc - tol) / tol);
    return energy * cover(distance, mix(r, min(r, surfaceCoc), front));
}
