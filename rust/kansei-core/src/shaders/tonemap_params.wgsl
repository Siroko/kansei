// The display transform's parameters, shared by tonemap.wgsl and local_exposure.wgsl (one
// upload per frame).

struct ToneMapParams {
    whiteBalance        : mat3x3f,
    gain                : vec3f,
    exposure            : f32,
    shadowGain          : vec3f,
    contrast            : f32,
    highlightGain       : vec3f,
    shadowsMax          : f32,
    saturation          : vec3f,   // per channel, as UE's ColorSaturation
    highlightsMin       : f32,
    shadowSaturation    : vec3f,
    vignette            : f32,
    highlightSaturation : vec3f,
    chromaticAberration : f32,
    grain               : f32,
    grainSize           : f32,
    width               : u32,
    height              : u32,
    frame               : u32,
    tonemapper          : u32,
    flags               : u32,
    _pad0               : u32,
    film                : vec4f,   // Unreal's film curve: slope, toe, shoulder, black clip
    film2               : vec4f,   // its white clip, blue correction, gamut expansion, highlights max
    // Unreal's local exposure (FLAG_LOCAL_EXPOSURE): highlight and shadow contrast, detail
    // strength, blurred luminance blend; log2 of middle grey (exposed), the grid's bins as
    // log2 scene luminance x scale + bias, log2 of the luminance floor; the grid's uv scale, the
    // blur's radius and tap count
    localExposure       : vec4f,
    localExposure2      : vec4f,
    localExposure3      : vec4f,
}

const FLAG_LOCAL_EXPOSURE : u32 = 4u;
// A cell of the bilateral grid: 64 x 64 half-resolution texels (Unreal's)
const LOCAL_CELL          : u32 = 64u;
const LOCAL_CELL_TEXELS   : f32 = 4096.0;
