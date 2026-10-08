// Ray-traced reflections (rt::RtReflectionsEffect), shared by the trace and the resolve: the
// parameters, the GBuffer's surface (its F0 and roughness, GBUFFER_OUT_WGSL's
// kansei_gbuffer_out_specular), and which pixel of each block is traced this frame. View space
// is right-handed, the camera looking down -z; the depth buffer is [0, 1] perspective.

struct RtReflectParams {
    invProj      : mat4x4f,
    invView      : mat4x4f,
    prevViewProj : mat4x4f,   // world -> the previous frame's clip space
    fullSize     : vec2f,
    traceSize    : vec2f,
    frame        : u32,
    downscale    : u32,       // a pixel of each block this many wide is traced a frame
    flags        : u32,       // RT_REFLECT_ALPHA, RT_REFLECT_STATS, RT_REFLECT_HISTORY, RT_REFLECT_GRID
    view         : u32,       // 0: lit; 1: the reflection Fresnel adds; 2: the mirror image; 3: its cost
    coneTan      : f32,       // the voxel cone past the grid (tan of its half-angle)
    coneSteps    : u32,
    maxDistance  : f32,
    intensity    : f32,
    skyScale     : f32,
    blend        : f32,       // weight of the new frame in the history
    heatScale    : f32,
    _pad         : u32,
    viewProj     : mat4x4f,   // world -> this frame's clip space (the hits looked up on screen)
    glassInterfaces : u32,    // the most surfaces a ray through the glass crosses or reflects off
    glassSamples : u32,       // paths a frosted glass pixel a frame (averaged, then over frames)
    _pad2        : u32,
    _pad3        : u32,
}

const RT_REFLECT_ALPHA : u32 = 1u;
const RT_REFLECT_STATS : u32 = 2u;
const RT_REFLECT_HISTORY : u32 = 4u;
// trace the grid of triangles (off: the voxel cone alone, from the surface)
const RT_REFLECT_GRID : u32 = 8u;
// light a hit by the lit image where the camera sees the same point (else the voxels)
const RT_REFLECT_SCREEN : u32 = 16u;
// light the hits the camera doesn't see by the lights (shadow rays) and the voxels' irradiance, not
// the voxels' radiance
const RT_REFLECT_DIRECT : u32 = 32u;
// the volume's anisotropic mips are bound (40-45): srcIrradiance gathers as voxel GI's composite
const RT_REFLECT_ANISO : u32 = 64u;
// the hits the camera doesn't see get no indirect light (an image with no GI on screen)
const RT_REFLECT_NO_INDIRECT : u32 = 128u;
// glass is drawn (rt_glass.wgsl): the rays hit the grid's glass triangles too
const RT_REFLECT_GLASS : u32 = 256u;

fn rtViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = rp.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

fn rtWorldPos(uv: vec2f, depth: f32) -> vec3f {
    return (rp.invView * vec4f(rtViewPos(uv, depth), 1.0)).xyz;
}

// The pixel of each downscale x downscale block traced this frame: every one in turn, spread out
// (a Bayer order).
fn rtJitter(frame: u32, downscale: u32) -> vec2u {
    if (downscale <= 1u) {
        return vec2u(0u);
    }
    if (downscale == 2u) {
        var order = array<vec2u, 4>(vec2u(0u, 0u), vec2u(1u, 1u), vec2u(1u, 0u), vec2u(0u, 1u));
        return order[frame % 4u];
    }
    var order = array<vec2u, 16>(
        vec2u(0u, 0u), vec2u(2u, 2u), vec2u(2u, 0u), vec2u(0u, 2u),
        vec2u(1u, 1u), vec2u(3u, 3u), vec2u(3u, 1u), vec2u(1u, 3u),
        vec2u(1u, 0u), vec2u(3u, 2u), vec2u(3u, 0u), vec2u(1u, 2u),
        vec2u(0u, 1u), vec2u(2u, 3u), vec2u(2u, 1u), vec2u(0u, 3u),
    );
    return order[frame % 16u] * (downscale / 4u);
}

// Glass at a pixel (GBUFFER_OUT_WGSL's kansei_gbuffer_out_glass): its albedo's alpha 0.05-0.45,
// below the 0.5-1 of the reflective surfaces.
fn rtIsGlassAlbedo(a: f32) -> bool {
    return a > 0.025 && a < 0.475;
}

// A reflective surface at a pixel: valid where the depth holds a surface whose material wrote a
// normal and an F0 (GBUFFER_OUT_WGSL's kansei_gbuffer_out_specular stores 1 - F0 in the normal's
// alpha and 1 - roughness / 2 in the albedo's; kansei_gbuffer_out leaves both at 1), not glass.
struct RtSurfacePixel {
    world     : vec3f,
    n         : vec3f,
    f0        : f32,
    roughness : f32,
    valid     : bool,
}

fn rtSurface(px: vec2u) -> RtSurfacePixel {
    var s : RtSurfacePixel;
    let depth = textureLoad(depthTex, px, 0);
    let raw = textureLoad(normalTex, px, 0);
    s.f0 = clamp(1.0 - raw.w, 0.0, 1.0);
    s.valid = depth < 1.0 && dot(raw.xyz, raw.xyz) > 1e-4 && s.f0 > 1e-3;
    if (!s.valid) {
        return s;
    }
    let a = textureLoad(albedoTex, px, 0).a;
    if (rtIsGlassAlbedo(a)) {
        s.valid = false;
        return s;
    }
    s.n = normalize(raw.xyz * 2.0 - 1.0);
    s.roughness = clamp(2.0 * (1.0 - a), 0.0, 1.0);
    s.world = rtWorldPos((vec2f(px) + 0.5) / rp.fullSize, depth);
    return s;
}

// The half-angle's tangent of a rough surface's reflection lobe (GGX's alpha, roughness squared).
fn rtLobeTan(roughness: f32) -> f32 {
    return min(roughness * roughness, 2.0);
}
