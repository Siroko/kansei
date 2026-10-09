// Ray-traced direct light (rt::RtShadowsEffect): the sun's and the spot lights' light on the
// GBuffer's surfaces that leave their direct light to it (rt_shadows_gbuffer.wgsl), shadowed by
// rays through the grid of triangles toward a point of each light's emitter (a cone round the
// sun, a disk or a rectangle for a spot light) and by short screen-space rays for contacts the grid
// misses. The visibility is traced at the trace resolution, accumulated over frames and filtered
// across neighbours with depth and normal edge stops, then upsampled and lit at full resolution.
//
// Passes: `trace` (visibility of up to 8 lights in two rgba16float textures, and the guide: the
// world normal octahedral-packed, linear depth, history length), `temporal` (blended with last
// frame's, reprojected), `atrous` (an a-trous wavelet step, run 2-3 times), `composite` (the BRDF
// times each light's illuminance times its visibility, added to the lit colour). Prefixed with
// RT_GRID_WGSL (its bindings in group 1) and RT_OPAQUE_WGSL in the trace pass.

struct RtShadowParams {
    invProj      : mat4x4f,
    invView      : mat4x4f,
    viewProj     : mat4x4f,   // world -> this frame's clip space (unjittered)
    prevViewProj : mat4x4f,   // world -> last frame's clip space (unjittered)
    fullSize     : vec2f,
    traceSize    : vec2f,
    frame        : u32,
    downscale    : u32,
    numLights    : u32,
    flags        : u32,       // RT_SHADOW_*
    contactLength: f32,       // metres the screen-space contact rays walk (0: none)
    contactThickness : f32,   // how thick a depth sample is taken to be (m)
    temporalAlpha: f32,
    maxDistance  : f32,       // metres a ray toward the sun looks
    view         : u32,       // 0 lit, 1 visibility, 2 direct light alone
    debugLight   : u32,       // the light the visibility view shows
    stepWidth    : u32,       // this a-trous pass's step (texels)
    lastStep     : u32,
    phiDepth     : f32,
    phiNormal    : f32,
    intensity    : f32,
    maxHistory   : f32,
}

struct RtShadowLight {
    position  : vec3f,
    kind      : u32,          // 0 directional, 1 spot
    direction : vec3f,        // the way the light travels (a spot's axis)
    range     : f32,
    color     : vec3f,        // lux (directional) or candela (spot), times the colour
    cosOuter  : f32,
    axisU     : vec3f,        // a rectangle's half extents (zero: a disk)
    cosInner  : f32,
    axisV     : vec3f,
    radius    : f32,          // the disk's radius (m), or the sun's angular radius (radians)
}

// last frame's guide and visibility are this view's (off: a cut, or the first frame)
const RT_SHADOW_HISTORY : u32 = 1u;
// screen-space contact rays
const RT_SHADOW_CONTACT : u32 = 2u;
// no velocity written at a pixel: reproject by depth
const RT_SHADOW_NO_VELOCITY : f32 = 1.0e4;
const RT_SHADOW_PI : f32 = 3.14159265;
const RT_SHADOW_MAX_LIGHTS : u32 = 8u;

@group(0) @binding(0) var<uniform> sp : RtShadowParams;
@group(0) @binding(1) var<storage, read> shadowLights : array<RtShadowLight>;

fn shViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = sp.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

fn shWorldPos(uv: vec2f, depth: f32) -> vec3f {
    return (sp.invView * vec4f(shViewPos(uv, depth), 1.0)).xyz;
}

fn shHash(a: u32, b: u32, c: u32) -> f32 {
    var h = a * 747796405u + b * 2891336453u + c * 1181783497u + 2654435769u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return f32(h >> 8u) / 16777216.0;
}

// The frame's 2D sample for a pixel and light: the R2 sequence by frame, rotated per pixel by
// interleaved gradient noise, so each pixel walks the whole sequence over time.
fn shSample2(px: vec2u, frame: u32, dim: u32) -> vec2f {
    let ign = fract(52.9829189 * fract(dot(vec2f(px) + f32(dim) * vec2f(17.0, 59.0), vec2f(0.06711056, 0.00583715))));
    let rot = vec2f(ign, shHash(px.x, px.y, dim + 7u));
    let r2 = fract(vec2f(0.7548776662, 0.5698402910) * f32(frame % 65536u) + 0.5);
    return fract(r2 + rot);
}

// Two unit vectors perpendicular to unit `n` and to each other.
fn shBasis(n: vec3f) -> mat2x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y));
}

fn shOctWrap(v: vec2f) -> vec2f {
    return (1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

fn shEncodeNormal(n: vec3f) -> vec2f {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    return select(shOctWrap(p), p, n.z >= 0.0);
}

fn shDecodeNormal(p: vec2f) -> vec3f {
    var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    n = vec3f(n.xy + select(vec2f(t), vec2f(-t), n.xy >= vec2f(0.0)), n.z);
    return normalize(n);
}

// The light's illuminance on a surface facing it at `p` (lux), and the unit vector toward it: a
// directional light's own, a spot light's as `kansei_spot_sample` computes it.
struct ShIncoming { toLight: vec3f, illuminance: vec3f, distance: f32 };

fn shIncoming(light: RtShadowLight, p: vec3f) -> ShIncoming {
    var s: ShIncoming;
    if (light.kind == 0u) {
        s.toLight = -normalize(light.direction);
        s.illuminance = light.color;
        s.distance = sp.maxDistance;
        return s;
    }
    let d = light.position - p;
    let dist2 = max(dot(d, d), 1e-4);
    let dist = sqrt(dist2);
    let l = d / dist;
    let r = dist2 / (light.range * light.range);
    let window = clamp(1.0 - r * r, 0.0, 1.0);
    let cone = clamp((dot(-l, normalize(light.direction)) - light.cosOuter) / max(light.cosInner - light.cosOuter, 1e-4), 0.0, 1.0);
    s.toLight = l;
    s.illuminance = light.color * (window * window * cone * cone / dist2);
    s.distance = dist;
    return s;
}

// GBuffer: the surfaces this effect lights mark the emissive target's alpha (rt_shadows_gbuffer.wgsl).
fn shLitHere(emissiveAlpha: f32) -> bool {
    return emissiveAlpha > 0.03 && emissiveAlpha < 0.47;
}

fn shRoughness(emissiveAlpha: f32) -> f32 {
    return clamp((emissiveAlpha - 0.05) / 0.4, 0.0, 1.0);
}
