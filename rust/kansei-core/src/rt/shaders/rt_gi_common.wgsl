// Ray-traced diffuse GI (rt::RtDiffuseGiEffect), shared by its passes: the parameters, the
// GBuffer's surface at a pixel, the trace pixel each full-resolution pixel maps to, and the
// guide texel (linear depth and normal) the denoiser's edge stops read. View space is
// right-handed, the camera looking down -z; the depth buffer is [0, 1] perspective.
//
// The signal every pass carries is the radiance arriving along one cosine-distributed direction
// (its mean is the irradiance over pi): the diffuse light leaving a pixel is its albedo times it.

struct RtGiParams {
    invProj      : mat4x4f,
    invView      : mat4x4f,
    prevViewProj : mat4x4f,   // world -> the previous frame's clip space (unjittered)
    fullSize     : vec2f,
    traceSize    : vec2f,
    frame        : u32,
    downscale    : u32,       // trace texel t is full-resolution pixel t * downscale
    flags        : u32,       // RT_GI_*
    view         : u32,       // 0 lit, 1 indirect, 2 signal, 3 variance, 4 history, 5 cost
    maxDistance  : f32,       // metres a ray looks, through the grid then the voxels
    coneTan      : f32,       // the voxel cone a ray takes past the grid
    coneSteps    : u32,
    skyScale     : f32,
    intensity    : f32,
    ambient      : f32,       // share of the material's own sky ambient the GI replaces
    hitConeTan   : f32,       // the cone a hit's indirect light is read through
    hitConeSteps : u32,
    bounces      : u32,       // reference: path vertices past the first
    accumCount   : u32,       // frames accumulated before this one
    alphaColor   : f32,       // SVGF's temporal blend floors
    alphaMoments : f32,
    phiColor     : f32,       // SVGF's edge stops
    phiNormal    : f32,
    phiDepth     : f32,
    numDirLights : u32,
    numPointLights : u32,
    hasShadowMap : u32,
    pixelSize    : f32,       // a full-resolution pixel's width at unit view depth
    heatScale    : f32,       // scale of the debug views' colours
    maxHistory   : f32,
    atrousRadius : u32,       // the wavelet's taps each way: 2 (5x5) or 1 (3x3)
    nearDistance : f32,       // metres a ray walks the grid before the voxels take over (0: the box)
    _pad0        : u32,
}

// alpha-test the grid's alpha-tested triangles
const RT_GI_ALPHA : u32 = 1u;
// count the rays' work
const RT_GI_STATS : u32 = 2u;
// last frame's guide and history are this view's (off: a cut, or the first frame)
const RT_GI_HISTORY : u32 = 4u;
// trace the grid (off: every ray is a voxel cone from the surface)
const RT_GI_GRID : u32 = 8u;
// hits lit by their exact direct light plus a voxel cone's indirect (off: the voxels' radiance)
const RT_GI_HIT_DIRECT : u32 = 16u;
// shadow rays through the grid for the hit's direct light (off: the shadow maps)
const RT_GI_SHADOW_RAY : u32 = 32u;
// a path tracer through the grid alone: no voxels, `bounces` vertices, direct light at each
const RT_GI_REFERENCE : u32 = 64u;
// accumulate the raw signal into a running mean (the composite shows the mean)
const RT_GI_ACCUMULATE : u32 = 128u;
const RT_GI_HAS_SKY : u32 = 256u;
const RT_GI_NO_VELOCITY : f32 = 1.0e4;
const RT_GI_PI : f32 = 3.14159265;

fn giViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = gp.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

fn giWorldPos(uv: vec2f, depth: f32) -> vec3f {
    return (gp.invView * vec4f(giViewPos(uv, depth), 1.0)).xyz;
}

fn giUv(px: vec2u) -> vec2f {
    return (vec2f(px) + 0.5) / gp.fullSize;
}

// The full-resolution pixel trace texel `t` stands for.
fn giPixel(t: vec2u) -> vec2u {
    return min(t * gp.downscale, vec2u(gp.fullSize) - 1u);
}

fn giLuminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// A hash of three words (PCG-like), in [0, 1).
fn giHash(a: u32, b: u32, c: u32) -> f32 {
    var h = a * 747796405u + b * 2891336453u + c * 1181783497u + 2654435769u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return f32(h >> 8u) / 16777216.0;
}

// The frame's 2D sample at a pixel: the R2 sequence (Roberts 2018) indexed by frame, rotated per
// pixel by interleaved gradient noise (Jimenez 2014) so neighbours' samples differ (Cranley-
// Patterson), the rotation fixed so each pixel walks the whole low-discrepancy sequence over time.
fn giSample2(px: vec2u, frame: u32, dim: u32) -> vec2f {
    let ign = fract(52.9829189 * fract(dot(vec2f(px) + f32(dim) * vec2f(17.0, 59.0), vec2f(0.06711056, 0.00583715))));
    let rot = vec2f(ign, giHash(px.x, px.y, dim + 7u));
    let r2 = fract(vec2f(0.7548776662, 0.5698402910) * f32(frame % 65536u) + 0.5);
    return fract(r2 + rot);
}

// A direction about `n`, distributed as its cosine (pdf cos / pi), from a 2D sample.
fn giCosineDir(n: vec3f, xi: vec2f) -> vec3f {
    let phi = 6.2831853 * xi.x;
    let r = sqrt(xi.y);
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3f(b, s + n.y * n.y * a, -n.y);
    return normalize(t * (r * cos(phi)) + bt * (r * sin(phi)) + n * sqrt(max(1.0 - xi.y, 0.0)));
}

// The guide texel (rg32uint): linear depth (f32 bits; 0 for no surface) and the world normal
// octahedral-packed into two snorm16 (Cigolle et al. 2014).
fn giOctWrap(v: vec2f) -> vec2f {
    return (1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

fn giEncodeGuide(z: f32, n: vec3f) -> vec4u {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    p = select(giOctWrap(p), p, n.z >= 0.0);
    return vec4u(bitcast<u32>(z), pack2x16snorm(p), 0u, 0u);
}

// (z, normal); z 0 where the texel holds no surface.
fn giDecodeGuide(g: vec4u) -> vec4f {
    let p = unpack2x16snorm(g.y);
    var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    n = vec3f(n.xy + select(vec2f(t), vec2f(-t), n.xy >= vec2f(0.0)), n.z);
    return vec4f(bitcast<f32>(g.x), normalize(n));
}

fn giHeat(x: f32) -> vec3f {
    let t = clamp(x, 0.0, 1.0);
    return clamp(vec3f(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), vec3f(0.0), vec3f(1.0));
}
