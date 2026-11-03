// Smoke volumes (simulations::smoke), shared by the fill, the solver and the fog's march: a
// volume's data as the GPU reads it, its box maths and its edge fade. Needs curl_noise.wgsl.
//
// A volume is a box in world space, yawed about +Y as LocalFogVolume is. Its grids span the
// box: local coordinates [0, 1]^3, cell (i, j, k) centred at (i + 0.5, j + 0.5, k + 0.5) / dims.

const SMOKE_MAX_VOLUMES : u32 = 4u;
const SMOKE_MAX_SOURCES : u32 = 4u;
const SMOKE_WAVES : u32 = 0u;
const SMOKE_CURL  : u32 = 1u;
const SMOKE_FLUID : u32 = 2u;

struct SmokeSourceData {
    position : vec3f,
    radius   : f32,
    velocity : vec3f,
    rate     : f32,     // density per second at the centre
}

struct SmokeVolumeData {
    center          : vec3f,
    cosYaw          : f32,
    size            : vec3f,    // full extents, metres
    sinYaw          : f32,
    dims            : vec3u,    // the grid's cells (Waves: the step length's)
    motion          : u32,      // SMOKE_WAVES, SMOKE_CURL or SMOKE_FLUID
    albedo          : vec3f,
    densityScale    : f32,
    extinction      : f32,
    dissipation     : f32,
    edgeFade        : f32,      // fraction of the box
    numSources      : u32,
    wavesAmplitude  : f32,
    wavesWavelength : f32,
    wavesSpeed      : f32,
    wavesSpread     : f32,
    wavesRelaxTime  : f32,
    curlStrength    : f32,
    curlScale       : f32,
    curlSpeed       : f32,
    curlRise        : f32,
    layerCount      : u32,      // SmokeFill::Layers (0: none)
    layerThickness  : f32,
    layerDensity    : f32,
    sources         : array<SmokeSourceData, 4>,
}

// One volume's step (SmokeSimulation::step), a slot of its own in one buffer.
struct SmokeStep {
    volume            : SmokeVolumeData,
    dt                : f32,
    time              : f32,
    buoyancy          : f32,
    weight            : f32,
    cooling           : f32,
    vorticity         : f32,
    maxSpeed          : f32,
    ambientDraft      : f32,
    draftScale        : f32,
    mouseActive       : u32,
    pressureIterations: u32,    // 0: no projection
    _pad1             : f32,
    sourceTemperature : vec4f,  // each source's
    mouseOrigin       : vec3f,
    mouseRadius       : f32,
    mouseDirection    : vec3f,
    _pad2             : f32,
    mousePush         : vec3f,  // m/s
    _pad3             : f32,
}

// World vector into the box's axes (metres, not scaled).
fn smokeVecToBox(v: SmokeVolumeData, d: vec3f) -> vec3f {
    return vec3f(v.cosYaw * d.x - v.sinYaw * d.z, d.y, v.sinYaw * d.x + v.cosYaw * d.z);
}

// A vector along the box's axes into world space.
fn smokeBoxToWorld(v: SmokeVolumeData, q: vec3f) -> vec3f {
    return vec3f(v.cosYaw * q.x + v.sinYaw * q.z, q.y, -v.sinYaw * q.x + v.cosYaw * q.z);
}

// A world point in the box's unit cube.
fn smokeToLocal(v: SmokeVolumeData, p: vec3f) -> vec3f {
    return smokeVecToBox(v, p - v.center) / v.size + 0.5;
}

// A point of the unit cube in world space.
fn smokeLocalToWorld(v: SmokeVolumeData, local: vec3f) -> vec3f {
    return v.center + smokeBoxToWorld(v, (local - 0.5) * v.size);
}

// The part [t0, t1] of the ray o + t d inside the box, t0 >= 0; empty when x >= y.
fn smokeRayInterval(v: SmokeVolumeData, o: vec3f, d: vec3f) -> vec2f {
    let q = smokeVecToBox(v, o - v.center);
    let r = smokeVecToBox(v, d);
    let half = v.size * 0.5;
    let inv = 1.0 / select(r, vec3f(1e-12), abs(r) < vec3f(1e-12));
    let a = (-half - q) * inv;
    let b = (half - q) * inv;
    let lo = min(a, b);
    let hi = max(a, b);
    return vec2f(max(max(lo.x, max(lo.y, lo.z)), 0.0), min(hi.x, min(hi.y, hi.z)));
}

// 1 inside the box, fading to 0 over the outer `edgeFade` of it at every face.
fn smokeEdgeFade(v: SmokeVolumeData, local: vec3f) -> f32 {
    let e = min(local, 1.0 - local);
    let m = min(e.x, min(e.y, e.z));
    if (v.edgeFade <= 0.0) { return select(0.0, 1.0, m >= 0.0); }
    let t = saturate(m / v.edgeFade);
    return t * t * (3.0 - 2.0 * t);
}

// The edge of a cell along the box's longest side, metres.
fn smokeCellSize(v: SmokeVolumeData) -> f32 {
    return max(v.size.x, max(v.size.y, v.size.z)) / f32(max(v.dims.x, max(v.dims.y, v.dims.z)));
}

// Henyey-Greenstein, as the fog's henyeyGreenstein (volumetric_fog_inject.wgsl).
fn smokeHG(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) * 0.0795774715 / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
}

// The light a sample of smoke scatters toward the camera per unit of a light's radiance, with
// `tau` the smoke's optical depth toward the light: single scattering through the smoke's own
// shadow, plus two of Wrenninge's multiple-scattering octaves (each half the extinction and half
// the anisotropy, at half the weight), counted only for what single scattering lost to the
// shadow. With tau = 0 it is the fog's phase, so thin smoke scatters as the fog does.
fn smokePhase(cosTheta: f32, g: f32, tau: f32) -> f32 {
    let single = exp(-tau);
    var p = smokeHG(cosTheta, g) * single;
    var a = 1.0;
    for (var o = 0; o < 2; o++) {
        a *= 0.5;
        p += a * smokeHG(cosTheta, a * g) * (exp(-a * tau) - single);
    }
    return p;
}

// The Curl noise level's velocity at a world point, m/s: curl noise of the volume's scale and
// speed, times its strength, and its rise (divergence-free, so the gas neither piles up nor thins).
fn smokeCurlVelocity(v: SmokeVolumeData, p: vec3f, time: f32) -> vec3f {
    return curlNoise(p / v.curlScale, time * v.curlSpeed) * v.curlStrength + vec3f(0.0, v.curlRise, 0.0);
}
