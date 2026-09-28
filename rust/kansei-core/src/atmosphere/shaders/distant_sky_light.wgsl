// The sky's distant light, after Unreal's (SkyAtmosphere.usf, RenderDistantSkyLightLutCS): the
// mean radiance of the sky all round a point 6 km above the ground under the world origin, lit by
// the sun and the moon, with the Mie phase taken as uniform. It depends on neither the camera nor
// the ground: one colour for light scattered from far away, the distant fog's and the clouds'
// ambient light. One workgroup; written before the sky lighting, which passes it on.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var multiScatteringLut : texture_2d<f32>;
@group(0) @binding(4) var lutSampler : sampler;
@group(0) @binding(5) var<storage, read_write> distantOut : vec4f;

const DISTANT_THREADS : u32 = 64u;
const DISTANT_DIRECTIONS : u32 = 256u;
const DISTANT_ALTITUDE_KM : f32 = 6.0;
const DISTANT_SAMPLES : u32 = 24u;

var<workgroup> distantSum : array<vec3f, DISTANT_THREADS>;

// As lightScattering (scattering.wgsl), the Mie phase uniform: over the whole sphere it averages
// to the same, and a mean of a few hundred directions would otherwise be noisy toward the sun.
fn distantLightScattering(p: vec3f, r: f32, med: Medium, viewDir: vec3f, lightDir: vec3f, angularRadius: f32, illuminance: vec3f) -> vec3f {
    let mu = dot(p, lightDir) / r;
    let phaseScattering = med.rayleighScattering * rayleighPhase(dot(viewDir, lightDir)) + med.mieScattering / (4.0 * PI);
    let single = horizonVisibility(r, mu, angularRadius) * transmittanceToTop(r, mu) * phaseScattering;
    return illuminance * (single + multiScattering(r, mu) * med.scattering);
}

// As integrateScattering (scattering.wgsl), with the lights above.
fn distantRadiance(ro: vec3f, rd: vec3f) -> vec3f {
    let tGround = rayGround(ro, rd);
    let tMax = select(rayTop(ro, rd), tGround, tGround > 0.0);
    var luminance = vec3f(0.0);
    var transmittance = vec3f(1.0);
    let n = f32(DISTANT_SAMPLES);
    var tPrev = 0.0;
    for (var i = 0u; i < DISTANT_SAMPLES; i++) {
        let f = (f32(i) + 1.0) / n;
        let tNext = tMax * f * f;
        let dt = tNext - tPrev;
        let p = ro + rd * (tPrev + 0.5 * dt);
        tPrev = tNext;
        let r = length(p);
        let med = sampleMedium(r - atm.bottomRadius);
        var s = distantLightScattering(p, r, med, rd, frame.sunDirection, frame.sunAngularRadius, frame.sunIlluminance);
        if (any(frame.moonIlluminance > vec3f(0.0))) {
            s += distantLightScattering(p, r, med, rd, frame.moonDirection, frame.moonAngularRadius, frame.moonIlluminance);
        }
        let segT = exp(-med.extinction * dt);
        luminance += transmittance * (s - s * segT) / max(med.extinction, vec3f(1e-9));
        transmittance *= segT;
    }
    return luminance;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li : u32) {
    let up = normalize(frame.worldOrigin);
    // the world's up is the planet frame's +Y near the origin; the point is right above it
    let ro = up * (atm.bottomRadius + DISTANT_ALTITUDE_KM);
    var sum = vec3f(0.0);
    for (var i = li; i < DISTANT_DIRECTIONS; i += DISTANT_THREADS) {
        // a Fibonacci sphere: directions of equal solid angle
        let y = 1.0 - 2.0 * (f32(i) + 0.5) / f32(DISTANT_DIRECTIONS);
        let s = sqrt(max(1.0 - y * y, 0.0));
        let phi = f32(i) * 2.399963;
        sum += distantRadiance(ro, vec3f(s * cos(phi), y, s * sin(phi)));
    }
    distantSum[li] = sum;
    workgroupBarrier();
    for (var stride = DISTANT_THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) { distantSum[li] += distantSum[li + stride]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        distantOut = vec4f(distantSum[0] / f32(DISTANT_DIRECTIONS) * frame.skyLuminanceFactor, 0.0);
    }
}
