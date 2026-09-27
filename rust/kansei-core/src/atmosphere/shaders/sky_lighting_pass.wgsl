// Projects the sky-view LUT onto order-2 SH (radiance), adds light bounced off the ground below the
// horizon, and evaluates the sun and the moon at the camera. One workgroup.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var skyViewLut : texture_2d<f32>;
@group(0) @binding(4) var lutSampler : sampler;
@group(0) @binding(5) var skyViewSampler : sampler;
@group(0) @binding(6) var<storage, read_write> skyLightingOut : SkyLighting;

// 64 threads keep the shared arrays (64 x 10 x 16 bytes) inside WebGPU's 16 KiB default
const THREADS : u32 = 64u;
const N_PHI : u32 = 128u;
const N_COS : u32 = 64u;

var<workgroup> sharedSh : array<array<vec3f, 9>, THREADS>;
var<workgroup> sharedUp : array<vec3f, THREADS>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li : u32) {
    // equal-area cells: uniform in azimuth and in cos(zenith) about the world's +Y
    let cellSolidAngle = 4.0 * PI / f32(N_PHI * N_COS);
    var sh : array<vec3f, 9>;
    for (var i = 0u; i < 9u; i++) { sh[i] = vec3f(0.0); }
    var up = vec3f(0.0);   // irradiance on an upward-facing surface, from the sky only
    for (var cell = li; cell < N_PHI * N_COS; cell += THREADS) {
        let phi = (f32(cell % N_PHI) + 0.5) / f32(N_PHI) * 2.0 * PI;
        let cosT = 1.0 - 2.0 * (f32(cell / N_PHI) + 0.5) / f32(N_COS);
        let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
        let d = vec3f(sinT * cos(phi), cosT, sinT * sin(phi));
        let lum = skyViewLuminance(d) * cellSolidAngle;
        var y = skyShBasis(d);
        for (var i = 0u; i < 9u; i++) { sh[i] += lum * y[i]; }
        up += lum * max(cosT, 0.0);
    }
    sharedSh[li] = sh;
    sharedUp[li] = up;
    workgroupBarrier();
    for (var stride = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            for (var i = 0u; i < 9u; i++) { sharedSh[li][i] += sharedSh[li + stride][i]; }
            sharedUp[li] += sharedUp[li + stride];
        }
        workgroupBarrier();
    }
    if (li != 0u) { return; }

    // the lights at the camera, after the atmosphere
    let r = length(frame.cameraPos);
    let upDir = frame.cameraPos / r;
    let muSun = dot(upDir, frame.sunDirection);
    let muMoon = dot(upDir, frame.moonDirection);
    let sunVis = horizonVisibility(r, muSun, frame.sunAngularRadius);
    let moonVis = horizonVisibility(r, muMoon, frame.moonAngularRadius);
    let sun = frame.sunIlluminance * transmittanceToTop(r, muSun) * sunVis;
    let moon = frame.moonIlluminance * transmittanceToTop(r, muMoon) * moonVis;

    // below the horizon: a Lambertian ground lit by the sky and the lights; a constant c over the
    // lower hemisphere (y < 0) projects onto band 0 as 2 pi 0.282095 c and onto y as -pi 0.488603 c
    let groundIrradiance = sharedUp[0] + sun * max(dot(vec3f(0.0, 1.0, 0.0), frame.sunDirection), 0.0)
                         + moon * max(dot(vec3f(0.0, 1.0, 0.0), frame.moonDirection), 0.0);
    let ground = frame.skyLightGroundAlbedo / PI * groundIrradiance;
    var out : SkyLighting;
    for (var i = 0u; i < 9u; i++) { out.sh[i] = vec4f(sharedSh[0][i], 0.0); }
    out.sh[0] += vec4f(ground * (2.0 * PI * 0.282095), 0.0);
    out.sh[1] += vec4f(ground * (-PI * 0.488603), 0.0);
    out.sunIlluminance = vec4f(sun, sunVis);
    out.sunDirection = vec4f(frame.sunDirection, 0.0);
    out.moonIlluminance = vec4f(moon, moonVis);
    out.moonDirection = vec4f(frame.moonDirection, 0.0);
    skyLightingOut = out;
}
