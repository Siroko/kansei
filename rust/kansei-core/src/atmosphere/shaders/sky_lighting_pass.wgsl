// Projects the sky-view LUT, with the clouds in front of it (cloud_map.wgsl) and the capture fog
// in front of both (sky_capture.wgsl), onto order-2 SH (radiance), adds light bounced off the
// ground below the horizon (unless the capture covers it), and evaluates the sun and the moon at
// the camera. One workgroup.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var skyViewLut : texture_2d<f32>;
@group(0) @binding(4) var lutSampler : sampler;
@group(0) @binding(5) var skyViewSampler : sampler;
@group(0) @binding(6) var<storage, read_write> skyLightingOut : SkyLighting;
@group(0) @binding(7) var cloudMap : texture_2d<f32>;
@group(0) @binding(8) var<uniform> capture : SkyCapture;
@group(0) @binding(9) var<uniform> distantSkyLight : vec4f;   // distant_sky_light.wgsl

// The sky with the clouds in front of it (cloud_map.wgsl)
fn skyWithClouds(d: vec3f) -> vec3f {
    let c = textureSampleLevel(cloudMap, skyViewSampler, cloudMapUv(d), 0.0);
    return skyViewLuminance(d) * (1.0 - c.a) + c.rgb;
}

// 64 threads keep the shared arrays (64 x 11 x 16 bytes) inside WebGPU's 16 KiB default
const THREADS : u32 = 64u;
const N_PHI : u32 = 128u;
const N_COS : u32 = 64u;

var<workgroup> sharedSh : array<array<vec3f, 9>, THREADS>;
var<workgroup> sharedUp : array<vec3f, THREADS>;
var<workgroup> sharedClearUp : array<vec3f, THREADS>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li : u32) {
    // equal-area cells: uniform in azimuth and in cos(zenith) about the world's +Y
    let cellSolidAngle = 4.0 * PI / f32(N_PHI * N_COS);
    var sh : array<vec3f, 9>;
    for (var i = 0u; i < 9u; i++) { sh[i] = vec3f(0.0); }
    var up = vec3f(0.0);   // irradiance on an upward-facing surface, from the sky only
    var clearUp = vec3f(0.0);   // the same without the clouds, which the clouds are lit by
    for (var cell = li; cell < N_PHI * N_COS; cell += THREADS) {
        let phi = (f32(cell % N_PHI) + 0.5) / f32(N_PHI) * 2.0 * PI;
        let cosT = 1.0 - 2.0 * (f32(cell / N_PHI) + 0.5) / f32(N_COS);
        let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
        let d = vec3f(sinT * cos(phi), cosT, sinT * sin(phi));
        let lum = capturedSky(d, skyWithClouds(d), distantSkyLight.rgb) * cellSolidAngle;
        var y = skyShBasis(d);
        for (var i = 0u; i < 9u; i++) { sh[i] += lum * y[i]; }
        up += lum * max(cosT, 0.0);
        if (cosT > 0.0) { clearUp += skyViewLuminance(d) * (cellSolidAngle * cosT); }
    }
    sharedSh[li] = sh;
    sharedUp[li] = up;
    sharedClearUp[li] = clearUp;
    workgroupBarrier();
    for (var stride = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            for (var i = 0u; i < 9u; i++) { sharedSh[li][i] += sharedSh[li + stride][i]; }
            sharedUp[li] += sharedUp[li + stride];
            sharedClearUp[li] += sharedClearUp[li + stride];
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
    let ground = select(frame.skyLightGroundAlbedo / PI * groundIrradiance, vec3f(0.0), captureCoversGround());
    var out : SkyLighting;
    for (var i = 0u; i < 9u; i++) { out.sh[i] = vec4f(sharedSh[0][i], 0.0); }
    out.sh[0] += vec4f(ground * (2.0 * PI * 0.282095), 0.0);
    out.sh[1] += vec4f(ground * (-PI * 0.488603), 0.0);
    out.sunIlluminance = vec4f(sun, sunVis);
    out.sunDirection = vec4f(frame.sunDirection, 0.0);
    out.moonIlluminance = vec4f(moon, moonVis);
    out.moonDirection = vec4f(frame.moonDirection, 0.0);
    out.clearSkyUp = vec4f(sharedClearUp[0], 0.0);
    out.distantSkyLight = distantSkyLight;
    skyLightingOut = out;
}
