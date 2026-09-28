// Volumetric clouds, march: a layer of cloud between two altitudes around the planet, marched at
// a reduced resolution (after Schneider 2015 and Hillaire 2016, "Physically Based Sky,
// Atmosphere and Cloud Rendering in Frostbite").
// - Density: a weather map says where clouds form and how tall they grow; a height profile shapes
//   them from flat stratus to towering cumulus; a Perlin-Worley shape noise builds them and a
//   Worley detail noise frays their edges.
// - Light: the sun after the atmosphere (the transmittance LUT at each sample) and through the
//   cloud toward it (a short march), with a dual-lobe phase and Wrenninge's approximation of
//   multiple scattering, which a diffusion term takes over from deep inside thick cloud; plus
//   the clear sky's light from above, dimmer toward the base.
// - The atmosphere in front of the cloud: the aerial-perspective LUT at the cloud's depth.
// - Like the sky they cover, the clouds follow the atmosphere's sky luminance factor: the sun
//   they scatter and the air in front of them are scaled by it (their sky light is the sky
//   lighting's, scaled already), so a sky darkened by the factor stays consistent where clouds
//   cover it.
// - Each frame jitters the march and blends with the previous frames, reprojected by the
//   cloud's depth, so a few dozen steps resolve smoothly.
// Output: rgb the light the cloud sends to the camera (after the atmosphere), a its
// transmittance; the composite adds it over the sky. Distances are in kilometres.

struct CloudParams {
    prevViewProj  : mat4x4f,   // world (metres) to the previous frame's clip space
    windOffset    : vec3f,     // km the clouds have drifted
    coverage      : f32,       // 0 clear .. 1 overcast
    bottomKm      : f32,       // altitude of the layer's base
    topKm         : f32,       // altitude of its top
    extinction    : f32,       // per km at full density
    cloudType     : f32,       // 0 stratus .. 1 cumulus
    albedo        : vec3f,
    shapeScale    : f32,       // shape texture repeats per km
    detailScale   : f32,
    weatherScale  : f32,
    maxDistance   : f32,       // km marched at most
    frame         : u32,
    size          : vec2u,     // of the cloud target
    historyValid  : u32,
    steps         : u32,
    lightSteps    : u32,
    blend         : f32,       // weight of the new frame in the history
    mapSteps      : u32,       // the cloud map's steps (its light steps are at most 3)
    mapInterleave : u32,       // the cloud map rewrites one row in this many each frame
}

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var<uniform> frame : SkyFrame;
@group(0) @binding(2) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(3) var lutSampler : sampler;
@group(0) @binding(4) var apScattering : texture_3d<f32>;
@group(0) @binding(5) var apTransmittance : texture_3d<f32>;
@group(0) @binding(6) var<uniform> sky : SkyLighting;
@group(0) @binding(7) var depthTex : texture_depth_2d;
@group(0) @binding(8) var shapeTex : texture_3d<f32>;
@group(0) @binding(9) var detailTex : texture_3d<f32>;
@group(0) @binding(10) var weatherTex : texture_2d<f32>;
@group(0) @binding(11) var noiseSampler : sampler;
@group(0) @binding(12) var historyTex : texture_2d<f32>;
@group(0) @binding(13) var outColor : texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var outDepth : texture_storage_2d<r32float, write>;
@group(0) @binding(15) var<uniform> cp : CloudParams;

const FAR_KM : f32 = 1.0e4;

fn remap(v: f32, lo: f32, hi: f32, newLo: f32, newHi: f32) -> f32 {
    return newLo + (v - lo) / max(hi - lo, 1e-5) * (newHi - newLo);
}

fn henyeyGreenstein(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// Forward-scattering silver lining plus a little back-scatter
fn cloudPhase(c: f32, scale: f32) -> f32 {
    return mix(henyeyGreenstein(c, 0.8 * scale), henyeyGreenstein(c, -0.3 * scale), 0.25);
}

// Vertical profile: stratus hug their base, cumulus rise and round off at the top.
fn heightProfile(h: f32, kind: f32) -> f32 {
    let stratus = saturate(remap(h, 0.0, 0.1, 0.0, 1.0)) * saturate(remap(h, 0.2, 0.35, 1.0, 0.0));
    let strato = saturate(remap(h, 0.0, 0.2, 0.0, 1.0)) * saturate(remap(h, 0.35, 0.7, 1.0, 0.0));
    let cumulus = saturate(remap(h, 0.0, 0.12, 0.0, 1.0)) * saturate(remap(h, 0.6, 0.98, 1.0, 0.0));
    return select(mix(strato, cumulus, kind * 2.0 - 1.0), mix(stratus, strato, kind * 2.0), kind < 0.5);
}

// Cloud density (0..1) at planet-frame point p (km), height h (0 base .. 1 top of the layer).
// The weather map (about uniform over [0.3, 1]) marks a `coverage` share of the sky where clouds
// may form; there, coverage also sets how much of the shape noise becomes cloud, so a thin
// coverage leaves scattered puffs and a full one a closed deck.
fn cloudDensity(p: vec3f, h: f32, detailed: bool) -> f32 {
    let q = p + cp.windOffset;
    let weather = textureSampleLevel(weatherTex, noiseSampler, q.xz * cp.weatherScale, 0.0);
    let where_ = saturate((weather.r - 0.3) / 0.7);
    let local = smoothstep(1.0 - cp.coverage - 0.15, 1.0 - cp.coverage + 0.15, where_);
    if (local <= 0.0) { return 0.0; }
    let kind = saturate(cp.cloudType + (weather.g - 0.4) * 0.5);
    let profile = heightProfile(h, kind);
    if (profile <= 0.0) { return 0.0; }
    // the shape leans downwind with height
    let s = textureSampleLevel(shapeTex, noiseSampler, (q + vec3f(h * 0.5, 0.0, 0.0)) * cp.shapeScale, 0.0);
    let fbm = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
    let base = saturate(remap(s.r, fbm - 1.0, 1.0, 0.0, 1.0));
    let amount = cp.coverage * local;
    var cloud = saturate(remap(base * profile, 1.0 - amount, 1.0, 0.0, 1.0)) * local;
    if (!detailed || cloud <= 0.0) { return cloud; }
    let d = textureSampleLevel(detailTex, noiseSampler, q * cp.detailScale, 0.0);
    let dfbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    // wispy at the base, billowy higher up
    let erode = mix(dfbm, 1.0 - dfbm, saturate(h * 5.0));
    return saturate(remap(cloud, erode * 0.3, 1.0, 0.0, 1.0));
}

// Optical depth (per unit extinction) toward the sun from p, through the rest of the layer.
fn sunOpticalDepth(p: vec3f, sunDir: vec3f, rBottom: f32, rTop: f32, lightSteps: u32) -> f32 {
    let exitTop = raySphere(p, sunDir, rTop).y;
    let span = min(max(exitTop, 0.0), (rTop - rBottom) * 3.0);
    var depth = 0.0;
    var t = 0.0;
    let n = max(lightSteps, 1u);
    for (var i = 0u; i < n; i++) {
        // steps growing toward the sun: fine detail near the sample, the bulk further out
        let dt = span * (f32(2u * i + 1u) / f32(n * n));
        t += dt * 0.5;
        let q = p + sunDir * t;
        let h = (length(q) - rBottom) / (rTop - rBottom);
        depth += cloudDensity(q, saturate(h), i < 2u) * dt;
        t += dt * 0.5;
    }
    return depth;
}

fn ign(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

fn unproject(uv: vec2f, depth: f32) -> vec3f {
    let p = frame.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

struct CloudMarch {
    light         : vec3f,   // scattered toward the start of the ray (before the air in front)
    transmittance : f32,
    depth         : f32,     // km: transmittance-weighted distance of the cloud, else the end
    hit           : bool,
}

// The cloud layer along a ray from ro (planet frame, km), at most maxKm, in `stepCount` steps (and
// `lightSteps` toward the sun); jitter in [0, 1) offsets the first step.
fn marchClouds(ro: vec3f, rd: vec3f, maxKm: f32, jitter: f32, stepCount: u32, lightSteps: u32) -> CloudMarch {
    let rBottom = atm.bottomRadius + cp.bottomKm;
    let rTop = atm.bottomRadius + cp.topKm;
    let r0 = length(ro);
    let inner = raySphere(ro, rd, rBottom);
    let outer = raySphere(ro, rd, rTop);
    var tStart = 0.0;
    var tEnd = 0.0;
    if (r0 < rBottom) {
        // below the layer: from leaving the inner sphere to leaving the outer one, if the ray
        // does not meet the ground first
        let ground = rayGround(ro, rd);
        if (ground < 0.0) { tStart = inner.y; tEnd = outer.y; }
    } else if (r0 < rTop) {
        tEnd = select(outer.y, inner.x, inner.x > 0.0);
    } else if (outer.x > 0.0) {
        tStart = outer.x;
        tEnd = select(outer.y, inner.x, inner.x > 0.0);
    }
    tEnd = min(min(tEnd, tStart + cp.maxDistance), maxKm);

    var transmittance = 1.0;
    var light = vec3f(0.0);
    var depthSum = 0.0;
    var depthWeight = 0.0;
    if (tEnd > tStart) {
        let steps = max(stepCount, 1u);
        let dt = max((tEnd - tStart) / f32(steps), 0.01);
        let sunDir = frame.sunDirection;
        let cosSun = dot(rd, sunDir);
        let phase = array<f32, 3>(cloudPhase(cosSun, 1.0), cloudPhase(cosSun, 0.5), cloudPhase(cosSun, 0.25));
        var t = tStart + dt * jitter;
        for (var i = 0u; i < steps; i++) {
            if (t >= tEnd) { break; }
            let p = ro + rd * t;
            let r = length(p);
            let h = saturate((r - rBottom) / (rTop - rBottom));
            let density = cloudDensity(p, h, true);
            if (density > 0.0) {
                let sigma = density * cp.extinction;
                let up = p / r;
                // the sun after the atmosphere above this point, and after the cloud toward it
                // the clouds are part of the sky, so they follow its luminance factor (their sky light
                // comes from the sky lighting, which already does)
                let sunIn = frame.sunIlluminance * frame.skyLuminanceFactor * transmittanceToTop(r, dot(up, sunDir))
                          * horizonVisibility(r, dot(up, sunDir), frame.sunAngularRadius);
                let od = sunOpticalDepth(p, sunDir, rBottom, rTop, lightSteps) * cp.extinction;
                // multiple scattering (Wrenninge 2013): octaves of weaker extinction, flatter phase
                var ms = 0.0;
                var a = 1.0;
                var b = 1.0;
                for (var o = 0; o < 3; o++) {
                    ms += a * phase[o] * exp(-b * od);
                    a *= 0.5;
                    b *= 0.5;
                }
                // deep inside, light diffuses through rather than decaying exponentially: a thick
                // cloud transmits about 1 / (1 + 0.75 (1 - g) tau) of the sun (g about 0.85), so
                // overcast bases stay grey rather than black
                ms = max(ms, 1.0 / (4.0 * PI) / (1.0 + 0.11 * od));
                // the clear sky's light from above (the sky lighting's SH holds the clouds too, as
                // seen from below), dimmer toward the base
                let ambient = sky.clearSkyUp.rgb / PI * mix(0.35, 1.0, h);
                let source = (sunIn * ms + ambient) * cp.albedo * sigma;
                let stepT = exp(-sigma * dt);
                // energy-conserving integration over the step (Hillaire 2016)
                light += transmittance * (source - source * stepT) / max(sigma, 1e-6);
                depthSum += transmittance * (1.0 - stepT) * t;
                depthWeight += transmittance * (1.0 - stepT);
                transmittance *= stepT;
                if (transmittance < 0.01) { break; }
            }
            t += dt;
        }
    }
    var m : CloudMarch;
    m.light = light;
    m.transmittance = transmittance;
    m.hit = depthWeight > 1e-4;
    m.depth = select(max(tEnd, tStart), depthSum / max(depthWeight, 1e-6), m.hit);
    return m;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= cp.size)) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(cp.size);
    // the farthest scene depth under this texel: clouds show wherever any sky does
    let dsize = vec2f(textureDimensions(depthTex));
    let dc = vec2i(uv * dsize);
    var depth = 0.0;
    for (var i = 0; i < 4; i++) {
        let c = clamp(dc + vec2i(i & 1, i >> 1) - 1, vec2i(0), vec2i(dsize) - 1);
        depth = max(depth, textureLoad(depthTex, c, 0));
    }
    let near = unproject(uv, 0.0);
    let rd = normalize(unproject(uv, 1.0) - near);
    var sceneKm = FAR_KM;
    if (depth < 1.0) { sceneKm = length(unproject(uv, depth) - frame.cameraWorld) * 0.001; }

    let march = marchClouds(frame.cameraPos, rd, sceneKm, fract(ign(vec2f(gid.xy)) + f32(cp.frame) * 0.618034), cp.steps, cp.lightSteps);
    var light = march.light;
    let transmittance = march.transmittance;
    let cloudKm = march.depth;
    let depthWeight = select(0.0, 1.0, march.hit);
    // the atmosphere between the camera and the cloud
    let world = frame.cameraWorld + rd * (cloudKm * 1000.0);
    if (depthWeight > 1e-4) {
        let ap = aerialPerspective(uv, world);
        light = light * ap.transmittance + ap.scattering * frame.skyLuminanceFactor * (1.0 - transmittance);
    }
    var result = vec4f(light, transmittance);

    // blend with the previous frames, reprojected by the cloud's depth
    if (cp.historyValid != 0u) {
        let clip = cp.prevViewProj * vec4f(world, 1.0);
        let prevUv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
        if (clip.w > 0.0 && all(prevUv >= vec2f(0.0)) && all(prevUv <= vec2f(1.0))) {
            let history = textureSampleLevel(historyTex, lutSampler, prevUv, 0.0);
            result = mix(history, result, cp.blend);
        }
    }
    textureStore(outColor, gid.xy, result);
    textureStore(outDepth, gid.xy, vec4f(select(cloudKm, FAR_KM, depthWeight <= 1e-4 && depth >= 1.0), 0.0, 0.0, 0.0));
}

@group(0) @binding(16) var skyViewLut : texture_2d<f32>;
@group(0) @binding(17) var skyViewSampler : sampler;
@group(0) @binding(18) var cloudMapOut : texture_storage_2d<rgba16float, write>;

// The cloud map (cloud_map.wgsl) for the sky lighting: the layer around the camera in every
// direction, with fewer steps than the view (it only feeds low-frequency lighting), one row in
// `mapInterleave` rewritten each frame. The air in front of each cloud is taken as the sky's own light along the ray times
// the air's opacity up to the cloud, which is close for the low-frequency lighting it feeds.
@compute @workgroup_size(8, 8)
fn skyMap(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(cloudMapOut);
    if (any(gid.xy >= size) || (gid.y + cp.frame) % max(cp.mapInterleave, 1u) != 0u) { return; }
    let d = cloudMapDirection((vec2f(gid.xy) + 0.5) / vec2f(size));
    let march = marchClouds(frame.cameraPos, d, FAR_KM, 0.5, cp.mapSteps, min(cp.lightSteps, 3u));
    var light = march.light;
    if (march.hit) {
        let air = transmittanceBetween(frame.cameraPos, d, march.depth);
        light = light * air + skyViewLuminance(d) * (vec3f(1.0) - air) * (1.0 - march.transmittance);
    }
    textureStore(cloudMapOut, gid.xy, vec4f(light, 1.0 - march.transmittance));
}

@group(0) @binding(19) var cloudShadowOut : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(20) var<uniform> shadowParams : CloudShadowParams;
const SHADOW_STEPS : u32 = 16u;

// The cloud shadow map (cloud_shadow.wgsl): from each texel's point on the plane, the layer's
// transmittance along the sun's ray, from the layer's base to its top (the base shapes only: at
// the map's tens of metres per texel the eroded detail would only alias).
@compute @workgroup_size(8, 8)
fn shadowMap(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(cloudShadowOut);
    if (any(gid.xy >= size)) { return; }
    let sunDir = shadowParams.sunDir;
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
    let xz = shadowParams.center + (uv - 0.5) / shadowParams.invSize;
    let ro = frame.cameraPos + (vec3f(xz.x, shadowParams.planeY, xz.y) - frame.cameraWorld) * 0.001;
    let rBottom = atm.bottomRadius + cp.bottomKm;
    let rTop = atm.bottomRadius + cp.topKm;
    let t0 = max(raySphere(ro, sunDir, rBottom).y, 0.0);
    let t1 = raySphere(ro, sunDir, rTop).y;
    var od = 0.0;
    if (t1 > t0) {
        // a low sun crosses a long stretch of the layer: the part that counts is near its base
        let dt = min(t1 - t0, (rTop - rBottom) * 4.0) / f32(SHADOW_STEPS);
        for (var i = 0u; i < SHADOW_STEPS; i++) {
            let q = ro + sunDir * (t0 + (f32(i) + 0.5) * dt);
            let h = saturate((length(q) - rBottom) / (rTop - rBottom));
            od += cloudDensity(q, h, false) * dt;
        }
    }
    textureStore(cloudShadowOut, gid.xy, vec4f(exp(-od * cp.extinction), 0.0, 0.0, 1.0));
}
