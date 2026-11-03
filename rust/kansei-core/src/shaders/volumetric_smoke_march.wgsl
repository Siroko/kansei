// Smoke marched through the fog (VolumetricFogEffect::set_smoke), appended to the injection's
// shader (for its lights, shadow maps, sky and fog parameters) and smoke_common.wgsl.
//
// One march per view ray (at half resolution, the front-most surface of each 2x2 pixels) over
// every enabled smoke volume at once: the volumes' intervals along the ray, cut at the scene's
// depth, are sorted into segments; each segment is stepped at `stepScale` of the finest cell
// covering it (a gap between volumes is one smoke-free step), every step growing by the same
// factor when the total would pass `maxSteps`. At each step the smoke is lit by the fog's own
// lights (fogDirLight, fogPointLight, fogSpotLight: the same shadow maps), through its own shadow
// toward each light (a short march, steps growing x1.5), with smokePhase, plus the fog's ambient
// and sky; and the fog's froxel grid gives the fog's light and transmittance at each step, so
//   out = Ts * (scene * fog.a + fog.rgb + shafts) + R
// in the composite (volumetric_smoke_composite.wgsl) puts the fog in front of the smoke over it,
// the smoke over the fog behind it: R holds the smoke's light veiled by the fog in front, the fog
// in front of the smoke, less the fog the smoke hides. With no smoke Ts = 1 and R = 0: the fog's
// own composite.
//
// Outputs: light (R, Ts); aux (the texel's scene linear depth, the representative point's uv on
// the previous frame, its distance along the ray), for the temporal filter and the upsample.

struct SmokePush {
    origin    : vec3f,
    radius    : f32,
    direction : vec3f,
    age       : f32,     // seconds
    push      : vec3f,   // m/s
    _pad      : f32,
}

struct SmokeFrame {
    invViewProj    : mat4x4f,
    prevViewProj   : mat4x4f,   // world to the previous frame's clip space
    cameraPos      : vec3f,
    frame          : u32,
    viewForward    : vec3f,
    numVolumes     : u32,
    size           : vec2u,     // the march's target, half the image
    fullSize       : vec2u,
    gridNear       : f32,
    gridFar        : f32,
    gridD          : f32,
    cameraNear     : f32,
    cameraFar      : f32,
    time           : f32,       // the simulation's clock
    advanced       : f32,       // simulated seconds since the previous frame
    stepScale      : f32,
    maxSteps       : u32,
    shadowSteps    : u32,
    jitter         : u32,       // 1: jitter the steps per pixel and frame; 0: the middle of each
    numPushes      : u32,
    shadowDistance : f32,       // the farthest the self-shadow march looks toward a light
    historyValid   : u32,       // 0: the smoke's history starts over (volumetric_smoke_temporal.wgsl)
    blend          : f32,       // the new frame's weight in the history
    _pad0          : f32,
    volumes        : array<SmokeVolumeData, 4>,
    pushes         : array<SmokePush, 8>,
}

@group(0) @binding(30) var<uniform> sf : SmokeFrame;
@group(0) @binding(31) var smokeDepthTex : texture_depth_2d;
@group(0) @binding(32) var smokeAccumTex : texture_3d<f32>;
@group(0) @binding(33) var smokeSampler : sampler;
@group(0) @binding(34) var smokeScalars0 : texture_3d<f32>;
@group(0) @binding(35) var smokeScalars1 : texture_3d<f32>;
@group(0) @binding(36) var smokeScalars2 : texture_3d<f32>;
@group(0) @binding(37) var smokeScalars3 : texture_3d<f32>;
@group(0) @binding(38) var smokeVelocity0 : texture_3d<f32>;
@group(0) @binding(39) var smokeVelocity1 : texture_3d<f32>;
@group(0) @binding(40) var smokeVelocity2 : texture_3d<f32>;
@group(0) @binding(41) var smokeVelocity3 : texture_3d<f32>;
@group(0) @binding(42) var smokeLightOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(43) var smokeAuxOut : texture_storage_2d<rgba32float, write>;

const SMOKE_OPAQUE : f32 = 0.003;

// Volume i's grid (density, temperature) at a point of its unit cube.
fn smokeGridAt(i: u32, local: vec3f) -> vec4f {
    switch i {
        case 0u: { return textureSampleLevel(smokeScalars0, smokeSampler, local, 0.0); }
        case 1u: { return textureSampleLevel(smokeScalars1, smokeSampler, local, 0.0); }
        case 2u: { return textureSampleLevel(smokeScalars2, smokeSampler, local, 0.0); }
        default: { return textureSampleLevel(smokeScalars3, smokeSampler, local, 0.0); }
    }
}

// Volume i's velocity grid (the MAC faces' u, v, w) at a normalized point.
fn smokeVelocityGridAt(i: u32, uvw: vec3f) -> vec4f {
    switch i {
        case 0u: { return textureSampleLevel(smokeVelocity0, smokeSampler, uvw, 0.0); }
        case 1u: { return textureSampleLevel(smokeVelocity1, smokeSampler, uvw, 0.0); }
        case 2u: { return textureSampleLevel(smokeVelocity2, smokeSampler, uvw, 0.0); }
        default: { return textureSampleLevel(smokeVelocity3, smokeSampler, uvw, 0.0); }
    }
}

// Volume i's gas velocity at a point of its unit cube, m/s along the box's axes: its MAC grid's
// (Fluid), its noise's (Curl noise), zero on Waves.
fn smokeVolumeVelocity(i: u32, local: vec3f) -> vec3f {
    let v = sf.volumes[i];
    if (v.motion == SMOKE_CURL) { return smokeVecToBox(v, smokeCurlVelocity(v, smokeLocalToWorld(v, local), sf.time)); }
    if (v.motion != SMOKE_FLUID) { return vec3f(0.0); }
    let d = vec3f(v.dims);
    let c = local * d;
    return vec3f(
        smokeVelocityGridAt(i, vec3f((c.x + 0.5) / d.x, c.y / d.y, c.z / d.z)).x,
        smokeVelocityGridAt(i, vec3f(c.x / d.x, (c.y + 0.5) / d.y, c.z / d.z)).y,
        smokeVelocityGridAt(i, vec3f(c.x / d.x, c.y / d.y, (c.z + 0.5) / d.z)).z,
    );
}

// The gas's velocity at p, world m/s: each volume's there, weighted by its density there.
fn smokeVelocityAtWorld(p: vec3f) -> vec3f {
    var vel = vec3f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < sf.numVolumes; i++) {
        let d = smokeDensityAt(i, p);
        if (d <= 0.0) { continue; }
        let v = sf.volumes[i];
        vel += d * smokeBoxToWorld(v, smokeVolumeVelocity(i, smokeToLocal(v, p)));
        weight += d;
    }
    return vel / max(weight, 1e-6);
}

// Where the mouse's recent pushes have moved the gas at p (the Waves level), metres: each push's
// displacement round its ray, relaxing over `relaxTime` (SmokePushes::displacement).
fn smokePushDisplacement(p: vec3f, relaxTime: f32) -> vec3f {
    var total = vec3f(0.0);
    for (var k = 0u; k < sf.numPushes; k++) {
        let push = sf.pushes[k];
        total += push.push * (smokeMouseFalloff(p, push.origin, push.direction, push.radius) * exp(-push.age / max(relaxTime, 1e-3)));
    }
    return total;
}

// Volume i's density at p (scaled by its densityScale, faded at its faces).
fn smokeDensityAt(i: u32, p: vec3f) -> f32 {
    let v = sf.volumes[i];
    let local = smokeToLocal(v, p);
    if (any(local < vec3f(0.0)) || any(local > vec3f(1.0))) { return 0.0; }
    var d = 0.0;
    if (v.motion == SMOKE_WAVES) {
        // the gas a push moved by D is found at p where it was at p - D
        d = smokeWavesDensity(v, p - smokePushDisplacement(p, v.wavesRelaxTime), sf.time);
    } else {
        d = max(smokeGridAt(i, local).r, 0.0);
    }
    return d * v.densityScale * smokeEdgeFade(v, local);
}

struct SmokeMedium {
    scattering : vec3f,
    extinction : f32,
}

// Every volume's smoke at p, summed.
fn smokeMedium(p: vec3f) -> SmokeMedium {
    var m : SmokeMedium;
    for (var i = 0u; i < sf.numVolumes; i++) {
        let d = smokeDensityAt(i, p);
        m.scattering += d * sf.volumes[i].albedo;
        m.extinction += d * sf.volumes[i].extinction;
    }
    return m;
}

fn smokeExtinctionAt(p: vec3f) -> f32 {
    var e = 0.0;
    for (var i = 0u; i < sf.numVolumes; i++) {
        e += smokeDensityAt(i, p) * sf.volumes[i].extinction;
    }
    return e;
}

// The smoke's optical depth from p toward a light (unit `toLight`, at most `maxDist` away):
// `shadowSteps` steps growing x1.5, out of every volume or to `shadowDistance`.
fn smokeShadowTau(p: vec3f, toLight: vec3f, maxDist: f32, jitter: f32) -> f32 {
    var exitT = 0.0;
    for (var i = 0u; i < sf.numVolumes; i++) {
        let seg = smokeRayInterval(sf.volumes[i], p, toLight);
        if (seg.y > seg.x) { exitT = max(exitT, seg.y); }
    }
    let reach = min(min(exitT, maxDist), sf.shadowDistance);
    if (reach <= 0.0 || sf.shadowSteps == 0u) { return 0.0; }
    var dt = reach * 0.5 / (pow(1.5, f32(sf.shadowSteps)) - 1.0);
    var t = 0.0;
    var tau = 0.0;
    for (var k = 0u; k < sf.shadowSteps; k++) {
        tau += smokeExtinctionAt(p + toLight * (t + dt * jitter)) * dt;
        t += dt;
        dt *= 1.5;
    }
    return tau;
}

fn smokeMax3(v: vec3f) -> f32 {
    return max(v.x, max(v.y, v.z));
}

// The light a unit of smoke scattering at p sends along -rd: the fog's lights through their
// shadow maps and the smoke's own shadow, with smokePhase, and the fog's ambient and sky.
fn smokeLight(p: vec3f, rd: vec3f, jitter: f32) -> vec3f {
    let g = params.anisotropy;
    var light = params.ambient + skyAmbient(rd, p);
    for (var di = 0u; di < params.numDirLights; di++) {
        let l = fogDirLight(di, p);
        if (smokeMax3(l.radiance) <= 0.0) { continue; }
        light += l.radiance * smokePhase(dot(rd, l.toLight), g, smokeShadowTau(p, l.toLight, 1e9, jitter));
    }
    for (var pi = 0u; pi < params.numPointLights; pi++) {
        let l = fogPointLight(pi, p);
        if (smokeMax3(l.radiance) <= 0.0) { continue; }
        let dist = length(ptLights[pi].position - p);
        light += l.radiance * smokePhase(dot(rd, l.toLight), g, smokeShadowTau(p, l.toLight, dist, jitter));
    }
    for (var si = 0u; si < spotLights.count; si++) {
        let l = fogSpotLight(si, p, 0.05);
        if (smokeMax3(l.radiance) <= 0.0) { continue; }
        let dist = length(spotLights.lights[si].position - p);
        light += l.radiance * smokePhase(dot(rd, l.toLight), g, smokeShadowTau(p, l.toLight, dist, jitter));
    }
    return light;
}

// The fog's accumulated (light, transmittance) along the pixel's ray to distance t, looked up in
// the froxels as the composite looks them up (without its per-pixel jitter).
fn smokeFogAt(uv: vec2f, t: f32, cosView: f32) -> vec4f {
    let lin = max(t * cosView, sf.gridNear);
    let w = clamp(depthToSlice(lin, sf.gridNear, sf.gridFar, sf.gridD) / sf.gridD, 0.0, 1.0);
    return textureSampleLevel(smokeAccumTex, smokeSampler, vec3f(uv, w), 0.0);
}

// The world-space ray through a point of the image (uv in [0, 1], y down).
fn smokeRay(uv: vec2f) -> vec3f {
    let p = sf.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    return normalize(p.xyz / p.w - sf.cameraPos);
}

fn smokeIgn(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

// The previous frame's uv of a world point; (-1, -1) behind the camera.
fn smokePrevUv(p: vec3f) -> vec2f {
    let clip = sf.prevViewProj * vec4f(p, 1.0);
    if (clip.w <= 1e-6) { return vec2f(-1.0); }
    return vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
}

@compute @workgroup_size(8, 8)
fn smokeMarch(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= sf.size)) { return; }
    // the front-most surface of the 2x2 pixels this texel stands for
    let full = vec2i(sf.fullSize);
    let px = vec2i(gid.xy) * 2;
    var depth = 1.0;
    var pick = px;
    for (var i = 0; i < 4; i++) {
        let q = min(px + vec2i(i & 1, i >> 1), full - 1);
        let dq = textureLoad(smokeDepthTex, q, 0);
        if (dq < depth) { depth = dq; pick = q; }
    }
    let uv = (vec2f(pick) + 0.5) / vec2f(sf.fullSize);
    let rd = smokeRay(uv);
    let cosView = max(dot(rd, sf.viewForward), 1e-3);
    var linear = sf.cameraFar;
    if (depth < 1.0) { linear = ndcToLinearDepth(depth, sf.cameraNear, sf.cameraFar); }
    let tScene = linear / cosView;

    // each volume's interval along the ray, cut at the scene; their ends, sorted
    var lo : array<f32, 4>;
    var hi : array<f32, 4>;
    var cell : array<f32, 4>;
    var bounds : array<f32, 8>;
    var nb = 0u;
    for (var i = 0u; i < sf.numVolumes; i++) {
        let v = sf.volumes[i];
        let seg = smokeRayInterval(v, sf.cameraPos, rd);
        lo[i] = seg.x;
        hi[i] = min(seg.y, tScene);
        cell[i] = smokeCellSize(v);
        if (hi[i] > lo[i]) {
            bounds[nb] = lo[i];
            bounds[nb + 1u] = hi[i];
            nb += 2u;
        }
    }
    // the history reprojects the texel's centre (the composite's grid), at the point's distance
    let rdTexel = smokeRay((vec2f(gid.xy) + 0.5) / vec2f(sf.size));
    if (nb == 0u) {
        textureStore(smokeLightOut, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
        let prev = smokePrevUv(sf.cameraPos + rdTexel * tScene);
        textureStore(smokeAuxOut, gid.xy, vec4f(linear, prev, tScene));
        return;
    }
    for (var i = 1u; i < nb; i++) {
        let b = bounds[i];
        var j = i;
        while (j > 0u && bounds[j - 1u] > b) {
            bounds[j] = bounds[j - 1u];
            j -= 1u;
        }
        bounds[j] = b;
    }

    // the step budget: every segment at stepScale of its finest cell, all grown alike past it
    var ideal = 0.0;
    var gaps = 0.0;
    for (var s = 0u; s + 1u < nb; s++) {
        let a = bounds[s];
        let b = bounds[s + 1u];
        if (b - a <= 1e-6) { continue; }
        let c = smokeCoverCell(&lo, &hi, &cell, 0.5 * (a + b));
        if (c > 0.0) { ideal += (b - a) / (sf.stepScale * c); } else { gaps += 1.0; }
    }
    let grow = max(ideal / max(f32(sf.maxSteps) - gaps, 1.0), 1.0);

    var jitter = 0.5;
    if (sf.jitter != 0u) { jitter = fract(smokeIgn(vec2f(gid.xy)) + f32(sf.frame % 64u) * 0.618034); }

    var ts = 1.0;
    var rs = vec3f(0.0);
    var rf = vec3f(0.0);
    var fogPrev = smokeFogAt(uv, bounds[0], cosView);
    var lPrev = fogPrev.rgb;
    var weight = 0.0;
    var tWeighted = 0.0;
    let tLast = bounds[nb - 1u];
    var opaque = false;
    for (var s = 0u; s + 1u < nb && !opaque; s++) {
        let a = bounds[s];
        let b = bounds[s + 1u];
        let len = b - a;
        if (len <= 1e-6) { continue; }
        let c = smokeCoverCell(&lo, &hi, &cell, 0.5 * (a + b));
        var n = 1u;
        if (c > 0.0) { n = clamp(u32(ceil(len / (sf.stepScale * c * grow))), 1u, 2u * max(sf.maxSteps, 1u)); }
        let dt = len / f32(n);
        for (var k = 0u; k < n; k++) {
            let t0 = a + f32(k) * dt;
            let t1 = t0 + dt;
            let fog1 = smokeFogAt(uv, t1, cosView);
            var m : SmokeMedium;
            let tx = t0 + jitter * dt;
            let x = sf.cameraPos + rd * tx;
            if (c > 0.0) { m = smokeMedium(x); }
            // the fog's light over this step, seen through the smoke before it and the step's own
            // (its mean transmittance over the step: exact for a constant step)
            var e = 1.0;
            var tsMean = ts;
            var centroid = 0.5;
            let tau = m.extinction * dt;
            if (tau > 1e-6) {
                e = exp(-tau);
                tsMean = ts * (1.0 - e) / tau;
                // where along the step the smoke's light comes from: the mean of e^-sigma s
                centroid = select(1.0 / tau - e / (1.0 - e), 0.5, tau < 1e-3);
            }
            rf += (1.0 - tsMean) * (fog1.rgb - lPrev);
            lPrev = fog1.rgb;
            if (tau > 1e-6) {
                // the smoke's light, veiled by the fog in front of it
                let fogT = mix(fogPrev.a, fog1.a, centroid);
                rs += ts * fogT * m.scattering * smokeLight(x, rd, jitter) * ((1.0 - e) / m.extinction);
                let w = ts * (1.0 - e);
                weight += w;
                tWeighted += w * tx;
                ts *= e;
            }
            fogPrev = fog1;
            if (ts < SMOKE_OPAQUE) {
                let fogLast = smokeFogAt(uv, tLast, cosView).rgb;
                rf += (1.0 - ts) * (fogLast - lPrev);
                lPrev = fogLast;
                opaque = true;
                break;
            }
        }
    }
    let r = max(rs + (1.0 - ts) * lPrev - rf, vec3f(0.0));

    // the point the smoke's light comes from, for the history: weighted by what it absorbed;
    // where it was a frame ago, moved back by the gas's velocity over the sim seconds stepped
    var tP = tScene;
    var gas = vec3f(0.0);
    if (weight > 1e-4) {
        tP = tWeighted / weight;
        gas = smokeVelocityAtWorld(sf.cameraPos + rdTexel * tP);
    }
    let prev = smokePrevUv(sf.cameraPos + rdTexel * tP - gas * sf.advanced);
    textureStore(smokeLightOut, gid.xy, vec4f(min(r, vec3f(60000.0)), ts));
    textureStore(smokeAuxOut, gid.xy, vec4f(linear, prev, tP));
}

// The finest cell among the volumes covering distance t along the ray; 0 in a gap.
fn smokeCoverCell(lo: ptr<function, array<f32, 4>>, hi: ptr<function, array<f32, 4>>, cell: ptr<function, array<f32, 4>>, t: f32) -> f32 {
    var c = 0.0;
    for (var i = 0u; i < sf.numVolumes; i++) {
        if ((*hi)[i] > (*lo)[i] && t >= (*lo)[i] && t <= (*hi)[i]) {
            c = select((*cell)[i], min(c, (*cell)[i]), c > 0.0);
        }
    }
    return c;
}
