// Spot-light shafts, raymarched per pixel (VolumetricFogEffect, SpotScattering::Raymarched),
// appended to the injection's shader for its fog parameters, media and spot lights.
//
// A froxel is a few metres deep where the beams are, so a trunk or a clump of grass shadowing a
// beam that crosses the view averages away with the lit fog around it. Here each view ray (at half
// resolution) is cut against each light's cone and range, and `steps` jittered samples over that
// segment alone take the fog's density there, the light after its shadow map, and the fog's
// transmittance from the camera (the froxel grid, which holds the other light and the medium).
// The frames are then filtered (volumetric_fog_shafts_temporal.wgsl) and upsampled by depth in the
// composite. Output: rgb the light scattered toward the camera, a the pixel's linear depth.

@group(0) @binding(13) var shaftDepthTex : texture_depth_2d;
@group(0) @binding(14) var shaftAccumTex : texture_3d<f32>;
@group(0) @binding(15) var shaftAccumSampler : sampler;
@group(0) @binding(16) var shaftsOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(17) var<uniform> sp : ShaftParams;

const SHAFT_EMPTY : vec2f = vec2f(1.0, 0.0);

// The part of the ray o + t d (0 <= t <= tMax) inside the light's range and its cone (the forward
// nappe); empty when x >= y.
fn shaftConeSegment(light: KanseiSpotLight, o: vec3f, d: vec3f, tMax: f32) -> vec2f {
    let co = o - light.position;
    let b = dot(co, d);
    let disc = b * b - (dot(co, co) - light.range * light.range);
    if (disc <= 0.0) { return SHAFT_EMPTY; }
    let sq = sqrt(disc);
    var t0 = max(-b - sq, 0.0);
    var t1 = min(-b + sq, tMax);
    // in front of the light: dot(x - apex, axis) >= 0
    let a = light.direction;
    let da = dot(d, a);
    let ca = dot(co, a);
    if (abs(da) > 1e-6) {
        let tApex = -ca / da;
        if (da > 0.0) { t0 = max(t0, tApex); } else { t1 = min(t1, tApex); }
    } else if (ca < 0.0) {
        return SHAFT_EMPTY;
    }
    if (t0 >= t1) { return SHAFT_EMPTY; }
    // inside the cone: dot(x - apex, axis)^2 >= cos^2 |x - apex|^2, a quadratic in t
    let c2 = light.cosOuter * light.cosOuter;
    let qa = da * da - c2;
    let qb = 2.0 * (da * ca - c2 * b);
    let qc = ca * ca - c2 * dot(co, co);
    if (abs(qa) < 1e-7) {
        if (abs(qb) < 1e-9) { return select(SHAFT_EMPTY, vec2f(t0, t1), qc >= 0.0); }
        let tc = -qc / qb;
        if (qb > 0.0) { t0 = max(t0, tc); } else { t1 = min(t1, tc); }
        return vec2f(t0, t1);
    }
    let disc2 = qb * qb - 4.0 * qa * qc;
    if (disc2 < 0.0) {
        // never crossing the cone: inside all along (qa > 0) or never
        return select(SHAFT_EMPTY, vec2f(t0, t1), qa > 0.0);
    }
    let r = sqrt(disc2);
    let r0 = (-qb - r) / (2.0 * qa);
    let r1 = (-qb + r) / (2.0 * qa);
    let lo = min(r0, r1);
    let hi = max(r0, r1);
    if (qa < 0.0) {
        // the ray crosses the cone's side twice: inside between the crossings
        return vec2f(max(t0, lo), min(t1, hi));
    }
    // the ray runs within the cone's angle: inside past the far crossing when it goes the way the
    // light points, before the near one otherwise (the other side is the cone behind the apex)
    if (da > 0.0) { return vec2f(max(t0, hi), t1); }
    return vec2f(t0, min(t1, lo));
}

fn shaftIgn(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn shafts(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= sp.size)) { return; }
    // the front-most surface of the 2x2 pixels this texel stands for
    let full = vec2i(sp.fullSize);
    let px = vec2i(gid.xy) * 2;
    var depth = 1.0;
    var pick = px;
    for (var i = 0; i < 4; i++) {
        let q = min(px + vec2i(i & 1, i >> 1), full - 1);
        let dq = textureLoad(shaftDepthTex, q, 0);
        if (dq < depth) { depth = dq; pick = q; }
    }
    let uv = (vec2f(pick) + 0.5) / vec2f(sp.fullSize);
    let rd = shaftRay(uv);
    let cosView = max(dot(rd, sp.viewForward), 1e-3);
    var linear = sp.cameraFar;
    if (depth < 1.0) { linear = ndcToLinearDepth(depth, sp.cameraNear, sp.cameraFar); }
    let tScene = linear / cosView;

    let jitter = fract(shaftIgn(vec2f(gid.xy)) + f32(sp.frame % 64u) * 0.618034);
    let steps = max(sp.steps, 1u);
    let sliceRatio = pow(sp.gridFar / sp.gridNear, 1.0 / sp.gridD) - 1.0;
    var scatter = vec3f(0.0);
    for (var i = 0u; i < spotLights.count; i++) {
        let light = spotLights.lights[i];
        if (light.volumetricScale <= 0.0) { continue; }
        let seg = shaftConeSegment(light, sp.cameraPos, rd, tScene);
        if (seg.x >= seg.y) { continue; }
        let dt = (seg.y - seg.x) / f32(steps);
        // no closer to the lamp than its emitter, so the inverse square stays finite
        let minDist2 = max(light.sourceRadius, 0.05) * max(light.sourceRadius, 0.05);
        for (var k = 0u; k < steps; k++) {
            let t = seg.x + (f32(k) + jitter) * dt;
            let x = sp.cameraPos + rd * t;
            let lin = t * cosView;
            // the injection's medium (volumetric_fog_inject.wgsl): the height fog from its start
            // distance, and the local volumes
            let samplePos = x + params.windOffset;
            let start = saturate((lin - params.startDistance) / max(lin * sliceRatio, 1e-3) + 0.5);
            let heightFog = start * params.baseDensity * exp(-params.heightFalloff * max(samplePos.y - params.fogHeight, 0.0));
            let media = fogMedia(x, heightFog);
            if (media.density <= 0.0) { continue; }
            var s = kansei_spot_sample(light, x);
            let dl = light.position - x;
            let dist2 = max(dot(dl, dl), 1e-4);
            s.illuminance *= dist2 / max(dist2, minDist2);
            var visibility = 1.0;
            if (light.shadowLayer >= 0) {
                let coord = kansei_spot_shadow_coord(light, x);
                if (coord.w > 0.0) {
                    visibility = textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, coord.xy, light.shadowLayer, coord.z);
                }
            }
            if (visibility <= 0.0) { continue; }
            // the fog in front of the sample, from the froxel grid
            let w = clamp(depthToSlice(max(lin, sp.gridNear), sp.gridNear, sp.gridFar, sp.gridD) / sp.gridD, 0.0, 1.0);
            let transmittance = textureSampleLevel(shaftAccumTex, shaftAccumSampler, vec3f(uv, w), 0.0).a;
            let phase = henyeyGreenstein(dot(rd, s.toLight), params.anisotropy);
            scatter += s.illuminance * media.albedo * (light.volumetricScale * visibility * phase * media.density * transmittance * dt);
        }
    }
    textureStore(shaftsOut, gid.xy, vec4f(min(scatter, vec3f(60000.0)), linear));
}
