// The smoke's history (VolumetricFogEffect::set_smoke, SmokeRenderOptions::temporal), appended
// to the march's shader (volumetric_smoke_march.wgsl) for its frame parameters: each frame's
// jittered march blended with the frames before it, separate from TAA, so the march's noise
// resolves with or without TAA. The history is reprojected through the point the smoke's light
// comes from (moved back by the gas's velocity where it has one), clamped to the current frame's
// 3x3 neighbourhood (so smoke that moved, or was stirred, leaves no ghost), and dropped where the
// scene's depth changed by more than 10% (disocclusions), on a camera cut and after a resize.

@group(0) @binding(44) var smokeCurrentTex : texture_2d<f32>;   // this frame's (R, Ts)
@group(0) @binding(45) var smokeCurrentAux : texture_2d<f32>;   // its aux
@group(0) @binding(46) var smokeHistoryTex : texture_2d<f32>;   // the previous frame's resolved (R, Ts)
@group(0) @binding(47) var smokePrevAux    : texture_2d<f32>;   // the previous frame's aux
@group(0) @binding(48) var smokeHistoryOut : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn smokeTemporal(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= sf.size)) { return; }
    let size = vec2i(sf.size);
    let current = textureLoad(smokeCurrentTex, gid.xy, 0);
    if (sf.historyValid == 0u) {
        textureStore(smokeHistoryOut, gid.xy, current);
        return;
    }
    let aux = textureLoad(smokeCurrentAux, gid.xy, 0);
    let prevUv = aux.yz;
    if (any(prevUv < vec2f(0.0)) || any(prevUv > vec2f(1.0))) {
        textureStore(smokeHistoryOut, gid.xy, current);
        return;
    }
    // a surface that was not there (more than 10% nearer or farther) keeps none of the history
    let prevTexel = clamp(vec2i(prevUv * vec2f(sf.size)), vec2i(0), size - 1);
    let prevDepth = textureLoad(smokePrevAux, prevTexel, 0).x;
    if (abs(prevDepth - aux.x) > 0.1 * aux.x) {
        textureStore(smokeHistoryOut, gid.xy, current);
        return;
    }
    var lo = current;
    var hi = current;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = textureLoad(smokeCurrentTex, clamp(vec2i(gid.xy) + vec2i(x, y), vec2i(0), size - 1), 0);
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    let history = textureSampleLevel(smokeHistoryTex, smokeSampler, prevUv, 0.0);
    let pad = (hi - lo) * 0.25;
    let clamped = clamp(history, lo - pad, hi + pad);
    textureStore(smokeHistoryOut, gid.xy, mix(clamped, current, sf.blend));
}
