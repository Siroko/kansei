// Spot-light shafts, temporal: each frame's jittered march blended with the previous frames,
// reprojected by the pixel's depth, the history clamped to the current neighbourhood (so a moving
// lamp does not smear its shafts) and dropped where the depth changed (disocclusions).

@group(0) @binding(0) var<uniform> sp : ShaftParams;
@group(0) @binding(1) var currentTex : texture_2d<f32>;
@group(0) @binding(2) var historyTex : texture_2d<f32>;
@group(0) @binding(3) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var linearSampler : sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= sp.size)) { return; }
    let size = vec2i(sp.size);
    let current = textureLoad(currentTex, gid.xy, 0);
    if (sp.historyValid == 0u) {
        textureStore(outTex, gid.xy, current);
        return;
    }
    // where this texel's surface was on the previous frame
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(sp.size);
    let rd = shaftRay(uv);
    let world = sp.cameraPos + rd * (current.a / max(dot(rd, sp.viewForward), 1e-3));
    let clip = sp.prevViewProj * vec4f(world, 1.0);
    let prevUv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if (clip.w <= 0.0 || any(prevUv < vec2f(0.0)) || any(prevUv > vec2f(1.0))) {
        textureStore(outTex, gid.xy, current);
        return;
    }
    var lo = current.rgb;
    var hi = current.rgb;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = textureLoad(currentTex, clamp(vec2i(gid.xy) + vec2i(x, y), vec2i(0), size - 1), 0).rgb;
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    let history = textureSampleLevel(historyTex, linearSampler, prevUv, 0.0);
    // a surface that was not there (more than 10 % nearer or farther) keeps none of the history
    if (abs(history.a - current.a) > 0.1 * current.a) {
        textureStore(outTex, gid.xy, current);
        return;
    }
    let pad = (hi - lo) * 0.25;
    let clamped = clamp(history.rgb, lo - pad, hi + pad);
    textureStore(outTex, gid.xy, vec4f(mix(clamped, current.rgb, sp.blend), current.a));
}
