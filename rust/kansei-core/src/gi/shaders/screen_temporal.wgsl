// Voxel GI on screen, temporal: each frame's trace (cones turned per pixel and frame) blended
// with the previous frames, reprojected by depth, the history clamped to the current
// neighbourhood so moving light and disocclusions don't ghost (as ssgi_temporal.wgsl does).

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var currentTex : texture_2d<f32>;
@group(0) @binding(2) var historyTex : texture_2d<f32>;
@group(0) @binding(3) var depthTex   : texture_depth_2d;
@group(0) @binding(4) var outTex     : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var linearSampler : sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let size = vec2i(gp.traceSize);
    let current = textureLoad(currentTex, gid.xy, 0);
    let uv = (vec2f(gid.xy) + 0.5) / gp.traceSize;
    let depth = gpDepth(gpPixel(uv));
    if (depth >= 1.0 || gp.historyValid == 0u) {
        textureStore(outTex, gid.xy, current);
        return;
    }
    let world = gp.invView * vec4f(gpViewPos(uv, depth), 1.0);
    let clip = gp.prevViewProj * world;
    let prevUv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if (clip.w <= 0.0 || any(prevUv < vec2f(0.0)) || any(prevUv > vec2f(1.0))) {
        textureStore(outTex, gid.xy, current);
        return;
    }
    // the neighbourhood's range, a little widened: the history may not leave it
    var lo = current;
    var hi = current;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = textureLoad(currentTex, clamp(vec2i(gid.xy) + vec2i(x, y), vec2i(0), size - 1), 0);
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    let pad = (hi - lo) * 0.25;
    let history = clamp(textureSampleLevel(historyTex, linearSampler, prevUv, 0.0), lo - pad, hi + pad);
    textureStore(outTex, gid.xy, mix(history, current, gp.blend));
}
