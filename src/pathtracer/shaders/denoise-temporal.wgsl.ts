export const denoiseTemporalShader = /* wgsl */`
struct TemporalParams {
    currentInvViewProj : mat4x4f,
    prevViewProj       : mat4x4f,
    blendFactor        : f32,
    width              : f32,
    height             : f32,
    frameIndex         : u32,
}

@group(0) @binding(0) var currentGI   : texture_2d<f32>;
@group(0) @binding(1) var historyGI   : texture_2d<f32>;
@group(0) @binding(2) var outputGI    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex    : texture_depth_2d;
@group(0) @binding(4) var normalTex   : texture_2d<f32>;
@group(0) @binding(5) var historySamp  : sampler;
@group(0) @binding(6) var<uniform> params : TemporalParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = vec2i(gid.xy);
    let size = vec2f(params.width, params.height);
    let isize = vec2i(i32(params.width), i32(params.height));
    if (f32(gid.x) >= params.width || f32(gid.y) >= params.height) { return; }

    let current = textureLoad(currentGI, coord, 0);
    let depth = textureLoad(depthTex, vec2u(gid.xy), 0);

    // Sky pixels — no history accumulation
    if (depth >= 1.0) {
        textureStore(outputGI, vec2u(gid.xy), current);
        return;
    }

    // Refractive pixels — pass through unfiltered (alpha = 0)
    if (current.a < 0.5) {
        textureStore(outputGI, vec2u(gid.xy), current);
        return;
    }

    // ── Neighborhood statistics (3×3) for color clamping ──
    // This prevents ghosting from moving objects: history samples outside
    // the current neighborhood range are clamped before blending.
    var nMin = current.rgb;
    var nMax = current.rgb;
    var nMean = vec3f(0.0);
    var nCount = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nc = coord + vec2i(dx, dy);
            if (nc.x < 0 || nc.x >= isize.x || nc.y < 0 || nc.y >= isize.y) { continue; }
            let s = textureLoad(currentGI, nc, 0).rgb;
            nMin = min(nMin, s);
            nMax = max(nMax, s);
            nMean += s;
            nCount += 1.0;
        }
    }
    nMean /= nCount;

    // Widen the clamp box slightly around the mean to allow some convergence
    let nExtent = nMax - nMin;
    let clampMin = nMin - nExtent * 0.1;
    let clampMax = nMax + nExtent * 0.1;

    // ── Reproject to previous frame ──
    let uv = (vec2f(gid.xy) + 0.5) / size;
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.currentInvViewProj * ndc;
    let worldPos = wp.xyz / wp.w;

    let prevClip = params.prevViewProj * vec4f(worldPos, 1.0);
    let prevNDC = prevClip.xyz / prevClip.w;
    let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

    // Out-of-bounds or first frame → use current only
    var blend = params.blendFactor;
    if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        blend = 1.0;
    }
    if (params.frameIndex == 0u) {
        blend = 1.0;
    }

    // Sample history with bilinear filtering
    let history = textureSampleLevel(historyGI, historySamp, prevUV, 0.0);

    // Clamp history to current neighborhood range (anti-ghosting)
    let clamped = vec4f(clamp(history.rgb, clampMin, clampMax), 1.0);

    // Increase blend toward current when history was clamped significantly
    let clampDist = length(history.rgb - clamped.rgb);
    let adaptiveBlend = max(blend, saturate(clampDist * 0.5));

    let result = mix(clamped, current, adaptiveBlend);
    textureStore(outputGI, vec2u(gid.xy), result);
}
`;
