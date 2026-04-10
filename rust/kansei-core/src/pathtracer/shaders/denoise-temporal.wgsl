struct TemporalParams {
    currentInvViewProj : mat4x4f,
    prevViewProj       : mat4x4f,
    blendFactor        : f32,
    width              : f32,
    height             : f32,
    frameIndex         : u32,
}

@group(0) @binding(0) var currentGI    : texture_2d<f32>;
@group(0) @binding(1) var historyGI    : texture_2d<f32>;
@group(0) @binding(2) var outputGI     : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex     : texture_depth_2d;
@group(0) @binding(4) var normalTex    : texture_2d<f32>;
@group(0) @binding(5) var historySamp  : sampler;
@group(0) @binding(6) var<uniform> params : TemporalParams;
@group(0) @binding(7) var momentsHistory : texture_2d<f32>;
@group(0) @binding(8) var momentsOutput  : texture_storage_2d<rgba16float, write>;

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn temporal_main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = vec2i(gid.xy);
    let size = vec2f(params.width, params.height);
    let isize = vec2i(i32(params.width), i32(params.height));
    if (f32(gid.x) >= params.width || f32(gid.y) >= params.height) { return; }

    let current = textureLoad(currentGI, coord, 0);
    let curLum = luminance(current.rgb);

    // Map trace-res coord to full-res GBuffer coord
    let uv_t = (vec2f(gid.xy) + 0.5) / size;
    let gbufDim = vec2u(textureDimensions(depthTex));
    let gbufCoord = min(vec2u(vec2f(gbufDim) * uv_t), gbufDim - vec2u(1u));
    let depth = textureLoad(depthTex, gbufCoord, 0);

    // Sky pixels
    if (depth >= 1.0) {
        textureStore(outputGI, vec2u(gid.xy), current);
        textureStore(momentsOutput, vec2u(gid.xy), vec4f(0.0));
        return;
    }

    // -- Spatial variance fallback (3x3) for short histories --
    var nMin = current.rgb;
    var nMax = current.rgb;
    var nMean = vec3f(0.0);
    var spatialM2 = 0.0;
    var nCount = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nc = coord + vec2i(dx, dy);
            if (nc.x < 0 || nc.x >= isize.x || nc.y < 0 || nc.y >= isize.y) { continue; }
            let s = textureLoad(currentGI, nc, 0).rgb;
            nMin = min(nMin, s);
            nMax = max(nMax, s);
            nMean += s;
            let sLum = luminance(s);
            spatialM2 += sLum * sLum;
            nCount += 1.0;
        }
    }
    nMean /= nCount;
    let spatialMeanLum = luminance(nMean);
    let spatialVariance = max(spatialM2 / nCount - spatialMeanLum * spatialMeanLum, 0.0);

    // Widen the clamp box slightly around the mean
    let nExtent = nMax - nMin;
    let clampMin = nMin - nExtent * 0.1;
    let clampMax = nMax + nExtent * 0.1;

    // -- Reproject to previous frame --
    let uv = (vec2f(gid.xy) + 0.5) / size;
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.currentInvViewProj * ndc;
    let worldPos = wp.xyz / wp.w;

    let prevClip = params.prevViewProj * vec4f(worldPos, 1.0);
    let prevNDC = prevClip.xyz / prevClip.w;
    let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

    var disoccluded = false;
    if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        disoccluded = true;
    }
    if (params.frameIndex == 0u) {
        disoccluded = true;
    }

    // Normal consistency: reject history if normals diverge too much
    let centerNormal = normalize(textureLoad(normalTex, vec2i(gbufCoord), 0).xyz * 2.0 - 1.0);
    if (!disoccluded) {
        let prevGBuf = min(vec2u(vec2f(gbufDim) * prevUV), gbufDim - vec2u(1u));
        let prevNormal = normalize(textureLoad(normalTex, vec2i(prevGBuf), 0).xyz * 2.0 - 1.0);
        if (dot(centerNormal, prevNormal) < 0.5) {
            disoccluded = true;
        }
    }

    var blendedColor: vec3f;
    var moment1: f32;
    var moment2: f32;
    var historyLen: f32;
    var variance: f32;

    if (disoccluded) {
        blendedColor = current.rgb;
        moment1 = curLum;
        moment2 = curLum * curLum;
        historyLen = 1.0;
        variance = spatialVariance;
    } else {
        let history = textureSampleLevel(historyGI, historySamp, prevUV, 0.0);
        let prevMoments = textureSampleLevel(momentsHistory, historySamp, prevUV, 0.0);

        // Clamp history to current neighborhood range (anti-ghosting)
        let clamped = clamp(history.rgb, clampMin, clampMax);
        let clampDist = length(history.rgb - clamped);
        let adaptiveBlend = max(params.blendFactor, saturate(clampDist * 0.5));

        blendedColor = mix(clamped, current.rgb, adaptiveBlend);

        // Accumulate moments
        let prevHistLen = prevMoments.b;
        historyLen = min(prevHistLen + 1.0, 256.0);
        let alpha = max(1.0 / historyLen, params.blendFactor);
        moment1 = mix(prevMoments.r, curLum, alpha);
        moment2 = mix(prevMoments.g, curLum * curLum, alpha);
        variance = max(moment2 - moment1 * moment1, 0.0);

        // For short history, blend with spatial variance estimate
        if (historyLen < 4.0) {
            variance = mix(spatialVariance, variance, historyLen / 4.0);
        }
    }

    textureStore(outputGI, vec2u(gid.xy), vec4f(blendedColor, current.a));
    textureStore(momentsOutput, vec2u(gid.xy), vec4f(moment1, moment2, historyLen, variance));
}
