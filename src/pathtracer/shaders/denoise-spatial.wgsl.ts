export const denoiseSpatialShader = /* wgsl */`
struct SpatialParams {
    stepSize    : u32,
    sigmaDepth  : f32,
    sigmaNormal : f32,
    sigmaLum    : f32,
    width       : u32,
    height      : u32,
    _pad        : vec2u,
}

@group(0) @binding(0) var inputGI    : texture_2d<f32>;
@group(0) @binding(1) var outputGI   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var depthTex   : texture_depth_2d;
@group(0) @binding(3) var normalTex  : texture_2d<f32>;
@group(0) @binding(4) var<uniform> params : SpatialParams;

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// 5×5 à-trous kernel weights (B3 spline)
const KERNEL_5x5 = array<f32, 25>(
    1.0/256.0, 4.0/256.0, 6.0/256.0, 4.0/256.0, 1.0/256.0,
    4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
    6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
    4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
    1.0/256.0, 4.0/256.0, 6.0/256.0, 4.0/256.0, 1.0/256.0,
);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2i(gid.xy);
    let traceSize = vec2f(f32(params.width), f32(params.height));
    let gbufDim = vec2u(textureDimensions(depthTex));
    let gbufDimF = vec2f(gbufDim);

    let centerColor = textureLoad(inputGI, coord, 0);
    let centerUV = (vec2f(gid.xy) + 0.5) / traceSize;
    let centerGBuf = min(vec2u(gbufDimF * centerUV), gbufDim - vec2u(1u));
    let centerDepth = textureLoad(depthTex, centerGBuf, 0);
    let centerNormal = normalize(textureLoad(normalTex, vec2i(centerGBuf), 0).xyz * 2.0 - 1.0);
    let centerLum = luminance(centerColor.rgb);

    // Sky pixels — pass through
    if (centerDepth >= 1.0) {
        textureStore(outputGI, vec2u(gid.xy), centerColor);
        return;
    }

    // Refractive pixels — pass through unfiltered (alpha = 0)
    if (centerColor.a < 0.5) {
        textureStore(outputGI, vec2u(gid.xy), centerColor);
        return;
    }

    var sumColor = vec3f(0.0);
    var sumWeight = 0.0;
    let step = i32(params.stepSize);

    for (var ky = -2; ky <= 2; ky++) {
        for (var kx = -2; kx <= 2; kx++) {
            let sampleCoord = coord + vec2i(kx, ky) * step;

            // Bounds check
            if (sampleCoord.x < 0 || sampleCoord.x >= i32(params.width) ||
                sampleCoord.y < 0 || sampleCoord.y >= i32(params.height)) {
                continue;
            }

            let kernelIdx = (ky + 2) * 5 + (kx + 2);
            let kernelWeight = KERNEL_5x5[kernelIdx];

            let sampleColor = textureLoad(inputGI, sampleCoord, 0);
            // Skip refractive neighbor samples — don't bleed into opaque
            if (sampleColor.a < 0.5) { continue; }
            let sampleUV = (vec2f(sampleCoord) + 0.5) / traceSize;
            let sampleGBuf = min(vec2u(gbufDimF * sampleUV), gbufDim - vec2u(1u));
            let sampleDepth = textureLoad(depthTex, sampleGBuf, 0);
            let sampleNormal = normalize(textureLoad(normalTex, vec2i(sampleGBuf), 0).xyz * 2.0 - 1.0);
            let sampleLum = luminance(sampleColor.rgb);

            // Edge-stopping: depth
            let depthDiff = abs(centerDepth - sampleDepth);
            let wDepth = exp(-depthDiff / max(params.sigmaDepth, 1e-6));

            // Edge-stopping: normal
            let normalDot = max(dot(centerNormal, sampleNormal), 0.0);
            let wNormal = pow(normalDot, params.sigmaNormal);

            // Edge-stopping: luminance
            let lumDiff = abs(centerLum - sampleLum);
            let wLum = exp(-lumDiff / max(params.sigmaLum, 1e-6));

            let w = kernelWeight * wDepth * wNormal * wLum;
            sumColor += sampleColor.rgb * w;
            sumWeight += w;
        }
    }

    let result = select(centerColor.rgb, sumColor / sumWeight, sumWeight > 1e-6);
    textureStore(outputGI, vec2u(gid.xy), vec4f(result, 1.0));
}
`;
