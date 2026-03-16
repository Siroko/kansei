export const shaderCode = /* wgsl */`
struct ShadowUniforms {
    lightViewProj       : mat4x4f,
    bias                : f32,
    normalBias          : f32,
    shadowEnabled       : f32,
    pointShadowEnabled  : f32,
    pointLightPos       : vec3f,
    pointShadowFar      : f32,
}

@group(3) @binding(0) var shadowDepthTex    : texture_depth_2d;
@group(3) @binding(1) var shadowSampler     : sampler_comparison;
@group(3) @binding(2) var<uniform> shadowUniforms : ShadowUniforms;
@group(3) @binding(3) var cubeShadowTex     : texture_2d_array<f32>;

fn calcDirectionalShadow(worldPos: vec3f, worldNormal: vec3f) -> f32 {
    let biasedPos = worldPos + worldNormal * shadowUniforms.normalBias;
    let lsPos = shadowUniforms.lightViewProj * vec4f(biasedPos, 1.0);
    let ndc = lsPos.xyz / lsPos.w;

    let uv = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

    // Clamp UVs so textureSampleCompare always runs from uniform control flow.
    let clampedUV = clamp(uv, vec2f(0.0), vec2f(1.0));

    // 1.0 when inside the shadow map, 0.0 when outside or shadows disabled.
    let inBounds = step(0.0, uv.x) * step(uv.x, 1.0)
                 * step(0.0, uv.y) * step(uv.y, 1.0)
                 * step(ndc.z, 1.0)
                 * step(0.5, shadowUniforms.shadowEnabled);

    // 3x3 PCF — always executed to satisfy uniform control flow.
    let texelSize = 1.0 / vec2f(textureDimensions(shadowDepthTex));
    let refDepth = ndc.z - shadowUniforms.bias;
    var shadow = 0.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            shadow += textureSampleCompare(
                shadowDepthTex, shadowSampler,
                clampedUV + vec2f(f32(x), f32(y)) * texelSize,
                refDepth
            );
        }
    }
    shadow /= 9.0;

    // Out-of-bounds or disabled → return 1.0 (fully lit).
    return mix(1.0, shadow, inBounds);
}

fn calcPointShadow(worldPos: vec3f) -> f32 {
    if (shadowUniforms.pointShadowEnabled < 0.5) {
        return 1.0;
    }

    let toFrag = worldPos - shadowUniforms.pointLightPos;
    let dist = length(toFrag);
    let dir = toFrag / max(dist, 1e-6);

    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);

    // Determine major axis → cubemap face index and UV
    var faceIndex : i32;
    var uv : vec2f;
    if (ax >= ay && ax >= az) {
        if (dir.x > 0.0) {
            faceIndex = 0; // +X
            uv = vec2f(-dir.z, -dir.y) / ax;
        } else {
            faceIndex = 1; // -X
            uv = vec2f(dir.z, -dir.y) / ax;
        }
    } else if (ay >= ax && ay >= az) {
        if (dir.y > 0.0) {
            faceIndex = 2; // +Y
            uv = vec2f(dir.x, dir.z) / ay;
        } else {
            faceIndex = 3; // -Y
            uv = vec2f(dir.x, -dir.z) / ay;
        }
    } else {
        if (dir.z > 0.0) {
            faceIndex = 4; // +Z
            uv = vec2f(dir.x, -dir.y) / az;
        } else {
            faceIndex = 5; // -Z
            uv = vec2f(-dir.x, -dir.y) / az;
        }
    }

    // Map [-1,1] to [0,1]
    uv = uv * 0.5 + 0.5;

    let texSize = vec2f(textureDimensions(cubeShadowTex));
    let texCoord = vec2i(clamp(uv * texSize, vec2f(0.0), texSize - 1.0));
    let storedDist = textureLoad(cubeShadowTex, texCoord, faceIndex, 0).r;
    let bias = 0.05;

    if (dist - bias > storedDist) {
        return 0.0;
    }
    return 1.0;
}
`;
