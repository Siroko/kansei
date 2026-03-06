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
    if (f32(gid.x) >= params.width || f32(gid.y) >= params.height) { return; }

    let current = textureLoad(currentGI, coord, 0);
    let depth = textureLoad(depthTex, vec2u(gid.xy), 0);

    // Sky pixels — no history accumulation
    if (depth >= 1.0) {
        textureStore(outputGI, vec2u(gid.xy), current);
        return;
    }

    // Reconstruct world position from current depth
    let uv = (vec2f(gid.xy) + 0.5) / size;
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.currentInvViewProj * ndc;
    let worldPos = wp.xyz / wp.w;

    // Reproject to previous frame
    let prevClip = params.prevViewProj * vec4f(worldPos, 1.0);
    let prevNDC = prevClip.xyz / prevClip.w;
    let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

    // Check if reprojected UV is in bounds
    if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        textureStore(outputGI, vec2u(gid.xy), current);
        return;
    }

    // Sample history
    let history = textureSampleLevel(historyGI, historySamp, prevUV, 0.0);

    // Disocclusion rejection based on normal consistency
    let currentNormal = textureLoad(normalTex, coord, 0).xyz * 2.0 - 1.0;
    let prevCoord = vec2i(vec2f(prevUV) * size);
    let prevNormal = textureLoad(normalTex, prevCoord, 0).xyz * 2.0 - 1.0;
    let normalDot = dot(normalize(currentNormal), normalize(prevNormal));

    // Reject history if normals diverge significantly
    var blend = params.blendFactor;
    if (normalDot < 0.8) {
        blend = 1.0; // reject history
    }

    // First frame has no valid history
    if (params.frameIndex == 0u) {
        blend = 1.0;
    }

    let result = mix(history, current, blend);
    textureStore(outputGI, vec2u(gid.xy), result);
}
`;
