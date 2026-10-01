// Temporal reprojection: blend this frame's injected froxels with last frame's history,
// reprojected through the previous view-projection.

struct TemporalParams {
    currentInvVP : mat4x4f,
    prevVP       : mat4x4f,
    gridNear     : f32,
    gridFar      : f32,
    cameraNear   : f32,
    cameraFar    : f32,
    gridW        : u32,
    gridH        : u32,
    gridD        : u32,
    blendFactor  : f32,
    hasPrevFrame : u32,
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,
}

@group(0) @binding(0) var currentTex  : texture_3d<f32>;
@group(0) @binding(1) var historyIn   : texture_3d<f32>;
@group(0) @binding(2) var historySamp : sampler;
@group(0) @binding(3) var historyOut  : texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<uniform> tp : TemporalParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= tp.gridW || gid.y >= tp.gridH || gid.z >= tp.gridD) { return; }

    let current = textureLoad(currentTex, gid, 0);
    let gridSize = vec3f(f32(tp.gridW), f32(tp.gridH), f32(tp.gridD));
    let worldPos = froxelToWorld(vec3f(f32(gid.x), f32(gid.y), f32(gid.z) + 0.5), gridSize,
                                 tp.gridNear, tp.gridFar, tp.cameraNear, tp.cameraFar, tp.currentInvVP);

    // reproject into the previous frame's froxel coordinates
    let prevClip = tp.prevVP * vec4f(worldPos, 1.0);
    let prevNDC = prevClip.xyz / prevClip.w;
    let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 0.5 - prevNDC.y * 0.5);
    let prevLinearD = ndcToLinearDepth(prevNDC.z, tp.cameraNear, tp.cameraFar);
    let prevSlice = depthToSlice(prevLinearD, tp.gridNear, tp.gridFar, gridSize.z);
    let prevUVW = vec3f(prevUV, prevSlice / gridSize.z);

    let valid = all(prevUVW >= vec3f(0.0)) && all(prevUVW <= vec3f(1.0))
             && prevClip.w > 0.0 && tp.hasPrevFrame != 0u;

    let history = textureSampleLevel(historyIn, historySamp, prevUVW, 0.0);

    // 100 % current on the first frame, after a cut, or when the froxel was off-screen
    let alpha = select(1.0, tp.blendFactor, valid);
    textureStore(historyOut, gid, mix(history, current, alpha));
}
