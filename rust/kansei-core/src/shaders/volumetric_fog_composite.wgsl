// Composite: scene * transmittance + in-scattered light, looked up in the accumulated froxel
// grid at each pixel's depth (with a per-pixel slice jitter against banding).

struct CompositeParams {
    cameraNear   : f32,
    cameraFar    : f32,
    gridNear     : f32,
    gridFar      : f32,
    gridD        : f32,
    screenWidth  : f32,
    screenHeight : f32,
    _pad         : f32,
}

@group(0) @binding(0) var inputTex     : texture_2d<f32>;
@group(0) @binding(1) var depthTex     : texture_depth_2d;
@group(0) @binding(2) var outputTex    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var accumTex     : texture_3d<f32>;
@group(0) @binding(4) var accumSampler : sampler;
@group(0) @binding(5) var<uniform> cp  : CompositeParams;

fn screenHash(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    if (f32(coord.x) >= cp.screenWidth || f32(coord.y) >= cp.screenHeight) { return; }

    let sceneColor = textureLoad(inputTex, coord, 0);
    let depth      = textureLoad(depthTex, coord, 0);

    let linearDepth = ndcToLinearDepth(depth, cp.cameraNear, cp.cameraFar);
    let sliceFloat = depthToSlice(linearDepth, cp.gridNear, cp.gridFar, cp.gridD);

    let jitter = screenHash(vec2f(coord)) - 0.5;
    let w = clamp((sliceFloat + jitter) / cp.gridD, 0.0, 1.0);
    let uv = (vec2f(coord) + 0.5) / vec2f(cp.screenWidth, cp.screenHeight);

    let fog = textureSampleLevel(accumTex, accumSampler, vec3f(uv, w), 0.0);
    textureStore(outputTex, coord, vec4f(sceneColor.rgb * fog.a + fog.rgb, sceneColor.a));
}
