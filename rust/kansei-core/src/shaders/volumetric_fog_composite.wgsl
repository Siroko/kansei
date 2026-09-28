// Composite: scene * transmittance + in-scattered light, looked up in the accumulated froxel
// grid at each pixel's depth (with a per-pixel slice jitter against banding), plus the spot
// lights' raymarched shafts when they are on (volumetric_fog_shafts.wgsl), upsampled from half
// resolution by depth.

struct CompositeParams {
    cameraNear   : f32,
    cameraFar    : f32,
    gridNear     : f32,
    gridFar      : f32,
    gridD        : f32,
    screenWidth  : f32,
    screenHeight : f32,
    shafts       : f32,   // 1: add the raymarched shafts
}

@group(0) @binding(0) var inputTex     : texture_2d<f32>;
@group(0) @binding(1) var depthTex     : texture_depth_2d;
@group(0) @binding(2) var outputTex    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var accumTex     : texture_3d<f32>;
@group(0) @binding(4) var accumSampler : sampler;
@group(0) @binding(5) var<uniform> cp  : CompositeParams;
@group(0) @binding(6) var shaftsTex    : texture_2d<f32>;   // rgb light, a linear depth

// The half-resolution shafts at a pixel: the four texels around it, weighted bilinearly and by
// how close their depth is to the pixel's, so shafts stop at the edges of what stands in them.
fn upsampleShafts(coord: vec2u, linearDepth: f32) -> vec3f {
    let size = vec2i(textureDimensions(shaftsTex));
    let pos = (vec2f(coord) + 0.5) * 0.5 - 0.5;
    let base = floor(pos);
    let f = pos - base;
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let s = textureLoad(shaftsTex, clamp(vec2i(base + o), vec2i(0), size - 1), 0);
        let bilinear = select(1.0 - f.x, f.x, o.x > 0.5) * select(1.0 - f.y, f.y, o.y > 0.5);
        let w = bilinear * (exp(-abs(s.a - linearDepth) / (0.05 * linearDepth + 0.1)) + 1e-4);
        sum += s.rgb * w;
        weight += w;
    }
    return sum / max(weight, 1e-6);
}

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
    var shafts = vec3f(0.0);
    if (cp.shafts > 0.5) { shafts = upsampleShafts(coord, linearDepth); }
    textureStore(outputTex, coord, vec4f(sceneColor.rgb * fog.a + fog.rgb + shafts, sceneColor.a));
}
