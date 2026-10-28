// The environment as the background (postprocessing::effects::EnvironmentBackgroundEffect): where
// the scene left the depth at the far plane, the environment cubemap along the pixel's view ray
// (at a roughness's mip to soften it), scaled; every other pixel passes through. Prefixed with
// SKY_ENVIRONMENT_WGSL.

struct EnvironmentBackground {
    invViewProj : mat4x4f,
    // roughness of the mip shown (0: sharp), scale of the radiance, 1 where the map is bound
    params      : vec4f,
}

@group(0) @binding(0) var<uniform> bg : EnvironmentBackground;
@group(0) @binding(1) var inputTex : texture_2d<f32>;
@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var environment : texture_cube<f32>;
@group(0) @binding(5) var environmentSampler : sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(outputTex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let colour = textureLoad(inputTex, gid.xy, 0);
    // the depth at the pixel's place in the depth buffer (it may be smaller than the output)
    let depthSize = textureDimensions(depthTex);
    let dp = min(gid.xy * depthSize / size, depthSize - 1u);
    if (bg.params.z < 0.5 || textureLoad(depthTex, dp, 0) < 1.0) {
        textureStore(outputTex, gid.xy, colour);
        return;
    }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = bg.invViewProj * vec4f(ndc, 0.0, 1.0);
    // (a point between the planes, not on the far one: an infinite projection puts it nowhere)
    let far = bg.invViewProj * vec4f(ndc, 0.5, 1.0);
    let d = normalize(far.xyz / far.w - near.xyz / near.w);
    let sky = skyEnvironment(environment, environmentSampler, d, bg.params.x) * bg.params.y;
    textureStore(outputTex, gid.xy, vec4f(sky, colour.a));
}
