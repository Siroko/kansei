export const compositeShader = /* wgsl */`
struct CompositeParams {
    width  : u32,
    height : u32,
    _pad   : vec2u,
}

@group(0) @binding(0) var inputTex    : texture_2d<f32>;
@group(0) @binding(1) var denoisedGI  : texture_2d<f32>;
@group(0) @binding(2) var albedoTex   : texture_2d<f32>;
@group(0) @binding(3) var outputTex   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params : CompositeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let direct   = textureLoad(inputTex,   vec2i(coord), 0).rgb;
    let indirect = textureLoad(denoisedGI, vec2i(coord), 0).rgb;
    let albedo   = textureLoad(albedoTex,  vec2i(coord), 0).rgb;

    // Direct already has albedo baked in from rasterizer.
    // Indirect is raw incoming radiance — multiply by albedo.
    let final_color = direct + albedo * indirect;

    textureStore(outputTex, coord, vec4f(final_color, 1.0));
}
`;
