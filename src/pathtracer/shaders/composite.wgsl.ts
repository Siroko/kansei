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
@group(0) @binding(5) var giSampler   : sampler;
@group(0) @binding(6) var emissiveTex : texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(params.width), f32(params.height));

    let giSample  = textureSampleLevel(denoisedGI, giSampler, uv, 0.0);
    let indirect  = giSample.rgb;

    // Path tracer provides full outgoing radiance for all surface types
    let hdr = indirect;

    let final_color = hdr / (hdr + vec3f(1.0));

    textureStore(outputTex, coord, vec4f(final_color, 1.0));
}
`;
