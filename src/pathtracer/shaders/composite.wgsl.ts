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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(params.width), f32(params.height));

    let direct    = textureLoad(inputTex, vec2i(coord), 0).rgb;
    // Sample low-res GI with bilinear filtering for smooth upscale
    let giSample  = textureSampleLevel(denoisedGI, giSampler, uv, 0.0);
    let indirect  = giSample.rgb;
    let albedoRaw = textureLoad(albedoTex, vec2i(coord), 0);
    let albedo    = albedoRaw.rgb;
    let alpha     = albedoRaw.a;

    // alpha == 0 → transparent/refractive: use indirect as-is (already colored by path tracer)
    // alpha == 1 → opaque: direct has albedo baked in, indirect is raw radiance * albedo
    let hdr = select(indirect, direct + albedo * indirect, alpha > 0.5);

    // Reinhard tone map to make HDR range visible
    let final_color = hdr / (hdr + vec3f(1.0));

    textureStore(outputTex, coord, vec4f(final_color, 1.0));
}
`;
