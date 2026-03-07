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
    let direct    = textureLoad(inputTex,   vec2i(coord), 0).rgb;
    let indirect  = textureLoad(denoisedGI, vec2i(coord), 0).rgb;
    let albedoRaw = textureLoad(albedoTex,  vec2i(coord), 0);
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
