struct CompositeParams {
    width        : u32,
    height       : u32,
    rasterDirect : u32,
    _pad         : u32,
}

@group(0) @binding(0) var inputTex    : texture_2d<f32>;
@group(0) @binding(1) var denoisedGI  : texture_2d<f32>;
@group(0) @binding(2) var albedoTex   : texture_2d<f32>;
@group(0) @binding(3) var outputTex   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params : CompositeParams;
@group(0) @binding(5) var giSampler   : sampler;
@group(0) @binding(6) var emissiveTex : texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn composite_main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(params.width), f32(params.height));

    let albedo   = textureLoad(albedoTex, vec2i(coord), 0).rgb;
    let emissive = textureLoad(emissiveTex, vec2i(coord), 0).rgb;

    let giSample = textureSampleLevel(denoisedGI, giSampler, uv, 0.0);
    let indirect = giSample.rgb;

    var hdr: vec3f;
    if (params.rasterDirect != 0u) {
        // Raster direct mode: rasterized scene has direct light, GI has probe indirect only.
        // output = GI * albedo + direct + emissive
        let direct = textureLoad(inputTex, vec2i(coord), 0).rgb;
        hdr = indirect * albedo + direct + emissive;
    } else {
        // Full path tracer mode: GI buffer has complete radiance.
        hdr = indirect * albedo + emissive;
    }

    // Reinhard tone mapping
    let final_color = hdr / (hdr + vec3f(1.0));

    textureStore(outputTex, coord, vec4f(final_color, 1.0));
}
