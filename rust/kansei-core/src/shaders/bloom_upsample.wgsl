struct BloomParams {
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    src_width: f32,
    src_height: f32,
    level: u32,
    _pad: u32,
};

@group(0) @binding(0) var smaller_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_sampler: sampler;
@group(0) @binding(2) var current_tex: texture_2d<f32>;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BloomParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dst_dims);
    let texel = 1.0 / vec2<f32>(textureDimensions(smaller_tex));

    // 9-tap tent filter
    var bloom = vec3<f32>(0.0);
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>(-1.0, -1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 0.0, -1.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 1.0, -1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>(-1.0,  0.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 0.0,  0.0) * texel * params.radius, 0.0).rgb * 4.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 1.0,  0.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>(-1.0,  1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 0.0,  1.0) * texel * params.radius, 0.0).rgb * 2.0;
    bloom += textureSampleLevel(smaller_tex, bloom_sampler, uv + vec2<f32>( 1.0,  1.0) * texel * params.radius, 0.0).rgb * 1.0;
    bloom /= 16.0;

    let current = textureLoad(current_tex, vec2<i32>(gid.xy), 0).rgb;
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(bloom + current, 1.0));
}
