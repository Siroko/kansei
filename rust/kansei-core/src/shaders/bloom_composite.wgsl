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

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_sampler: sampler;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: BloomParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(u32(params.src_width), u32(params.src_height));
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let scene = textureLoad(scene_tex, vec2<i32>(gid.xy), 0).rgb;
    let bloom = textureSampleLevel(bloom_tex, bloom_sampler, uv, 0.0).rgb;
    let result = scene + bloom * params.intensity;

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
