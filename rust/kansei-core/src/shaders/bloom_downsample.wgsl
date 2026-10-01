struct BloomParams {
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    src_width: f32,
    src_height: f32,
    level: u32,
    // scene multiplier for the threshold, so it is in the tonemapper's exposed units
    exposure: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BloomParams;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn soft_threshold(color: vec3<f32>, t: f32, k: f32) -> vec3<f32> {
    let lum = luminance(color);
    let soft = lum - t + k;
    let soft2 = clamp(soft, 0.0, 2.0 * k);
    let contrib = soft2 * soft2 / (4.0 * k + 0.0001);
    let w = max(contrib, lum - t) / max(lum, 0.0001);
    return color * max(w, 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) { return; }

    let src_dims = vec2<f32>(textureDimensions(src_tex));
    let uv_center = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dst_dims);
    let src_coord = vec2<i32>(uv_center * src_dims);

    // 3x3 tent filter. On the first level, each tap is also weighted by 1 / (1 + luma) (Karis
    // average), so a single very bright pixel can't flicker as a large bloom blob.
    let tent = array<f32, 9>(0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625);
    let max_coord = vec2<i32>(src_dims) - 1;
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = 0; i < 9; i++) {
        let offset = vec2<i32>(i % 3 - 1, i / 3 - 1);
        let tap = textureLoad(src_tex, clamp(src_coord + offset, vec2<i32>(0), max_coord), 0).rgb;
        var w = tent[i];
        if (params.level == 0u) {
            w /= 1.0 + luminance(tap) * params.exposure;
        }
        color += tap * w;
        weight_sum += w;
    }
    color /= weight_sum;

    // threshold <= 0: no threshold (physically based bloom, every light scatters a little)
    if (params.level == 0u && params.threshold > 0.0) {
        color = soft_threshold(color * params.exposure, params.threshold, params.knee) / max(params.exposure, 1e-8);
    }

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
