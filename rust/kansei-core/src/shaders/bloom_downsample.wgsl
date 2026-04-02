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

    // 13-tap box filter
    var color = vec3<f32>(0.0);
    color += textureLoad(src_tex, src_coord + vec2<i32>(-1, -1), 0).rgb * 0.0625;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 0, -1), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 1, -1), 0).rgb * 0.0625;
    color += textureLoad(src_tex, src_coord + vec2<i32>(-1,  0), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 0,  0), 0).rgb * 0.25;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 1,  0), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>(-1,  1), 0).rgb * 0.0625;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 0,  1), 0).rgb * 0.125;
    color += textureLoad(src_tex, src_coord + vec2<i32>( 1,  1), 0).rgb * 0.0625;

    if (params.level == 0u) {
        color = soft_threshold(color, params.threshold, params.knee);
    }

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
