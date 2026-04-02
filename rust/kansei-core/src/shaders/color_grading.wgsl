struct GradeParams {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    temperature: f32,
    tint: f32,
    highlights: f32,
    shadows: f32,
    black_point: f32,
    screen_w: f32,
    screen_h: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: GradeParams;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(u32(params.screen_w), u32(params.screen_h));
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    var color = textureLoad(input_tex, vec2<i32>(gid.xy), 0).rgb;

    // Black point lift
    if (params.black_point > 0.0) {
        color = max(color - vec3<f32>(params.black_point), vec3<f32>(0.0)) / (1.0 - params.black_point);
    }

    // Shadows/highlights
    let lum = luminance(color);
    let sh_mix = smoothstep(0.0, 1.0, lum);
    let sh_mul = mix(params.shadows, params.highlights, sh_mix);
    color *= sh_mul;

    // Brightness
    color += vec3<f32>(params.brightness);

    // Contrast (pivot at 0.5)
    color = (color - vec3<f32>(0.5)) * params.contrast + vec3<f32>(0.5);

    // Saturation
    let gray = luminance(color);
    color = mix(vec3<f32>(gray), color, params.saturation);

    // Temperature
    color.r += params.temperature * 0.1;
    color.b -= params.temperature * 0.1;

    // Tint
    color.g += params.tint * 0.1;

    color = max(color, vec3<f32>(0.0));
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
