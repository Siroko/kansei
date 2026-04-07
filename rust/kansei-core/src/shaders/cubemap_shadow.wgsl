// Cubemap shadow pass — renders linear distance from point light per face.
// Output: r32float distance stored in a 2D-array texture (6 layers per light).

struct LightUniform {
    light_view_proj: mat4x4<f32>,
    light_world_pos: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> light: LightUniform;
@group(1) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn shadow_vs(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VSOut {
    let wp = world_matrix * position;
    var out: VSOut;
    out.position = light.light_view_proj * wp;
    out.world_pos = wp.xyz;
    return out;
}

@fragment
fn shadow_fs(in: VSOut) -> @location(0) f32 {
    return length(in.world_pos - light.light_world_pos);
}
