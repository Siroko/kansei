// Shadow pass — depth-only vertex shader
@group(0) @binding(0) var<uniform> light_view_proj: mat4x4<f32>;

@group(2) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

@vertex
fn shadow_vs(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> @builtin(position) vec4<f32> {
    return light_view_proj * world_matrix * position;
}
