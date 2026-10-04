// Wave line rendering: each vertex is a point along a horizontal line.
// Positions are updated by the FFT elevation compute shader.
// Rendered as line-strip topology — one continuous line per text row.

@group(0) @binding(0) var<uniform> lineColor: vec4<f32>;

@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    out.clip_pos = projection_matrix * view_matrix * world_matrix * position;
    return out;
}

@fragment
fn fragment_main(v: VOut) -> @location(0) vec4<f32> {
    return lineColor;
}
