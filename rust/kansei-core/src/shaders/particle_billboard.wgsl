// Billboard particle shader — renders instanced quads always facing the camera.
// Each instance provides a vec4 position at @location(3).
// Base geometry: a PlaneGeometry(1,1) whose XY offsets drive the billboard corners.

// ── Group 0: Material ──
struct ParticleParams {
    size: f32,
    height_min: f32,
    height_max: f32,
    _pad: f32,
    color_low: vec4<f32>,
    color_high: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: ParticleParams;

// ── Group 1: Camera ──
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Group 2: Mesh (dynamic offset) ──
@group(2) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> _world_matrix: mat4x4<f32>;

// ── Group 3: Shadows (required by engine layout, unused here) ──
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    bias: f32,
    normal_bias: f32,
    shadow_enabled: f32,
    point_shadow_enabled: f32,
    point_light_pos: vec3<f32>,
    point_shadow_far: f32,
};
@group(3) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;
@group(3) @binding(3) var cube_shadow_tex: texture_2d_array<f32>;
@group(3) @binding(4) var cube_shadow_sampler: sampler;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) instance_pos: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let s = params.size;
    let cam_right = vec3<f32>(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let cam_up    = vec3<f32>(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);

    let world_pos = input.instance_pos.xyz
        + cam_right * input.position.x * s
        + cam_up    * input.position.y * s;

    var out: VertexOutput;
    out.clip_pos = projection_matrix * view_matrix * vec4<f32>(world_pos, 1.0);

    let height_range = params.height_max - params.height_min;
    let t = clamp((input.instance_pos.y - params.height_min) / height_range, 0.0, 1.0);
    out.color = mix(params.color_low.rgb, params.color_high.rgb, t);
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
