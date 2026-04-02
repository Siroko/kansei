// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Mesh (dynamic offset) ──
@group(1) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// ── Group 2: Camera ──
@group(2) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Vertex I/O ──
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = world_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_normal = (normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    out.uv = input.uv;
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(input.world_normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let ambient = 0.15;
    return vec4<f32>(material.color.rgb * (ambient + ndotl * 0.85), material.color.a);
}
