// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Camera ──
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Group 2: Mesh (dynamic offset) ──
@group(2) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> _world_matrix: mat4x4<f32>;

// ── Lights (camera group binding 2) ──
struct DirLight {
    direction: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
};
struct PtLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
};
struct LightUniforms {
    num_directional: u32,
    num_point: u32,
    _pad0: u32,
    _pad1: u32,
    directional: array<DirLight, 4>,
    point: array<PtLight, 8>,
};
@group(1) @binding(2) var<uniform> lights: LightUniforms;

// ── Group 3: Shadows ──
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    bias: f32,
    normal_bias: f32,
    shadow_enabled: f32,
    point_shadow_enabled: f32,
    point_light_pos: vec3<f32>,
    point_shadow_far: f32,
};
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;
@group(3) @binding(3) var cube_shadow_tex: texture_2d_array<f32>;
@group(3) @binding(4) var cube_shadow_sampler: sampler;

// ── Vertex I/O ──
struct VertexInput {
    // Per-vertex (slot 0)
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance (slot 1) — mat4 as 4 vec4 rows
    @location(3) instance_mat_0: vec4<f32>,
    @location(4) instance_mat_1: vec4<f32>,
    @location(5) instance_mat_2: vec4<f32>,
    @location(6) instance_mat_3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let instance_matrix = mat4x4<f32>(
        input.instance_mat_0,
        input.instance_mat_1,
        input.instance_mat_2,
        input.instance_mat_3,
    );

    var out: VertexOutput;
    let world_pos = instance_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_normal = (instance_matrix * vec4<f32>(input.normal, 0.0)).xyz;
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
