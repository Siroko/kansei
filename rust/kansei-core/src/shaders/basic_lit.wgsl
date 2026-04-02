// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
    specular: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Mesh (dynamic offset) ──
@group(1) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// ── Group 2: Camera ──
@group(2) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

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
@group(2) @binding(2) var<uniform> lights: LightUniforms;

// ── Group 3: Shadows ──
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    bias: f32,
    normal_bias: f32,
    shadow_enabled: f32,
    _pad: f32,
};
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;

// ── Vertex I/O ──
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = world_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = (normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    out.uv = input.uv;
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);

    // Extract camera position from view matrix inverse:
    // camera_pos = -(transpose(mat3(view)) * view[3].xyz)
    let view3 = mat3x3<f32>(
        view_matrix[0].xyz,
        view_matrix[1].xyz,
        view_matrix[2].xyz,
    );
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);

    let view_dir = normalize(camera_pos - input.world_position);

    let base_color = material.color.rgb;
    let spec_color = material.specular.rgb;
    let shininess = material.specular.a * 256.0;

    let ambient = vec3<f32>(0.08, 0.08, 0.08);
    var diffuse = vec3<f32>(0.0, 0.0, 0.0);
    var specular = vec3<f32>(0.0, 0.0, 0.0);

    // Directional lights
    for (var i = 0u; i < lights.num_directional; i = i + 1u) {
        let dl = lights.directional[i];
        let light_dir = normalize(-dl.direction);
        let ndotl = max(dot(n, light_dir), 0.0);

        let half_vec = normalize(light_dir + view_dir);
        let ndoth = max(dot(n, half_vec), 0.0);
        let spec = pow(ndoth, shininess);

        diffuse += dl.color * ndotl;
        specular += dl.color * spec_color * spec;
    }

    // Point lights
    for (var i = 0u; i < lights.num_point; i = i + 1u) {
        let pl = lights.point[i];
        let to_light = pl.position - input.world_position;
        let dist = length(to_light);
        let light_dir = to_light / dist;

        let atten_linear = max(1.0 - dist / pl.radius, 0.0);
        let atten = atten_linear * atten_linear;

        let ndotl = max(dot(n, light_dir), 0.0);

        let half_vec = normalize(light_dir + view_dir);
        let ndoth = max(dot(n, half_vec), 0.0);
        let spec = pow(ndoth, shininess);

        diffuse += pl.color * ndotl * atten;
        specular += pl.color * spec_color * spec * atten;
    }

    let final_color = base_color * (ambient + diffuse) + specular;
    return vec4<f32>(final_color, material.color.a);
}
