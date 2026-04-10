// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
    specular: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Camera ──
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Group 2: Mesh (dynamic offset) ──
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

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

fn calcDirectionalShadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if (shadow_uniforms.shadow_enabled < 0.5) {
        return 1.0;
    }

    let biased_pos = world_pos + world_normal * shadow_uniforms.normal_bias;
    let ls_pos = shadow_uniforms.light_view_proj * vec4<f32>(biased_pos, 1.0);
    let ndc = ls_pos.xyz / ls_pos.w;

    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

    let in_bounds = step(0.0, uv.x) * step(uv.x, 1.0)
                  * step(0.0, uv.y) * step(uv.y, 1.0)
                  * step(ndc.z, 1.0);

    // PCF must run in uniform control flow — no early return before textureSampleCompare
    let tex_size = vec2<f32>(textureDimensions(shadow_depth_tex));
    let texel_size = 1.0 / tex_size;
    let ref_depth = ndc.z - shadow_uniforms.bias;
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    var shadow = 0.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let sample_uv = clamp(
                clamped_uv + vec2<f32>(f32(x), f32(y)) * texel_size,
                vec2<f32>(0.0), vec2<f32>(1.0)
            );
            shadow += textureSampleCompare(
                shadow_depth_tex, shadow_sampler,
                sample_uv, ref_depth
            );
        }
    }
    // Out-of-bounds → fully lit (1.0); in-bounds → PCF result
    return mix(1.0, shadow / 9.0, in_bounds);
}

fn calcPointShadow(world_pos: vec3<f32>) -> f32 {
    if (shadow_uniforms.point_shadow_enabled < 0.5) {
        return 1.0;
    }

    let to_frag = world_pos - shadow_uniforms.point_light_pos;
    let dist = length(to_frag);
    let dir = to_frag / max(dist, 1e-6);

    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);

    var face_index: i32;
    var uv: vec2<f32>;

    if (ax >= ay && ax >= az) {
        if (dir.x > 0.0) {
            face_index = 0; uv = vec2<f32>(-dir.z, -dir.y) / ax;
        } else {
            face_index = 1; uv = vec2<f32>(dir.z, -dir.y) / ax;
        }
    } else if (ay >= ax && ay >= az) {
        if (dir.y > 0.0) {
            face_index = 2; uv = vec2<f32>(dir.x, dir.z) / ay;
        } else {
            face_index = 3; uv = vec2<f32>(dir.x, -dir.z) / ay;
        }
    } else {
        if (dir.z > 0.0) {
            face_index = 4; uv = vec2<f32>(dir.x, -dir.y) / az;
        } else {
            face_index = 5; uv = vec2<f32>(-dir.x, -dir.y) / az;
        }
    }

    uv = uv * 0.5 + 0.5;

    let tex_size = vec2<f32>(textureDimensions(cube_shadow_tex));
    let tex_coord = vec2<i32>(clamp(uv * tex_size, vec2<f32>(0.0), tex_size - 1.0));
    let stored_dist = textureLoad(cube_shadow_tex, tex_coord, face_index, 0).r;

    let bias = 0.05;
    if (dist - bias > stored_dist) {
        return 0.0;
    }
    return 1.0;
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

    let dir_shadow = calcDirectionalShadow(input.world_position, n);
    let point_shadow = calcPointShadow(input.world_position);
    let shadow = dir_shadow * point_shadow;
    let final_color = base_color * (ambient + diffuse * shadow) + specular * shadow;
    return vec4<f32>(final_color, material.color.a);
}
