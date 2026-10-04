// A skinned mesh (SKINNING_WGSL) with colour, normal and occlusion/roughness/metallic textures,
// lit by a sun (GGX specular, cascade-shadowed) and a sky hemisphere (occluded), in physical units
// as `skinned_lit.wgsl`; writes motion vectors. The uniform is `SkinnedLitParams` (its base_color
// tints the colour texture); bindings 3-6 of group 0 are the three textures and their sampler.
// `animation::SKINNED_LIT_TEXTURED_WGSL` prepends the skinning, motion-vector and cascade chunks.

struct KanseiSkinnedTexturedSurface { tint: vec4<f32>, sun_direction: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: KanseiSkinnedTexturedSurface;
@group(0) @binding(3) var base_color_texture: texture_2d<f32>;
@group(0) @binding(4) var normal_texture: texture_2d<f32>;
@group(0) @binding(5) var orm_texture: texture_2d<f32>;
@group(0) @binding(6) var texture_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn { @builtin(vertex_index) vertex: u32, @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };
struct VOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) curr: vec4<f32>,
    @location(4) prev: vec4<f32>,
};
struct FOut { @location(0) color: vec4<f32>, @location(4) velocity: vec2<f32> };

@vertex
fn vertex_main(v: VIn) -> VOut {
    let s = kansei_skin(v.vertex, v.position.xyz, v.normal);
    let world = mesh.world * vec4<f32>(s.position, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(s.normal, 0.0)).xyz;
    out.uv = v.uv;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4<f32>(s.prev_position, 1.0));
    return out;
}

// The normal map in the frame of the surface's position and uv derivatives (no stored tangents).
// glTF uv run down the image while the map's +Y is up, hence -B.
fn mapped_normal(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>, m: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let scale = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    return normalize(t * scale * m.x - b * scale * m.y + n * m.z);
}

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> FOut {
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    let albedo = textureSample(base_color_texture, texture_sampler, in.uv).rgb * surface.tint.rgb;
    let orm = textureSample(orm_texture, texture_sampler, in.uv).rgb;
    let m = textureSample(normal_texture, texture_sampler, in.uv).xyz * 2.0 - 1.0;
    n = mapped_normal(n, in.world, in.uv, m);
    let l = -normalize(surface.sun_direction.xyz);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let view = normalize(-(transpose(mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz)) * view_matrix[3].xyz) - in.world);
    let h = normalize(l + view);
    let roughness = clamp(orm.g, 0.08, 1.0);
    let a2 = roughness * roughness * roughness * roughness;
    let nh = max(dot(n, h), 0.0);
    let d = a2 / (3.14159265 * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let nl = max(dot(n, l), 0.0);
    let specular = mix(vec3<f32>(0.04), albedo, orm.b) * d * 0.25;
    let sky = mix(surface.sky.rgb * 0.2, surface.sky.rgb, n.y * 0.5 + 0.5);
    let lit = (albedo * (1.0 - orm.b) / 3.14159265 + specular) * surface.sun.rgb * nl * shadow + albedo * sky * orm.r;
    return FOut(vec4<f32>(lit, 1.0), kansei_motion_vector(in.curr, in.prev));
}
