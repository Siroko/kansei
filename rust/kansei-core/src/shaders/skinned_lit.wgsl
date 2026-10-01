// A skinned mesh (SKINNING_WGSL) lit by a sun and a sky hemisphere, in physical units: the sun's
// illuminance (lux) shadowed by the renderer's cascaded shadow map when it has one, the sky's
// luminance (cd/m²). Writes motion vectors from last frame's palette and world matrix.
// `animation::SKINNED_LIT_WGSL` prepends the skinning, motion-vector and cascade chunks.

struct KanseiSkinnedSurface {
    base_color    : vec4f,   // linear albedo; w unused
    sun_direction : vec4f,   // xyz: the direction sunlight travels
    sun           : vec4f,   // rgb: colour times illuminance (lux)
    sky           : vec4f,   // rgb: zenith luminance (cd/m²)
}

@group(0) @binding(0) var<uniform> surface: KanseiSkinnedSurface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn {
    @builtin(vertex_index) vertex: u32,
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

struct VOut {
    @builtin(position) @invariant clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) curr: vec4f,
    @location(3) prev: vec4f,
};

struct FOut {
    @location(0) color: vec4f,
    @location(4) velocity: vec2f,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let s = kansei_skin(v.vertex, v.position.xyz, v.normal);
    let world = mesh.world * vec4f(s.position, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(s.normal, 0.0)).xyz;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(s.prev_position, 1.0));
    return out;
}

@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = normalize(in.normal);
    let l = -normalize(surface.sun_direction.xyz);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let albedo = surface.base_color.rgb;
    let sky = mix(surface.sky.rgb * 0.2, surface.sky.rgb, n.y * 0.5 + 0.5);
    let lit = albedo / 3.14159265 * surface.sun.rgb * max(dot(n, l), 0.0) * shadow + albedo * sky;
    return FOut(vec4f(lit, 1.0), kansei_motion_vector(in.curr, in.prev));
}
