// The floor: floor_grid.wgsl's technical grid on dark grey, anti-aliased by the pixel's footprint
// (screen-space derivatives), lit as the standard material is by the spot lights (with their
// shadows) and a hemisphere of sky; polished, so rt::RtReflectionsEffect reflects the room in it.
// Prefixed by lights::SPOT_LIGHTS_WGSL, materials::GBUFFER_OUT_WGSL and floor_grid.wgsl.

@group(0) @binding(0) var<uniform> grid : FloorGrid;
@group(1) @binding(0) var<uniform> view_matrix : mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix : mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix : mat4x4f;
@group(2) @binding(1) var<uniform> world_matrix : mat4x4f;

struct FloorVaryings {
    @builtin(position) @invariant clip : vec4f,
    @location(0) world  : vec3f,
    @location(1) normal : vec3f,
}

@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> FloorVaryings {
    let world = world_matrix * vec4f(position.xyz, 1.0);
    var out: FloorVaryings;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in: FloorVaryings) -> KanseiGBufferOut {
    let N = normalize(in.normal);
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let V = normalize(eye - in.world);

    let p = in.world.xz;
    let footprint = vec2f(length(vec2f(dpdx(p.x), dpdy(p.x))), length(vec2f(dpdx(p.y), dpdy(p.y))));
    let albedo = floorGridAlbedo(grid, p, footprint);

    let roughness = grid.widths.w;
    var radiance = kansei_spot_lights_radiance(in.world, N, V, albedo, roughness, 0.0, in.clip.xy);
    radiance += albedo * mix(grid.skyDown.rgb, grid.skyUp.rgb, N.y * 0.5 + 0.5);
    return kansei_gbuffer_out_specular(radiance, vec3f(0.0), N, albedo, grid.specular.x, roughness);
}
