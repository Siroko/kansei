// Kansei's standard lit material (materials::StandardLitOptions, Material::standard_lit): a GGX
// / Lambert surface (lights/spot_lights.wgsl's kansei_brdf) lit by every light in the scene:
// the directional lights (the sun through the cascaded shadow map, or the single map), the point
// lights (the shadow-casting one through its cube shadow), the spot lights with their shadows, a hemisphere of sky, and its own emission. It writes
// the GBuffer's four targets (and the velocity target with `outputs_velocity`). Prefixed by
// LIGHTS_WGSL, SHADOW_MAP_WGSL, CASCADED_SHADOWS_WGSL, SPOT_LIGHTS_WGSL (and MOTION_VECTORS_WGSL);
// the Rust side fills the KANSEI_* placeholders.

struct KanseiStandardSurface {
    baseColor : vec4f,   // rgb albedo (linear)
    emissive  : vec4f,   // rgb emitted radiance (cd/m²)
    skyUp     : vec4f,   // rgb sky radiance from straight up (cd/m²)
    skyDown   : vec4f,   // rgb radiance from straight down (the ground's bounce)
    params    : vec4f,   // roughness, metallic
}

@group(0) @binding(0) var<uniform> kansei_surface : KanseiStandardSurface;
@group(1) @binding(0) var<uniform> view_matrix : mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix : mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix : mat4x4f;
KANSEI_WORLD_BINDING

struct KanseiStandardIn {
    @location(0) position : vec4f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    KANSEI_INSTANCE_INPUT
}

struct KanseiStandardVaryings {
    @builtin(position) @invariant clip : vec4f,
    @location(0) world  : vec3f,
    @location(1) normal : vec3f,
    KANSEI_VELOCITY_VARYINGS
}

struct KanseiStandardOut {
    @location(0) color    : vec4f,
    @location(1) emissive : vec4f,
    @location(2) normal   : vec4f,
    @location(3) albedo   : vec4f,
    KANSEI_VELOCITY_OUTPUT
}

@vertex
fn vertex_main(v: KanseiStandardIn) -> KanseiStandardVaryings {
    var local = v.position.xyz;
    var normal = v.normal;
    KANSEI_INSTANCE_PLACE
    let world = KANSEI_WORLD * vec4f(local, 1.0);
    var out: KanseiStandardVaryings;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(normal, 0.0)).xyz;
    KANSEI_VELOCITY_VERTEX
    return out;
}

@fragment
fn fragment_main(in: KanseiStandardVaryings) -> KanseiStandardOut {
    let N = normalize(in.normal);
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let V = normalize(eye - in.world);
    let base = kansei_surface.baseColor.rgb;
    let roughness = kansei_surface.params.x;
    let metallic = kansei_surface.params.y;

    // the sun's shadow: the cascades when they are on, else the single map (first directional light)
    let cascades = kansei_cascades.count > 0u;
    let sunShadow = select(kansei_shadow_map(in.world, N), kansei_sun_shadow(in.world, N, in.clip.xy), cascades);
    var radiance = vec3f(0.0);
    for (var i = 0u; i < kansei_lights.num_directional; i++) {
        let light = kansei_lights.directional[i];
        let L = -normalize(light.direction);
        let isSun = select(i == 0u, dot(normalize(light.direction), normalize(kansei_cascades.lightDirection)) > 0.9999, cascades);
        radiance += kansei_brdf(N, V, L, base, roughness, metallic) * light.color * select(1.0, sunShadow, isSun);
    }
    for (var i = 0u; i < kansei_lights.num_point; i++) {
        let light = kansei_lights.point[i];
        let L = normalize(light.position - in.world);
        let shadow = select(1.0, kansei_point_shadow(in.world), kansei_is_point_shadow_light(light.position));
        radiance += kansei_brdf(N, V, L, base, roughness, metallic) * light.color * kansei_point_falloff(light, in.world) * shadow;
    }
    radiance += kansei_spot_lights_radiance(in.world, N, V, base, roughness, metallic, in.clip.xy);
    let sky = mix(kansei_surface.skyDown.rgb, kansei_surface.skyUp.rgb, N.y * 0.5 + 0.5);
    let emissive = kansei_surface.emissive.rgb;
    radiance += base * (1.0 - metallic) * sky + emissive;

    var out: KanseiStandardOut;
    out.color = vec4f(radiance, 1.0);
    // alpha 0: FluidMask::EmissiveAlpha takes an alpha of 0.5 and over for the fluid's surface
    out.emissive = vec4f(emissive, 0.0);
    out.normal = vec4f(N * 0.5 + 0.5, 1.0);
    out.albedo = vec4f(base, 1.0);
    KANSEI_VELOCITY_FRAGMENT
    return out;
}
