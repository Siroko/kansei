// The fluid surface as glass (`FluidSurfaceEffect::surface_renderable_glass`): the marching-cubes
// mesh into the GBuffer as GBUFFER_OUT_WGSL's glass, which rt::RtReflectionsEffect's glass pass
// refracts and reflects through the ray tracing grid. Where it faces up, its normal takes
// `kansei_water_ripples` (WATER_RIPPLES_WGSL, prepended with GBUFFER_OUT_WGSL), scaled by
// `ripples`, at `time`.
struct FluidGlass {
    tint      : vec3f,
    ior       : f32,
    roughness : f32,
    ripples   : f32,
    time      : f32,
    _pad      : f32,
}
@group(0) @binding(0) var<uniform> glass: FluidGlass;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) world: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    let world = world_matrix * vec4<f32>(position.xyz, 1.0);
    out.clip = projection_matrix * view_matrix * world;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    out.world = world.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    var n = normalize(in.normal);
    if (glass.ripples > 0.0) {
        let s = kansei_water_ripples(in.world.xz, glass.time) * glass.ripples * smoothstep(0.5, 0.9, abs(n.y));
        n = normalize(n - vec3f(s.x, 0.0, s.y) * sign(n.y));
    }
    return kansei_gbuffer_out_glass(n, glass.tint, glass.ior, glass.roughness);
}
