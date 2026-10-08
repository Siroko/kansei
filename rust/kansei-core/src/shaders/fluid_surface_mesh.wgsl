// The fluid surface's GBuffer material (`FluidSurfaceEffect::surface_renderable`). Shared with the
// TS engine.
@group(0) @binding(0) var<uniform> color: vec4<f32>;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(position.xyz, 1.0);
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
@fragment
fn fragment_main(in: VOut) -> FOut {
    // the composite reads the normal as is, and with FluidMask::EmissiveAlpha the emissive alpha
    return FOut(color, vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(normalize(in.normal), 1.0), color);
}
