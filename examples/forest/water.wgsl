// The lake (after forest.wgsl): a dark, still surface at the lake's level, the terrain hiding it
// on land. It mirrors the sky's SH by its Fresnel, or with reflect=1 hands its F0 and roughness to
// the ray-traced reflections, which mirror the forest on the far shore.
struct Water { albedo: vec4<f32>, specular: vec4<f32> }; // specular: x F0, y roughness, z 1 = traced
@group(0) @binding(0) var<uniform> water: Water;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world = in.position.xyz;
    out.clip = projection_matrix * view_matrix * vec4<f32>(out.world, 1.0);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> KanseiGBufferOut {
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let p = in.world;
    let v = normalize(view_eye() - p);
    let l = -kansei_cascades.lightDirection;
    let sun = kansei_cascades.lightColor * kansei_sun_shadow(p, n, in.clip.xy);
    let albedo = water.albedo.rgb;
    let f0 = water.specular.x;
    let rough = water.specular.y;
    let color = albedo / PI * (sun * max(l.y, 0.0) + sky_light(p, n)) + sun * ggx_specular(n, v, l, max(rough, 0.08), f0);
    if (water.specular.z > 0.5) {
        return kansei_gbuffer_out_specular(color, vec3<f32>(0.0), n, albedo, f0, rough);
    }
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(v.y, 0.0), 5.0);
    let r = reflect(-v, n);
    return kansei_gbuffer_out(color + skyRadiance(sky, r) * fresnel * sky_visibility(p, n), vec3<f32>(0.0), n, albedo);
}

@fragment
fn voxel_main(in: VertexOutput, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, water.albedo.rgb, vec3<f32>(0.0));
}
