// Kansei's gradient sky (Material::gradient_sky): a dome whose radiance runs from the horizon up
// to the zenith and down to the ground, by the direction of each point from the mesh's centre.
// Prefixed by GBUFFER_OUT_WGSL.

struct KanseiGradientSky {
    zenith  : vec4f,   // rgb radiance straight up (cd/m²); w: the curve's exponent
    horizon : vec4f,
    ground  : vec4f,
}

@group(0) @binding(0) var<uniform> kansei_sky : KanseiGradientSky;
@group(1) @binding(0) var<uniform> view_matrix : mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix : mat4x4f;
@group(2) @binding(1) var<uniform> world_matrix : mat4x4f;

struct KanseiSkyVaryings {
    @builtin(position) @invariant clip : vec4f,
    @location(0) direction : vec3f,
}

@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> KanseiSkyVaryings {
    var out: KanseiSkyVaryings;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4f(position.xyz, 1.0);
    out.direction = position.xyz;
    return out;
}

@fragment
fn fragment_main(in: KanseiSkyVaryings) -> KanseiGBufferOut {
    let d = normalize(in.direction);
    let up = pow(saturate(d.y), kansei_sky.zenith.w);
    let down = sqrt(saturate(-d.y));
    let sky = select(mix(kansei_sky.horizon.rgb, kansei_sky.ground.rgb, down), mix(kansei_sky.horizon.rgb, kansei_sky.zenith.rgb, up), d.y >= 0.0);
    return kansei_gbuffer_out(sky, vec3f(0.0), -d, vec3f(0.0));
}
