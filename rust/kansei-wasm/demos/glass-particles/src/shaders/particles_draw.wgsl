// The particles: a unit box per instance (particles_sim.wgsl's Instance at locations 3-6),
// sized, stretched along z, turned to its velocity and placed; lit as the standard material is by
// the scene's spot lights (with their shadows) and a hemisphere of sky, plus its own glow, into the
// GBuffer's four targets and the velocity target (its motion over the last step). Prefixed by
// lights::SPOT_LIGHTS_WGSL, cameras::MOTION_VECTORS_WGSL and rt::RT_EMISSION_WGSL.

struct ParticleLook {
    roughness : f32,
    metallic  : f32,
    // the last step (s): the motion vectors' previous position is the velocity times it back
    dt        : f32,
    _pad      : f32,
    skyUp     : vec4f,
    skyDown   : vec4f,
}

@group(0) @binding(0) var<uniform> look : ParticleLook;
@group(1) @binding(0) var<uniform> view_matrix : mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix : mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix : mat4x4f;
@group(2) @binding(1) var<uniform> mesh : KanseiMeshTransforms;

struct ParticleIn {
    @location(0) position  : vec4f,
    @location(1) normal    : vec3f,
    @location(2) uv        : vec2f,
    @location(3) placement : vec4f,
    @location(4) rotation  : vec4f,
    @location(5) motion    : vec4f,
    @location(6) surface   : vec4u,
}

struct ParticleVaryings {
    @builtin(position) @invariant clip : vec4f,
    @location(0) world    : vec3f,
    @location(1) normal   : vec3f,
    @location(2) currClip : vec4f,
    @location(3) prevClip : vec4f,
    @location(4) @interpolate(flat) surface : vec2u,
}

struct ParticleOut {
    @location(0) color    : vec4f,
    @location(1) emissive : vec4f,
    @location(2) normal   : vec4f,
    @location(3) albedo   : vec4f,
    @location(4) velocity : vec2f,
}

fn rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@vertex
fn vertex_main(v: ParticleIn) -> ParticleVaryings {
    let size = v.placement.w;
    let stretch = v.motion.w;
    let local = v.position.xyz * vec3f(size, size, size * stretch);
    let p = v.placement.xyz + rotate(v.rotation, local);
    // the inverse transpose of the stretch, then the turn
    let n = rotate(v.rotation, v.normal * vec3f(1.0, 1.0, 1.0 / max(stretch, 1e-3)));
    let world = mesh.world * vec4f(p, 1.0);
    var out: ParticleVaryings;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(n, 0.0)).xyz;
    out.currClip = kansei_camera_temporal.viewProj * world;
    out.prevClip = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(p - v.motion.xyz * look.dt, 1.0));
    out.surface = v.surface.xy;
    return out;
}

@fragment
fn fragment_main(in: ParticleVaryings) -> ParticleOut {
    let N = normalize(in.normal);
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let V = normalize(eye - in.world);
    let base = unpack4x8unorm(in.surface.x).rgb;
    let glow = kansei_rt_unpack_emission(in.surface.y);
    var radiance = kansei_spot_lights_radiance(in.world, N, V, base, look.roughness, look.metallic, in.clip.xy);
    let sky = mix(look.skyDown.rgb, look.skyUp.rgb, N.y * 0.5 + 0.5);
    radiance += base * (1.0 - look.metallic) * sky + glow;
    var out: ParticleOut;
    out.color = vec4f(radiance, 1.0);
    out.emissive = vec4f(glow, 0.0);
    out.normal = vec4f(N * 0.5 + 0.5, 1.0);
    out.albedo = vec4f(base, 1.0);
    out.velocity = kansei_motion_vector(in.currClip, in.prevClip);
    return out;
}
