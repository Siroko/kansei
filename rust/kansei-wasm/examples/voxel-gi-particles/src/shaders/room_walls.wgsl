// The lightbox's walls (after VOXEL_CONES_WGSL, SKY_LIGHTING_WGSL, SCENE_WGSL and
// room_common.wgsl): the panel's light, shadowed through the volume by a cone toward it; five cones
// over the hemisphere for what the volume gathers (the other walls' bounce, the particles' glow),
// their first metres also the occlusion of corners and of the pile; the gradient sky
// (ParticleGi::sky_buffer) standing in for the ceiling past the volume. `flags.x` draws the wall mirrored under the floor (the
// reflection).
struct Surface {
    albedo   : vec4f,
    emission : vec4f,   // scene radiance (the panel)
    flags    : vec4f,   // x: mirrored
};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(0) @binding(1) var<uniform> scene: SceneParams;
@group(0) @binding(2) var<uniform> vol: VoxelVolume;
@group(0) @binding(3) var radiance: texture_3d<f32>;
@group(0) @binding(4) var linearClamp: sampler;
@group(0) @binding(5) var<uniform> sky: SkyLighting;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
};

@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> VOut {
    let world = world_matrix * vec4f(position.xyz, 1.0);
    var shown = world.xyz;
    if (surface.flags.x > 0.5) {
        shown.y = 2.0 * scene.mirrorY - shown.y;
    }
    var out: VOut;
    out.clip = projection_matrix * view_matrix * vec4f(shown, 1.0);
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(normal, 0.0)).xyz;
    return out;
}

struct Hemisphere {
    gathered : vec3f,   // irradiance from what the cones met in the volume
    escaped  : vec3f,   // irradiance from past the volume (the sky standing in for the ceiling)
    open     : f32,     // the cones' mean transmittance
};

// one cone along n and four at 60 degrees round it (VOXEL_CONES_WGSL's voxelHemisphereCone),
// each split at `nearDist`: `open` is their mean transmittance there, so the walls' occlusion
// cones are the first metres of their light cones
fn hemisphere(o: vec3f, n: vec3f, nearDist: f32) -> Hemisphere {
    var h: Hemisphere;
    for (var k = 0u; k < VOXEL_HEMISPHERE_CONES; k++) {
        let cone = voxelHemisphereCone(n, k);
        let s = voxelConeTraceSplit(vol, radiance, linearClamp, o, cone.xyz, VOXEL_HEMISPHERE_TAN, vol.voxelSize, nearDist, 1e4, u32(scene.coneSteps));
        h.gathered += cone.w * s.far.rgb;
        h.escaped += cone.w * s.far.a * skyRadiance(sky, cone.xyz);
        h.open += cone.w / PI * s.nearOpen;
    }
    return h;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let albedo = surface.albedo.rgb;
    var direct = panelIrradiance(scene, in.world, n);
    var color = vec3f(0.0);
    if (scene.giOn > 0.5) {
        // start out of the wall's own voxels
        let o = in.world + n * vol.voxelSize;
        if (any(direct > vec3f(0.0))) {
            let aim = vec3f(clamp(o.x, scene.panelMin.x, scene.panelMax.x), scene.panelMin.y, clamp(o.z, scene.panelMin.z, scene.panelMax.z));
            direct *= voxelConeTrace(vol, radiance, linearClamp, o, normalize(aim - o), scene.panelConeTan, 0.5 * vol.voxelSize, 1e4, u32(scene.coneSteps)).a;
        }
        let cones = hemisphere(o, n, scene.aoDistance);
        let ao = mix(1.0, cones.open, scene.aoStrength);
        color = albedo / PI * ((direct + cones.escaped) * ao + cones.gathered);
        if (scene.view > 0.5) {
            color = albedo / PI * (cones.escaped * ao + cones.gathered);
        }
    } else {
        color = albedo / PI * (direct + scene.boxSkyScale * skyIrradiance(sky, n));
    }
    color += surface.emission.rgb;
    if (surface.flags.x > 0.5) {
        color *= floorReflection(scene, in.world.y);
    }
    return vec4f(tonemap(scene, color), 1.0);
}
