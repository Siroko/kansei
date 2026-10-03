// The lightbox's walls (after VOXEL_CONES_WGSL, SCENE_WGSL and room_common.wgsl): the panel's
// light, shadowed through the volume by a cone toward it; five cones over the hemisphere for what
// the volume gathers (the other walls' bounce, the particles' glow) and, short, for the occlusion
// of corners and of the pile. `flags.x` draws the wall mirrored under the floor (the reflection).
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
    escaped  : vec3f,   // irradiance from past the volume (the gradient standing in for the ceiling)
    open     : f32,     // the cones' mean transmittance
};

// one cone along n and four at 60 degrees, each 60 degrees wide (Crassin et al. 2011)
fn hemisphere(o: vec3f, n: vec3f, maxDist: f32) -> Hemisphere {
    let t = normalize(select(cross(n, vec3f(0.0, 1.0, 0.0)), cross(n, vec3f(1.0, 0.0, 0.0)), abs(n.y) > 0.9));
    let b = cross(n, t);
    var dirs = array<vec3f, 5>(n, 0.5 * n + 0.866 * t, 0.5 * n - 0.866 * t, 0.5 * n + 0.866 * b, 0.5 * n - 0.866 * b);
    var h: Hemisphere;
    for (var k = 0u; k < 5u; k++) {
        let c = voxelConeTrace(vol, radiance, linearClamp, o, dirs[k], 0.577, vol.voxelSize, maxDist, u32(scene.coneSteps));
        let w = select(0.15, 0.25, k == 0u) / 0.85;
        h.gathered += w * PI * c.rgb;
        h.escaped += w * PI * c.a * skyRad(scene, dirs[k]);
        h.open += w * c.a;
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
        let far = hemisphere(o, n, 1e4);
        let ao = mix(1.0, hemisphere(o, n, scene.aoDistance).open, scene.aoStrength);
        color = albedo / PI * ((direct + far.escaped) * ao + far.gathered);
        if (scene.view > 0.5) {
            color = albedo / PI * (far.escaped * ao + far.gathered);
        }
    } else {
        color = albedo / PI * (direct + scene.boxSkyScale * skyIrr(scene, n));
    }
    color += surface.emission.rgb;
    if (surface.flags.x > 0.5) {
        color *= floorReflection(scene, in.world.y);
    }
    return vec4f(tonemap(scene, color), 1.0);
}
