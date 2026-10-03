// p=1476's gather (gi::ParticleConeShading): each particle traces six 90-degree cones along the
// axes (light arrives from anywhere, and a particle has no normal), whose light past the volume
// is the sky, plus one narrow cone toward the sun for its soft volumetric shadow. Needs
// voxel_volume.wgsl, voxel_cones.wgsl, particle_emission.wgsl and SKY_LIGHTING_WGSL.
//
// It writes two vec4 per particle, for its material to read as instance attributes:
// - the mean radiance arriving at the particle (rgb; a Lambertian particle of albedo k under it
//   reflects k times it) and its visibility of the sun (a), blended over frames: particle
//   indices are stable, so the average needs no history volume and hides what flicker the splat
//   and the cones' jitter leave;
// - the particle's own emission (rgb), as the splat put it in the volume.
// With `useVolume` 0 it skips the cones (no volume is built): the whole sky and the sun reach
// every particle, the cheap fallback when voxel GI is off.
struct ConeParams {
    toSun          : vec3f,
    sunConeTan     : f32,   // tan of the sun cone's half aperture (~0.05)
    diffuseConeTan : f32,   // tan(45 deg) = 1: six cones tiling the sphere
    startVoxels    : f32,   // first sample this many voxels out, past the particle's own splat
    maxDistance    : f32,   // metres
    temporalBlend  : f32,   // weight of this frame in the average (1: no history)
    particleCount  : u32,
    maxSteps       : u32,
    frame          : u32,   // 0 on the first frame: no history yet
    jitter         : f32,   // voxels the cones' start moves by, per particle and frame
    useVolume      : u32,   // 0: no cones, the sky alone (voxel GI off)
    _pad0          : u32,
    _pad1          : u32,
    _pad2          : u32,
}

@group(0) @binding(0) var<uniform> vol: VoxelVolume;
@group(0) @binding(1) var<uniform> cp: ConeParams;
@group(0) @binding(2) var<uniform> emission: ParticleEmission;
@group(0) @binding(3) var<uniform> sky: SkyLighting;
@group(0) @binding(4) var radiance: texture_3d<f32>;
@group(0) @binding(5) var linearClamp: sampler;
@group(0) @binding(6) var<storage, read> positions: array<vec4f>;
@group(0) @binding(7) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> lighting: array<vec4f>;

const AXES = array<vec3f, 6>(
    vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
    vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0),
);

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= cp.particleCount) { return; }
    lighting[2u * i + 1u] = vec4f(particleEmission(emission, i, velocities[i].xyz), 0.0);
    if (cp.useVolume == 0u) {
        // voxel GI off: the whole sky arrives and the sun is unoccluded
        var sky6 = vec3f(0.0);
        for (var k = 0u; k < 6u; k++) { sky6 += skyRadiance(sky, AXES[k]); }
        lighting[2u * i] = vec4f(sky6 / 6.0, 1.0);
        return;
    }
    let p = positions[i].xyz;
    let start = (cp.startVoxels + cp.jitter * (giHash01(i * 9781u + cp.frame * 6271u) - 0.5)) * vol.voxelSize;
    var incoming = vec3f(0.0);
    for (var k = 0u; k < 6u; k++) {
        let c = voxelConeTrace(vol, radiance, linearClamp, p, AXES[k], cp.diffuseConeTan, start, cp.maxDistance, cp.maxSteps);
        incoming += c.rgb + c.a * skyRadiance(sky, AXES[k]);
    }
    incoming /= 6.0;
    let sun = voxelConeTrace(vol, radiance, linearClamp, p, cp.toSun, cp.sunConeTan, start, cp.maxDistance, cp.maxSteps).a;
    let current = vec4f(incoming, sun);
    let blend = select(cp.temporalBlend, 1.0, cp.frame == 0u);
    lighting[2u * i] = mix(lighting[2u * i], current, blend);
}
