// Fog media beyond the height fog, appended to volumetric_fog_inject.wgsl (after the sky lighting
// helpers): a scattering albedo, local fog volumes, and ambient light from the sky.
//
// Local fog volumes are ellipsoids (as Unreal's LocalFogVolume: mist over a lake, in a hollow) or
// boxes, with two terms in the volume's unit shape q (|q| < 1, or max |q_i| < 1 for a box; r is
// that norm):
//   radial: radialExtinction * (1 - r^2), densest at the centre
//   height: heightExtinction * exp(-heightFalloff * max(q.y - heightOffset, 0)), lying low
// both faded to zero over the outer `edgeFade` of the radius. They take the fog's lights and
// ambient with their own albedo, and neither the wind nor the start distance moves them.

struct FogMediaParams {
    albedo           : vec3f,   // height fog: scattering = density * albedo
    skyAmbientScale  : f32,
    numVolumes       : u32,
    hasSkyLighting   : u32,
    hasClipmapProbes : u32,     // 1: the ambient from a voxel clipmap's probes where they reach
    _pad1            : u32,
}

struct LocalFogVolume {
    center           : vec3f,
    radialExtinction : f32,    // per metre
    invRadii         : vec3f,
    heightExtinction : f32,    // per metre
    albedo           : vec3f,
    heightFalloff    : f32,    // per unit of the volume's half height
    cosYaw           : f32,
    sinYaw           : f32,
    heightOffset     : f32,    // in the unit sphere: -1 bottom .. 1 top
    edgeFade         : f32,    // fraction of the radius
    shape            : u32,    // 0 ellipsoid, 1 box
    _pad0            : u32,
    _pad1            : u32,
    _pad2            : u32,
}

@group(0) @binding(10) var<uniform> mediaParams : FogMediaParams;
@group(0) @binding(11) var<storage, read> fogVolumes : array<LocalFogVolume>;
@group(0) @binding(12) var<uniform> skyLighting : SkyLighting;
// how much of the sky each point sees (SkyOcclusion; off, it is 1 everywhere)
@group(0) @binding(18) var skyOcclusionVolume : texture_3d<f32>;
@group(0) @binding(19) var skyOcclusionSampler : sampler;
@group(0) @binding(20) var<uniform> skyOcclusion : SkyOcclusionParams;
// a voxel clipmap's irradiance probes (gi::ClipmapProbes; stand-ins without them)
@group(0) @binding(21) var<uniform> kansei_clip_probe_grid : ClipProbeGrid;
@group(0) @binding(22) var<storage, read> kansei_clip_probes : array<vec4f>;

struct FogMedia {
    density    : f32,     // what the light terms are scaled by
    extinction : f32,
    albedo     : vec3f,   // scattering / density, rgb
}

fn localFogExtinction(v: LocalFogVolume, worldPos: vec3f) -> f32 {
    let d = worldPos - v.center;
    // into the unit sphere: undo the yaw (about +Y), then divide by the radii
    let q = vec3f(v.cosYaw * d.x - v.sinYaw * d.z, d.y, v.sinYaw * d.x + v.cosYaw * d.z) * v.invRadii;
    var r = length(q);
    if (v.shape == 1u) { r = max(abs(q.x), max(abs(q.y), abs(q.z))); }
    if (r >= 1.0) { return 0.0; }
    let r2 = r * r;
    var edge = 1.0;
    if (v.edgeFade > 0.0) {
        let t = saturate((1.0 - r) / v.edgeFade);
        edge = t * t * (3.0 - 2.0 * t);
    }
    let radial = v.radialExtinction * (1.0 - r2);
    let height = v.heightExtinction * exp(-v.heightFalloff * max(q.y - v.heightOffset, 0.0));
    return (radial + height) * edge;
}

// The height fog (density `heightFog`) plus every local volume at worldPos.
fn fogMedia(worldPos: vec3f, heightFog: f32) -> FogMedia {
    var density = heightFog;
    var extinction = heightFog * params.extinctionCoeff;
    var scattering = heightFog * mediaParams.albedo;
    for (var i = 0u; i < mediaParams.numVolumes; i++) {
        let v = fogVolumes[i];
        let e = localFogExtinction(v, worldPos);
        density += e;
        extinction += e;
        scattering += e * v.albedo;
    }
    var m : FogMedia;
    m.density = density;
    m.extinction = extinction;
    m.albedo = select(mediaParams.albedo, scattering / max(density, 1e-12), density > 0.0);
    return m;
}

// Sky light scattered toward the camera per unit scattering coefficient, with the fog's phase,
// at worldPos: dimmed by how much of the sky it sees. With a voxel clipmap's probes, where they
// reach, the light they gather in its place: the sky past the trees and what the scene bounces.
fn skyAmbient(viewDir: vec3f, worldPos: vec3f) -> vec3f {
    if (mediaParams.hasClipmapProbes != 0u) {
        let probes = kansei_clipmap_inscatter(worldPos, viewDir, params.anisotropy);
        if (probes.a >= 0.0) { return probes.rgb * mediaParams.skyAmbientScale; }
    }
    if (mediaParams.hasSkyLighting == 0u) { return vec3f(0.0); }
    let visibility = skyVisibility(skyOcclusionVolume, skyOcclusionSampler, skyOcclusion, worldPos);
    return skyInscatter(skyLighting, viewDir, params.anisotropy) * (mediaParams.skyAmbientScale * visibility);
}
