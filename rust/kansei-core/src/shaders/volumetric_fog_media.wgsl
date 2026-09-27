// Fog media beyond the height fog, appended to volumetric_fog_inject.wgsl (after the sky lighting
// helpers): a scattering albedo, local fog volumes, and ambient light from the sky.
//
// Local fog volumes are ellipsoids (as Unreal's LocalFogVolume: mist over a lake, in a hollow)
// with two terms in the volume's unit sphere q (|q| < 1):
//   radial: radialExtinction * (1 - |q|^2), densest at the centre
//   height: heightExtinction * exp(-heightFalloff * max(q.y - heightOffset, 0)), lying low
// both faded to zero over the outer `edgeFade` of the radius. They take the fog's lights and
// ambient with their own albedo, and neither the wind nor the start distance moves them.

struct FogMediaParams {
    albedo          : vec3f,   // height fog: scattering = density * albedo
    skyAmbientScale : f32,
    numVolumes      : u32,
    hasSkyLighting  : u32,
    _pad0           : u32,
    _pad1           : u32,
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
}

@group(0) @binding(10) var<uniform> mediaParams : FogMediaParams;
@group(0) @binding(11) var<storage, read> fogVolumes : array<LocalFogVolume>;
@group(0) @binding(12) var<uniform> skyLighting : SkyLighting;

struct FogMedia {
    density    : f32,     // what the light terms are scaled by
    extinction : f32,
    albedo     : vec3f,   // scattering / density, rgb
}

fn localFogExtinction(v: LocalFogVolume, worldPos: vec3f) -> f32 {
    let d = worldPos - v.center;
    // into the unit sphere: undo the yaw (about +Y), then divide by the radii
    let q = vec3f(v.cosYaw * d.x - v.sinYaw * d.z, d.y, v.sinYaw * d.x + v.cosYaw * d.z) * v.invRadii;
    let r2 = dot(q, q);
    if (r2 >= 1.0) { return 0.0; }
    var edge = 1.0;
    if (v.edgeFade > 0.0) {
        let t = saturate((1.0 - sqrt(r2)) / v.edgeFade);
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

// Sky light scattered toward the camera per unit scattering coefficient, with the fog's phase.
fn skyAmbient(viewDir: vec3f) -> vec3f {
    if (mediaParams.hasSkyLighting == 0u) { return vec3f(0.0); }
    return skyInscatter(skyLighting, viewDir, params.anisotropy) * mediaParams.skyAmbientScale;
}
