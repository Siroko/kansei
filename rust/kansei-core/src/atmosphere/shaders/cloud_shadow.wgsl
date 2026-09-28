// Cloud shadows: the cloud layer's transmittance toward the sun, for the sun's direct light on the
// scene. VolumetricCloudsEffect marches it every frame into a map projected along the sun onto a
// horizontal plane under the camera; every point below the clouds reads it by following the sun's
// ray to that plane. Bind `SkyAtmosphereBindings::cloud_shadow` (texture_2d<f32>), a linear
// clamping sampler (`lut_sampler`) and `cloud_shadow_params` (uniform CloudShadowParams), and
// multiply the sun's light by `cloudShadow(...)`. It is 1 without clouds (or with their shadows
// off), with the sun down, and beyond the map, towards whose edge it fades to 1.

struct CloudShadowParams {
    center  : vec2f,   // world xz of the map's centre
    invSize : f32,     // 1 / the map's side (m)
    planeY  : f32,     // world height of the plane it lies on
    sunDir  : vec3f,   // toward the sun
    enabled : f32,     // 0: no cloud shadows
}

fn cloudShadow(map: texture_2d<f32>, mapSampler: sampler, p: CloudShadowParams, worldPos: vec3f) -> f32 {
    if (p.enabled < 0.5 || p.sunDir.y <= 1e-3) { return 1.0; }
    let onPlane = worldPos + p.sunDir * ((p.planeY - worldPos.y) / p.sunDir.y);
    let q = (onPlane.xz - p.center) * p.invSize;
    let t = textureSampleLevel(map, mapSampler, q + 0.5, 0.0).r;
    return mix(t, 1.0, smoothstep(0.4, 0.5, max(abs(q.x), abs(q.y))));
}
