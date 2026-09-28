// The cloud map: the clouds around the camera in every direction, written each frame by
// VolumetricCloudsEffect and read by the sky lighting and the environment cubemap, so the
// scene's ambient light and reflections see the clouds. rgb the light the clouds send toward the
// camera (after the air in front of them), a their opacity (1 - transmittance): an empty map is
// a clear sky. u is the azimuth about +Y, v the zenith angle (0 straight up, 1 straight down).

fn cloudMapDirection(uv: vec2f) -> vec3f {
    let phi = (uv.x - 0.5) * 2.0 * PI;
    let theta = uv.y * PI;
    return vec3f(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

fn cloudMapUv(d: vec3f) -> vec2f {
    return vec2f(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}
