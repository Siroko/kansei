// Sky lighting for materials and participating media: the sky's radiance around the camera as
// order-2 spherical harmonics (the clouds in front of it included), and the sun and the moon
// after the atmosphere. Written every frame
// by SkyAtmosphere::update; bind it as a uniform (SkyAtmosphereBindings::sky_lighting).
//
// The SH basis is the real, orthonormal one in world coordinates:
//   0: 0.282095                1: 0.488603 y    2: 0.488603 z    3: 0.488603 x
//   4: 1.092548 xy             5: 1.092548 yz   6: 0.315392 (3z^2 - 1)
//   7: 1.092548 xz             8: 0.546274 (x^2 - y^2)

struct SkyLighting {
    sh              : array<vec4f, 9>,   // radiance (rgb) arriving from each direction
    sunIlluminance  : vec4f,             // rgb at the camera, after the atmosphere; w = visibility
    sunDirection    : vec4f,             // xyz toward the sun
    moonIlluminance : vec4f,
    moonDirection   : vec4f,
    clearSkyUp      : vec4f,             // irradiance on an upward surface from the sky without its clouds
    distantSkyLight : vec4f,             // the sky's mean radiance from 6 km up (Unreal's distant sky light)
}

fn skyShBasis(d: vec3f) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
        1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y));
}

// Sum of sh[i] * basis[i] * band[l(i)]: the SH convolved with a zonal kernel given per band.
fn skyShEval(sky: SkyLighting, d: vec3f, band: vec3f) -> vec3f {
    let c = sky.sh[0].rgb * (0.282095 * band.x)
          + (sky.sh[1].rgb * d.y + sky.sh[2].rgb * d.z + sky.sh[3].rgb * d.x) * (0.488603 * band.y)
          + (sky.sh[4].rgb * (1.092548 * d.x * d.y) + sky.sh[5].rgb * (1.092548 * d.y * d.z)
             + sky.sh[6].rgb * (0.315392 * (3.0 * d.z * d.z - 1.0)) + sky.sh[7].rgb * (1.092548 * d.x * d.z)
             + sky.sh[8].rgb * (0.546274 * (d.x * d.x - d.y * d.y))) * band.z;
    return max(c, vec3f(0.0));
}

// Irradiance on a surface facing n from the whole sky (Ramamoorthi and Hanrahan 2001). A
// Lambertian surface reflects albedo / pi times this.
fn skyIrradiance(sky: SkyLighting, n: vec3f) -> vec3f {
    return skyShEval(sky, n, vec3f(3.141593, 2.094395, 0.785398));
}

// The sky's radiance arriving from direction d, low-pass (for rough reflections).
fn skyRadiance(sky: SkyLighting, d: vec3f) -> vec3f {
    return skyShEval(sky, d, vec3f(1.0));
}

// Sky light a medium with Henyey-Greenstein phase g scatters toward the camera, per unit
// scattering coefficient, at a point seen along viewDir (camera to point): the SH of the sky
// convolved with the phase function, whose zonal coefficients are g^l.
fn skyInscatter(sky: SkyLighting, viewDir: vec3f, g: f32) -> vec3f {
    return skyShEval(sky, viewDir, vec3f(1.0, g, g * g));
}
