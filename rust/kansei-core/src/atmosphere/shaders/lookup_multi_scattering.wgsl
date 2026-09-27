// Multiple-scattering LUT lookup: the isotropic luminance Psi_ms of all scattering orders >= 2
// per unit illuminance (Hillaire 2020 section 5.5), at radius r for a light at cos zenith mu.
// The including shader binds `multiScatteringLut` and a clamping linear `lutSampler`.

fn multiScattering(r: f32, mu: f32) -> vec3f {
    let size = vec2f(textureDimensions(multiScatteringLut, 0));
    let x = saturate(vec2f(mu * 0.5 + 0.5, (r - atm.bottomRadius) / (atm.topRadius - atm.bottomRadius)));
    let uv = vec2f(unitToTexelUv(x.x, size.x), unitToTexelUv(x.y, size.y));
    return textureSampleLevel(multiScatteringLut, lutSampler, uv, 0.0).rgb;
}
