// Transmittance LUT lookup. Declares nothing: the including shader binds `transmittanceLut` and a
// clamping linear `lutSampler`.

// Transmittance to the top of the atmosphere from radius r along cos zenith mu (horizon-clamped).
fn transmittanceToTop(r: f32, mu: f32) -> vec3f {
    let size = vec2f(textureDimensions(transmittanceLut, 0));
    let x = transmittanceParams(r, mu);
    let uv = vec2f(unitToTexelUv(x.x, size.x), unitToTexelUv(x.y, size.y));
    return textureSampleLevel(transmittanceLut, lutSampler, uv, 0.0).rgb;
}

// Fraction of a light disk of angular radius `angularRadius` above the horizon seen from radius
// r, for a light at cos zenith mu: 1 above the horizon, 0 below, a soft terminator across it.
fn horizonVisibility(r: f32, mu: f32, angularRadius: f32) -> f32 {
    let w = max(angularRadius, 1e-4);
    return smoothstep(-w, w, mu - horizonCos(r));
}

// Transmittance between `ro` and the point `t` further along the unit ray `rd`, both inside the
// atmosphere, as a ratio of LUT values: T(ro -> top) / T(p -> top). A ray headed for the ground
// never reaches the top, so it is reversed: T(p -> top along -rd) / T(ro -> top along -rd).
fn transmittanceBetween(ro: vec3f, rd: vec3f, t: f32) -> vec3f {
    let p = ro + rd * t;
    let r0 = length(ro);
    let r1 = length(p);
    let mu0 = dot(ro, rd) / r0;
    let mu1 = dot(p, rd) / r1;
    if (rayGround(ro, rd) > 0.0) {
        return saturate(transmittanceToTop(r1, -mu1) / max(transmittanceToTop(r0, -mu0), vec3f(1e-6)));
    }
    return saturate(transmittanceToTop(r0, mu0) / max(transmittanceToTop(r1, mu1), vec3f(1e-6)));
}
