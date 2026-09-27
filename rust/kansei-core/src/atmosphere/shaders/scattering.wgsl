// In-scattering along view rays: single scattering from the sun and the moon (planet shadow,
// transmittance LUT) plus all higher orders from the multiple-scattering LUT. The including
// shader binds `frame`, `transmittanceLut`, `multiScatteringLut` and `lutSampler`.

// Light scattered toward -viewDir at planet-frame point p (radius r) by one light: `lightDir`
// points toward it and `illuminance` is its illuminance at the top of the atmosphere.
fn lightScattering(p: vec3f, r: f32, med: Medium, viewDir: vec3f, lightDir: vec3f, angularRadius: f32,
                   illuminance: vec3f) -> vec3f {
    let mu = dot(p, lightDir) / r;
    let cosTheta = dot(viewDir, lightDir);
    let phaseScattering = med.rayleighScattering * rayleighPhase(cosTheta) + med.mieScattering * miePhase(cosTheta, atm.mieG);
    let single = horizonVisibility(r, mu, angularRadius) * transmittanceToTop(r, mu) * phaseScattering;
    return illuminance * (single + multiScattering(r, mu) * med.scattering);
}

fn scatteringAt(p: vec3f, r: f32, med: Medium, viewDir: vec3f) -> vec3f {
    var s = lightScattering(p, r, med, viewDir, frame.sunDirection, frame.sunAngularRadius, frame.sunIlluminance);
    if (any(frame.moonIlluminance > vec3f(0.0))) {
        s += lightScattering(p, r, med, viewDir, frame.moonDirection, frame.moonAngularRadius, frame.moonIlluminance);
    }
    return s;
}

struct Integration {
    luminance     : vec3f,
    transmittance : vec3f,
}

// March `tMax` km from ro along the unit ray rd in `samples` segments, spaced quadratically so
// they are dense near the start where the air is thickest, integrating each segment
// analytically for its extinction (Hillaire 2020, section 5.1).
fn integrateScattering(ro: vec3f, rd: vec3f, tMax: f32, samples: u32) -> Integration {
    var result : Integration;
    result.luminance = vec3f(0.0);
    result.transmittance = vec3f(1.0);
    let n = f32(samples);
    var tPrev = 0.0;
    for (var i = 0u; i < samples; i++) {
        let f = (f32(i) + 1.0) / n;
        let tNext = tMax * f * f;
        let dt = tNext - tPrev;
        let p = ro + rd * (tPrev + 0.5 * dt);
        tPrev = tNext;
        let r = length(p);
        let med = sampleMedium(r - atm.bottomRadius);
        let s = scatteringAt(p, r, med, rd);
        let segT = exp(-med.extinction * dt);
        result.luminance += result.transmittance * (s - s * segT) / max(med.extinction, vec3f(1e-9));
        result.transmittance *= segT;
    }
    return result;
}
