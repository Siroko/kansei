// Small wind ripples on still water (`simulations::fluid::WATER_RIPPLES_WGSL`): the slope (dh/dx,
// dh/dz) of a sum of six travelling waves, 13 to 53 cm long, each as steep (k a = 0.05) and
// moving at the speed of gravity waves of its length, at world position `xz` and time `t`
// (seconds). A surface of normal n becomes normalize(n - (s.x, 0, s.y) * amount) where it faces
// up. Shared by the water's glass material and the caustics' photons, so both see the same
// surface.
fn kansei_water_ripples(xz: vec2f, t: f32) -> vec2f {
    var slope = vec2f(0.0);
    // (direction angle, wavelength m)
    let waves = array<vec2f, 6>(vec2f(0.3, 0.53), vec2f(2.1, 0.37), vec2f(4.0, 0.29), vec2f(1.2, 0.21), vec2f(5.3, 0.17), vec2f(3.1, 0.13));
    for (var i = 0u; i < 6u; i++) {
        let w = waves[i];
        let d = vec2f(cos(w.x), sin(w.x));
        let k = 6.2831853 / w.y;
        let omega = sqrt(9.81 * k);
        slope += d * (0.05 * cos(k * dot(d, xz) - omega * t + f32(i) * 1.7));
    }
    return slope;
}

// Their curvature at xz (the height's second derivatives: xx, xz, zz), for the light they focus:
// a bed `depth` under them, lit from above, receives 1 / det(I - depth (1 - 1 / ior) H) of the
// light (the refraction's Jacobian), `kansei_water_focus`.
fn kansei_water_ripples_curvature(xz: vec2f, t: f32) -> vec3f {
    var h = vec3f(0.0);
    let waves = array<vec2f, 6>(vec2f(0.3, 0.53), vec2f(2.1, 0.37), vec2f(4.0, 0.29), vec2f(1.2, 0.21), vec2f(5.3, 0.17), vec2f(3.1, 0.13));
    for (var i = 0u; i < 6u; i++) {
        let w = waves[i];
        let d = vec2f(cos(w.x), sin(w.x));
        let k = 6.2831853 / w.y;
        let omega = sqrt(9.81 * k);
        // the slope's derivative along d: -k a k sin = -0.05 k sin
        let c = -0.05 * k * sin(k * dot(d, xz) - omega * t + f32(i) * 1.7);
        h += c * vec3f(d.x * d.x, d.x * d.y, d.y * d.y);
    }
    return h;
}

// The light a bed `depth` under rippled water (`amount` of the ripples, index of refraction `ior`)
// receives from straight above, relative to still water: the focus of the caustics (its lines held
// to 3x, the pattern's mean about 1).
fn kansei_water_focus(xz: vec2f, t: f32, amount: f32, depth: f32, ior: f32) -> f32 {
    let h = kansei_water_ripples_curvature(xz, t) * amount * max(depth, 0.0) * (1.0 - 1.0 / max(ior, 1.0));
    let det = (1.0 - h.x) * (1.0 - h.z) - h.y * h.y;
    // (its lines held to 3x, the whole scaled back to about the still water's mean)
    return clamp(1.0 / max(abs(det), 0.05), 0.0, 3.0) * 0.85;
}
