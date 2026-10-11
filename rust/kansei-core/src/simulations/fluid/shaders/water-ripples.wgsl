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

// The cells' edges of one animated cellular layer at p (in cells): 0 on an edge, about the
// distance to it (in cells) off one. Each cell's point circles its own centre.
fn kansei_water_cell_edge(p: vec2f, t: f32) -> f32 {
    let i = floor(p);
    let f = p - i;
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let g = vec2f(f32(x), f32(y));
            let c = i + g;
            let h = fract(sin(vec2f(dot(c, vec2f(127.1, 311.7)), dot(c, vec2f(269.5, 183.3)))) * 43758.547);
            let o = 0.5 + 0.38 * sin(t + 6.2831853 * h);
            let r = g + o - f;
            let d = dot(r, r);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    return sqrt(d2) - sqrt(d1);
}

// A game's pool caustics at world xz, time t: a net of bright curved lines (0 to 1, about 0.2 on
// average), two cellular layers 34 and 23 cm across, drifting and bent by four slow waves. Not
// the ripples' optics (`kansei_water_focus`): what reads as water caustics at a glance, crisp at
// any distance, for about 20 cells' hashes a pixel.
fn kansei_water_caustic_lines(xz: vec2f, t: f32) -> f32 {
    let bend = vec2f(sin(xz.y * 7.3 + t * 1.1) + 0.6 * sin(xz.x * 13.7 + xz.y * 5.1 - t * 1.7), sin(xz.x * 6.1 - t * 0.9) + 0.6 * sin(xz.y * 12.3 - xz.x * 4.7 + t * 1.5)) * 0.06;
    let a = kansei_water_cell_edge((xz + bend) / 0.34 + vec2f(0.11, -0.07) * t, t * 1.3);
    let b = kansei_water_cell_edge(mat2x2f(0.8, 0.6, -0.6, 0.8) * (xz - bend * 0.7) / 0.23 + vec2f(-0.09, 0.05) * t, t * 1.7 + 2.0);
    // thin lines where either layer's edge passes, brightest where they cross
    let la = 1.0 - smoothstep(0.0, 0.11, a);
    let lb = 1.0 - smoothstep(0.0, 0.13, b);
    return clamp(la * la * 0.75 + lb * lb * 0.55 + la * lb * 0.8, 0.0, 1.0);
}
