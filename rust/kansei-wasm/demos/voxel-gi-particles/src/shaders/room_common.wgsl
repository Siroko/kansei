// The lightbox look (after SCENE_WGSL): a closed white room lit by an emissive panel in its
// ceiling, shared by the walls and the particles.

// Lambert's formula for a polygon (the vector form factor; Arvo 1995): the irradiance the panel
// (a Lambertian rectangle facing down, radiance panelRadiance) gives a surface at p facing n,
// unoccluded and clamped where the panel sinks below the surface's horizon.
fn panelIrradiance(s: SceneParams, p: vec3f, n: vec3f) -> vec3f {
    let y = s.panelMin.y;
    var corners = array<vec3f, 4>(
        vec3f(s.panelMin.x, y, s.panelMin.z),
        vec3f(s.panelMax.x, y, s.panelMin.z),
        vec3f(s.panelMax.x, y, s.panelMax.z),
        vec3f(s.panelMin.x, y, s.panelMax.z),
    );
    var f = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let a = normalize(corners[k] - p);
        let b = normalize(corners[(k + 1u) % 4u] - p);
        let g = cross(b, a);
        let len = length(g);
        if (len > 1e-6) {
            f += acos(clamp(dot(a, b), -1.0, 1.0)) * dot(g / len, n);
        }
    }
    return s.panelRadiance * max(0.5 * f, 0.0);
}

// The panel's radiance seen from p along d (zero off it), its edges softened with distance as a
// slightly rough reflection would blur them.
fn panelSeen(s: SceneParams, p: vec3f, d: vec3f) -> vec3f {
    if (d.y <= 1e-3) { return vec3f(0.0); }
    let t = (s.panelMin.y - p.y) / d.y;
    if (t <= 0.0) { return vec3f(0.0); }
    let h = p + d * t;
    let e = 0.03 * t + 0.05;
    let inside = smoothstep(-e, e, h.x - s.panelMin.x) * smoothstep(-e, e, s.panelMax.x - h.x)
        * smoothstep(-e, e, h.z - s.panelMin.z) * smoothstep(-e, e, s.panelMax.z - h.z);
    return s.panelRadiance * inside;
}

fn fresnel(f0: f32, cosTheta: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
}

// how much of a point's colour the glossy floor under the box reflects
fn floorReflection(s: SceneParams, y: f32) -> f32 {
    return s.reflectivity * exp(-max(y - s.mirrorY, 0.0) / s.reflectFade);
}
