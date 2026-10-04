// The GBuffer's normals, for the passes that bind `normalTex` (and `depthTex`).

// The GBuffer's world normal (stored n * 0.5 + 0.5; zero where the material wrote none).
fn gpWorldNormal(px: vec2i) -> vec4f {
    let raw = textureLoad(normalTex, px, 0).xyz;
    if (dot(raw, raw) < 1e-4) { return vec4f(0.0); }
    return vec4f(normalize(raw * 2.0 - 1.0), 1.0);
}

// A world normal: the GBuffer's, or one rebuilt from the depth around the pixel.
fn surfaceNormal(px: vec2i, p: vec3f) -> vec3f {
    let n = gpWorldNormal(px);
    if (n.w > 0.0) { return n.xyz; }
    let uv = (vec2f(px) + 0.5) / gp.fullSize;
    let du = vec2f(1.0 / gp.fullSize.x, 0.0);
    let dv = vec2f(0.0, 1.0 / gp.fullSize.y);
    let r = gpViewPos(uv + du, gpDepth(px + vec2i(1, 0)));
    let l = gpViewPos(uv - du, gpDepth(px - vec2i(1, 0)));
    let d = gpViewPos(uv + dv, gpDepth(px + vec2i(0, 1)));
    let u = gpViewPos(uv - dv, gpDepth(px - vec2i(0, 1)));
    // the side with the smaller depth step, so edges don't bend the normal
    let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
    let dy = select(p - d, u - p, abs(u.z - p.z) < abs(p.z - d.z));
    var nv = normalize(cross(dx, dy));
    if (dot(nv, -p) < 0.0) { nv = -nv; }
    return normalize((gp.invView * vec4f(nv, 0.0)).xyz);
}
