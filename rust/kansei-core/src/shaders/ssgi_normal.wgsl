// The GBuffer's normals, for the passes that bind `normalTex`.

// The GBuffer's world normal (stored n * 0.5 + 0.5; zero where the material wrote none).
fn ssgiWorldNormal(px: vec2i) -> vec4f {
    let raw = textureLoad(normalTex, px, 0).xyz;
    if (dot(raw, raw) < 1e-4) { return vec4f(0.0); }
    return vec4f(normalize(raw * 2.0 - 1.0), 1.0);
}
