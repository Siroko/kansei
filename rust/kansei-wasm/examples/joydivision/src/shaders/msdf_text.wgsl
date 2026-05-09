// MSDF text instanced rendering shader.
// Each instance is a single glyph quad positioned by the FFT compute shader.

// ── Group 0: Material (MSDF atlas) ──
@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_samp: sampler;

// ── Group 1: Camera ──
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Group 2: Mesh ──
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// ── Helpers ──

fn remap(x: f32, lo: f32, hi: f32, oLo: f32, oHi: f32) -> f32 {
    return oLo + (x - lo) * (oHi - oLo) / (hi - lo);
}

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

// ── Vertex ──

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_color: vec4<f32>,
};

@vertex
fn vertex_main(
    // Per-vertex (from PlaneGeometry)
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance
    @location(3) particlePos: vec4<f32>,
    @location(4) imageBounds: vec4<f32>,  // UV rect: left, top, right, bottom
    @location(5) planeBounds: vec4<f32>,  // glyph rect: left, top, right, bottom (scaled by font_size)
    @location(6) color: vec4<f32>,
) -> VOut {
    // PlaneGeometry vertices are in -0.5..0.5 range.
    // Map to 0..1 for mixing into the glyph's plane bounds.
    let tx = step(0.0, position.x);  // 0 when x <= -0.5, 1 when x >= 0.5
    let ty = step(0.0, position.y);

    // Map local quad position to glyph plane bounds
    let gx = mix(planeBounds.x, planeBounds.z, tx);
    let gy = mix(planeBounds.w, planeBounds.y, ty); // flip Y: bottom→top

    // Apply per-glyph X rotation (angle in particlePos.w, radians)
    let rotAngle = particlePos.w;
    let cosR = cos(rotAngle);
    let sinR = sin(rotAngle);
    // Rotate glyph offset (gx stays in X, gy rotates between Y and Z)
    let rotY = gy * cosR;
    let rotZ = gy * sinR;

    let world_pos = vec3<f32>(
        particlePos.x + gx,
        particlePos.y + rotY,
        particlePos.z + rotZ,
    );

    // UV: map to atlas image bounds
    let u = mix(imageBounds.x, imageBounds.z, tx);
    let v = mix(imageBounds.w, imageBounds.y, ty); // flip V to match glyph orientation

    var out: VOut;
    out.clip_pos = projection_matrix * view_matrix * world_matrix * vec4<f32>(world_pos, 1.0);
    out.v_uv = vec2<f32>(u, v);
    out.v_color = color;
    return out;
}

// ── Fragment ──

@fragment
fn fragment_main(v: VOut) -> @location(0) vec4<f32> {
    let s = textureSample(atlas_tex, atlas_samp, v.v_uv);
    let d = median(s.r, s.g, s.b);

    // Screen-space derivative for anti-aliased edge.
    // With 4x MSAA + native DPR the edges are hardware-smoothed,
    // so we can use a moderate fwidth range for clean anti-aliasing.
    let fw = fwidth(d);
    let alpha = smoothstep(0.5 - fw, 0.5 + fw, d);

    // Discard low-alpha fragments to prevent dark fringe from atlas
    // bilinear filtering sampling black pixels outside the glyph.
    if (alpha < 0.15) {
        discard;
    }

    return vec4<f32>(v.v_color.rgb, v.v_color.a * alpha);
}
