// Screen-space ambient occlusion, the depth-aware blur (after ssao_common.wgsl): the occlusion
// averaged over `SSAO_BLUR_RADIUS` pixels each side along one axis, each weighed by its distance
// on screen and by how near its point's distance from the camera is to the centre's, so the
// noise of the rotated kernel goes and the occlusion of one surface does not bleed onto another.
// `blur_x` blurs across into `aoOut`; ssao_composite.wgsl blurs down with `ssaoBlur(.., y)`.

const SSAO_BLUR_RADIUS : i32 = 4;

@group(0) @binding(3) var aoIn : texture_2d<f32>;

// The blurred (share left open, distance) at `px` along `axis`.
fn ssaoBlur(px: vec2i, axis: vec2i) -> vec2f {
    let centre = textureLoad(aoIn, px, 0).xy;
    if (centre.y <= 0.0) { return centre; }
    let size = vec2i(params.size);
    // a point counts while its distance is within a few per cent of the centre's
    let tolerance = 0.03 * centre.y;
    var sum = 0.0;
    var weight = 0.0;
    for (var i = -SSAO_BLUR_RADIUS; i <= SSAO_BLUR_RADIUS; i++) {
        let q = clamp(px + axis * i, vec2i(0), size - 1);
        let v = textureLoad(aoIn, q, 0).xy;
        let x = f32(i) / f32(SSAO_BLUR_RADIUS);
        let w = exp(-2.0 * x * x) * exp(-abs(v.y - centre.y) / tolerance) * select(0.0, 1.0, v.y > 0.0);
        sum += v.x * w;
        weight += w;
    }
    return vec2f(sum / max(weight, 1e-6), centre.y);
}
