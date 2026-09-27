// Scatter-as-gather at half resolution. Every sample within the tile's reach is a bokeh of its
// own CoC; it lands on this pixel if its CoC reaches it, with its energy spread over its disc
// (weight 1 / CoC^2, so a small bright highlight becomes a large, dimmer disc). Samples in front
// of the focus plane, or clearly in front of this pixel, form a foreground layer with coverage
// (they spill over anything behind them, this pixel's own sample included when it is in the near
// field); the others form the background, where a sample may only spread as far as the smaller
// of its own and this pixel's CoC, so a blurred background never bleeds over a sharper object in
// front of it. Polygonal apertures reshape both the sampling and the coverage test.

@group(0) @binding(0) var halfTex : texture_2d<f32>;
@group(0) @binding(1) var tileTex : texture_2d<f32>;
@group(0) @binding(2) var bgOut   : texture_storage_2d<rgba16float, write>;   // rgb, a = own CoC (half px)
@group(0) @binding(3) var fgOut   : texture_storage_2d<rgba16float, write>;   // premultiplied rgb, a = coverage
@group(0) @binding(4) var<uniform> p : DofParams;

// the gather reads the chain level whose texels suit the disc: level 0 up to a 12-pixel radius,
// then one level coarser for every doubling, so the sample count stays the same
const LEVEL_RADIUS : f32 = 12.0;
const LEVELS : u32 = 3u;

const GOLDEN_ANGLE : f32 = 2.39996323;

// Distance from the centre to the aperture's edge along angle theta, for a unit aperture.
fn apertureRadius(theta: f32) -> f32 {
    if (p.bladeCount < 3u) { return 1.0; }
    let seg = 2.0 * PI / f32(p.bladeCount);
    let a = theta - p.bladeRotation;
    let local = a - seg * floor(a / seg) - seg * 0.5;
    return cos(seg * 0.5) / cos(local);
}

// Interleaved gradient noise (Jimenez 2014): a per-pixel rotation of the sample pattern.
fn ign(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

fn cover(distance: f32, coc: f32) -> f32 {
    return saturate(coc - distance + 0.5);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    if (gid.x >= hs.x || gid.y >= hs.y) { return; }
    let center = textureLoad(halfTex, gid.xy, 0);
    let cc = center.a;
    let tile = textureLoad(tileTex, gid.xy / 8u, 0).rg;
    let radius = min(max(max(tile.r, tile.g), abs(cc)), p.maxCoc * 0.5);
    if (radius < 0.5) {
        textureStore(bgOut, gid.xy, center);
        textureStore(fgOut, gid.xy, vec4f(0.0));
        return;
    }

    let n = max(p.sampleCount, 1u);
    let levelU = min(u32(ceil(log2(max(radius / LEVEL_RADIUS, 1.0)))), LEVELS - 1u);
    let level = i32(levelU);
    let scale = f32(1u << levelU);
    let levelCenter = (vec2f(gid.xy) + 0.5) / scale - 0.5;
    // each sample stands for radius^2 * pi / n of the disc; its bokeh spreads over pi * coc^2
    let area = radius * radius / f32(n);
    let absC = abs(cc);
    let centerW = area / max(absC * absC, 0.25) * cover(0.0, absC);
    let centerFront = saturate(-cc - 0.5);
    var bgSum = center.rgb * centerW * (1.0 - centerFront);
    var bgWeight = centerW * (1.0 - centerFront);
    var fgSum = center.rgb * centerW * centerFront;
    var fgWeight = centerW * centerFront;
    // behind a near-field pixel the background is hidden: estimate it from the nearest
    // background samples when none of them spreads this far
    var behindSum = vec3f(0.0);
    var behindWeight = 0.0;
    let nearField = cc < -0.5;
    let rotation = 2.0 * PI * ign(vec2f(gid.xy) + f32(p.frame % 64u) * vec2f(5.588238, 5.588238));
    let lim = vec2i(textureDimensions(halfTex, level)) - 1;
    let invN = 1.0 / f32(n);
    let polygonal = p.bladeCount >= 3u;
    // the Vogel spiral, stepped by a golden-angle rotation (no trigonometry per sample)
    let step = vec2f(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE));
    var dir = vec2f(cos(rotation), sin(rotation));
    for (var i = 0u; i < n; i++) {
        var shape = 1.0;
        if (polygonal) { shape = apertureRadius(atan2(dir.y, dir.x)); }
        let r = sqrt((f32(i) + 0.5) * invN) * radius * shape;
        let offset = dir * r;
        dir = vec2f(dir.x * step.x - dir.y * step.y, dir.x * step.y + dir.y * step.x);
        let sc = clamp(vec2i(round(levelCenter + offset / scale)), vec2i(0), lim);
        let s = textureLoad(halfTex, sc, level);
        let cs = s.a;
        let absS = abs(cs);
        let distance = length(vec2f(sc) - levelCenter) * scale / shape;
        let energy = area / max(absS * absS, 0.25);
        // in the near field, or in front of this pixel by more than the layer tolerance: foreground
        let tol = layerTolerance(min(absC, absS));
        let front = max(saturate((cc - cs - tol) / tol), saturate(-cs - 0.5));
        let wf = cover(distance, absS) * energy * front;
        let spread = select(min(absS, absC), absS, nearField);
        let wb = cover(distance, spread) * energy * (1.0 - front);
        let wh = (1.0 - front) / (1.0 + distance * distance);
        behindSum += s.rgb * wh;
        behindWeight += wh;
        fgSum += s.rgb * wf;
        fgWeight += wf;
        bgSum += s.rgb * wb;
        bgWeight += wb;
    }
    var bg = select(center.rgb, behindSum / max(behindWeight, 1e-6), behindWeight > 1e-6 && nearField);
    if (bgWeight > 1e-6) { bg = bgSum / bgWeight; }
    let alpha = saturate(fgWeight);
    let fg = select(vec3f(0.0), fgSum / max(fgWeight, 1e-6), fgWeight > 1e-6);
    textureStore(bgOut, gid.xy, vec4f(bg, cc));
    textureStore(fgOut, gid.xy, vec4f(fg * alpha, alpha));
}
