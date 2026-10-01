// Scatter-as-gather at half resolution, one entry point per layer. Every sample within the
// tile's reach is a bokeh of its own CoC; it lands on this pixel if its CoC reaches it, with its
// energy (colour x coverage) spread over its disc: weight 1 / CoC^2, so a small bright highlight
// becomes a large, dimmer disc. Polygonal apertures reshape both the sampling and the coverage
// test.
// - gatherNear: the near field, premultiplied with its coverage: it spills over whatever is
//   behind it, with an alpha that is the fraction of the aperture it hides.
// - gatherFar: the background, normalised. A sample farther than this pixel's surface may only
//   spread as far as the smaller of the two CoCs, so a blurred background never bleeds over a
//   sharper surface in front of it. Where the near field hides the background, its colour and
//   CoC are filled in push-pull from the layer's coverage-weighted chain (holeFill), and the
//   pixel gathers like a visible one: the fill equals the visible background at a near
//   object's silhouette and fades smoothly inward, and the same rule stops a blurred distance
//   from flooding it, so nothing of the silhouette shows through the near field's blur. The
//   lens only sees the hidden background within the near field's CoC of a visible edge, so the
//   colour is only pulled from that far; beyond it the near field's own colour stands in.

@group(0) @binding(0) var layerTex : texture_2d<f32>;
@group(0) @binding(1) var tileTex  : texture_2d<f32>;
@group(0) @binding(2) var outTex   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var nearTex  : texture_2d<f32>;   // gatherFar: the near layer
@group(0) @binding(4) var<uniform> p : DofParams;

const GOLDEN_ANGLE : f32 = 2.39996323;
// the gather reads the chain level whose texels suit the disc: level 0 up to a 12-pixel radius,
// then one level coarser for every doubling, so the sample count stays the same
const LEVEL_RADIUS : f32 = 12.0;
const LEVELS : u32 = 3u;

// Interleaved gradient noise (Jimenez 2014): a per-pixel rotation of the sample pattern.
fn ign(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

struct Sample {
    color    : vec3f,
    coc      : f32,
    coverage : f32,
    distance : f32,   // from this pixel, in the aperture's metric (half-resolution pixels)
}

struct Pattern {
    n         : u32,
    level     : i32,
    scale     : f32,
    center    : vec2f,   // this pixel in the level's texels
    lim       : vec2i,
    dir       : vec2f,
    step      : vec2f,
    invN      : f32,
}

fn pattern(gid: vec2u, radius: f32) -> Pattern {
    var pt : Pattern;
    pt.n = max(p.sampleCount, 1u);
    let levelU = min(u32(ceil(log2(max(radius / LEVEL_RADIUS, 1.0)))), LEVELS - 1u);
    pt.level = i32(levelU);
    pt.scale = f32(1u << levelU);
    pt.center = (vec2f(gid) + 0.5) / pt.scale - 0.5;
    pt.lim = vec2i(textureDimensions(layerTex, pt.level)) - 1;
    let rotation = 2.0 * PI * ign(vec2f(gid) + f32(p.frame % 64u) * vec2f(5.588238, 5.588238));
    pt.dir = vec2f(cos(rotation), sin(rotation));
    // the Vogel spiral, stepped by a golden-angle rotation (no trigonometry per sample)
    pt.step = vec2f(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE));
    pt.invN = 1.0 / f32(pt.n);
    return pt;
}

fn nextSample(pt: ptr<function, Pattern>, i: u32, radius: f32) -> Sample {
    let dir = (*pt).dir;
    (*pt).dir = vec2f(dir.x * (*pt).step.x - dir.y * (*pt).step.y, dir.x * (*pt).step.y + dir.y * (*pt).step.x);
    var shape = 1.0;
    if (p.bladeCount >= 3u) { shape = apertureRadius(atan2(dir.y, dir.x)); }
    let offset = dir * (sqrt((f32(i) + 0.5) * (*pt).invN) * radius * shape);
    let sc = clamp(vec2i(round((*pt).center + offset / (*pt).scale)), vec2i(0), (*pt).lim);
    let t = textureLoad(layerTex, sc, (*pt).level);
    var s : Sample;
    s.color = t.rgb;
    s.coc = layerCoc(t.a);
    s.coverage = layerCoverage(t.a);
    s.distance = length(vec2f(sc) - (*pt).center) * (*pt).scale / shape;
    return s;
}

@compute @workgroup_size(8, 8)
fn gatherNear(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    if (gid.x >= hs.x || gid.y >= hs.y) { return; }
    let radius = min(textureLoad(tileTex, gid.xy / 8u, 0).r, p.maxCoc * 0.5);
    if (radius < 0.5) {
        textureStore(outTex, gid.xy, vec4f(0.0));
        return;
    }
    // each sample stands for radius^2 * pi / n of the disc; its bokeh spreads over pi * coc^2
    let area = radius * radius / f32(max(p.sampleCount, 1u));
    let own = textureLoad(layerTex, gid.xy, 0);
    let ownCoc = layerCoc(own.a);
    let ownW = cover(0.0, ownCoc) * layerCoverage(own.a) * area / max(ownCoc * ownCoc, 0.25);
    var sum = own.rgb * ownW;
    var weight = ownW;
    var pt = pattern(gid.xy, radius);
    for (var i = 0u; i < pt.n; i++) {
        let s = nextSample(&pt, i, radius);
        let w = cover(s.distance, s.coc) * s.coverage * area / max(s.coc * s.coc, 0.25);
        sum += s.color * w;
        weight += w;
    }
    let alpha = saturate(weight);
    let color = select(vec3f(0.0), sum / max(weight, 1e-6), weight > 1e-6);
    textureStore(outTex, gid.xy, vec4f(color * alpha, alpha));
}

// The background hidden behind this pixel, pulled from the finest level of the chain that
// covers it: each level is sampled bilinearly, weighted by coverage, and takes over what the finer
// levels did not cover. The CoC comes from the whole chain; the colour only from levels whose
// texels are within `reach` (half-resolution pixels), and what they leave to `stand_in`.
fn holeFill(gid: vec2u, reach: f32, standIn: vec3f) -> vec4f {
    var color = vec3f(0.0);
    var coc = 0.0;
    var colorLeft = 1.0;
    var cocLeft = 1.0;
    let levels = i32(textureNumLevels(layerTex));
    let colorLevels = i32(floor(log2(max(reach, 1.0)))) + 2;
    for (var level = 0; level < levels; level++) {
        let scale = f32(1u << u32(level));
        let pos = (vec2f(gid) + 0.5) / scale - 0.5;
        let base = floor(pos);
        let f = pos - base;
        let lim = vec2i(textureDimensions(layerTex, level)) - 1;
        var sum = vec4f(0.0);
        var covered = 0.0;
        for (var i = 0u; i < 4u; i++) {
            let o = vec2f(f32(i & 1u), f32(i >> 1u));
            let t = textureLoad(layerTex, clamp(vec2i(base + o), vec2i(0), lim), level);
            let bilinear = select(1.0 - f.x, f.x, o.x > 0.5) * select(1.0 - f.y, f.y, o.y > 0.5);
            let a = layerCoverage(t.a) * bilinear;
            sum += vec4f(t.rgb, layerCoc(t.a)) * a;
            covered += a;
        }
        // trust a level once a quarter of it is covered
        let w = saturate(covered * 4.0);
        let mean = sum / max(covered, 1e-6);
        coc += cocLeft * w * mean.a;
        cocLeft *= 1.0 - w;
        if (level < colorLevels) {
            color += colorLeft * w * mean.rgb;
            colorLeft *= 1.0 - w;
        }
        if (cocLeft < 1e-3) { break; }
    }
    return vec4f(color + colorLeft * standIn, coc / max(1.0 - cocLeft, 1e-6));
}

@compute @workgroup_size(8, 8)
fn gatherFar(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    if (gid.x >= hs.x || gid.y >= hs.y) { return; }
    let raw = textureLoad(layerTex, gid.xy, 0);
    let ownCoverage = layerCoverage(raw.a);
    // Where the near field hides part of the background, the part it hides continues the
    // background around it: colour and CoC pulled from the nearest levels that see it. The
    // pixel then gathers like any visible one, so its surface limits what spreads over it and
    // it joins the visible background without a seam.
    var own = vec4f(raw.rgb, layerCoc(raw.a));
    if (ownCoverage < 0.999) {
        let near = textureLoad(nearTex, gid.xy, 0);
        own = mix(holeFill(gid.xy, layerCoc(near.a), near.rgb), own, ownCoverage);
    }
    let cc = own.a;
    let absC = abs(cc);
    let tile = textureLoad(tileTex, gid.xy / 8u, 0).rg;
    let radius = min(max(tile.g, absC), p.maxCoc * 0.5);
    if (radius < 0.5) {
        textureStore(outTex, gid.xy, vec4f(own.rgb, cc));
        return;
    }
    let area = radius * radius / f32(max(p.sampleCount, 1u));
    let ownW = cover(0.0, absC) * area / max(absC * absC, 0.25);
    var sum = own.rgb * ownW;
    var weight = ownW;
    var pt = pattern(gid.xy, radius);
    for (var i = 0u; i < pt.n; i++) {
        let s = nextSample(&pt, i, radius);
        let absS = abs(s.coc);
        // farther than this pixel's surface by more than the layer tolerance: spread limited to it
        let tol = layerTolerance(min(absC, absS));
        let behind = saturate((s.coc - cc - tol) / tol);
        let spread = mix(absS, min(absS, absC), behind);
        let w = cover(s.distance, spread) * s.coverage * area / max(absS * absS, 0.25);
        sum += s.color * w;
        weight += w;
    }
    var color = own.rgb;
    if (weight > 1e-6) { color = sum / weight; }
    textureStore(outTex, gid.xy, vec4f(color, cc));
}
