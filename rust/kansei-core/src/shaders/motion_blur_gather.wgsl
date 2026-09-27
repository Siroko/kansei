// The blur: samples in mirrored pairs along the tile's dominant blur vector, every other pair
// along the pixel's own where it moves (Guertin et al. 2014: a background panning one way
// behind a car driving the other keeps its own blur), jittered per pixel and frame
// (interleaved gradient noise) so the gaps between samples read as fine noise, not bands. Each sample counts where it plausibly covers this pixel during the exposure:
//   - a sample behind this pixel, as far as this pixel's own streak reaches (the background
//     seen through a moving object's smeared edge);
//   - a sample in front, as far as its own streak reaches (a moving object smearing over what
//     is behind it; a static background never smears over a sharp object in front of it);
// with Jimenez' mirroring of the pair's weights, which fills in the background behind a moving
// foreground from the side where it is visible. What the samples leave uncovered is this
// pixel's own colour, so a static pixel next to a moving one stays sharp (no halos).

@group(0) @binding(0) var colorTex     : texture_2d<f32>;
@group(0) @binding(1) var motionTex    : texture_2d<f32>;
@group(0) @binding(2) var neighbourTex : texture_2d<f32>;
@group(0) @binding(3) var outputTex    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> p   : MotionBlurParams;

// how sharply samples are sorted into in front of and behind this pixel, per relative depth
// difference (full weight at 2.5 %)
const DEPTH_SCALE : f32 = 20.0;

fn ign(pixel: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
}

// x: the sample is behind the centre, y: in front of it
fn depthCompare(centerDepth: f32, sampleDepth: f32) -> vec2f {
    let d = DEPTH_SCALE * (sampleDepth - centerDepth) / max(min(centerDepth, sampleDepth), 1e-4);
    return saturate(vec2f(0.5 + d, 0.5 - d));
}

// whether a streak of each radius (x: centre's, y: sample's) reaches `offset` pixels
fn spreadCompare(offset: f32, radius: vec2f) -> vec2f {
    return saturate(radius - offset + 1.0);
}

fn sampleWeight(centerDepth: f32, centerRadius: f32, sampleDepth: f32, sampleRadius: f32, offset: f32) -> f32 {
    return dot(depthCompare(centerDepth, sampleDepth), spreadCompare(offset, vec2f(centerRadius, sampleRadius)));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = vec2i(p.size);
    let ip = vec2i(gid.xy);
    if (any(ip >= size)) { return; }
    let center = textureLoad(colorTex, ip, 0);
    if (p.enabled == 0u) {
        textureStore(outputTex, ip, center);
        return;
    }

    let seed = vec2f(ip) + 5.588238 * f32(p.frame % 64u);
    let noise = vec2f(ign(seed), ign(seed.yx + 31.0));
    // a jittered tile lookup hides the tile grid where the dominant direction changes
    let jitter = (noise - 0.5) * (0.5 * f32(TILE));
    let tile = clamp(vec2i((vec2f(ip) + 0.5 + jitter) / f32(TILE)), vec2i(0), vec2i(tileCount()) - 1);
    let tileMotion = textureLoad(neighbourTex, tile, 0);
    let dominant = tileMotion.xy;
    let dominantLen = length(dominant);
    if (dominantLen < 0.5) {
        textureStore(outputTex, ip, center);
        return;
    }

    let steps = max(p.steps, 1u);
    // rounded to whole pixels, offsets up to the radius plus half a pixel land evenly on 0..r
    let reach = dominant * ((dominantLen + 0.5) / dominantLen);

    // everything around moves alike (a camera pan over distant scenery): a plain average
    if (tileMotion.z > 0.9 * dominantLen) {
        var sum = vec3f(0.0);
        for (var i = 0u; i < steps; i++) {
            let d = vec2i(round(reach * ((f32(i) + noise.x) / f32(steps))));
            sum += textureLoad(colorTex, clamp(ip + d, vec2i(0), size - 1), 0).rgb;
            sum += textureLoad(colorTex, clamp(ip - d, vec2i(0), size - 1), 0).rgb;
        }
        textureStore(outputTex, ip, vec4f(sum / f32(2u * steps), center.a));
        return;
    }

    let m = textureLoad(motionTex, ip, 0);
    let centerRadius = length(m.xy);
    let centerDepth = m.z;
    let own = select(reach, m.xy * ((centerRadius + 0.5) / centerRadius), centerRadius >= 1.0);
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + noise.x) / f32(steps);
        let d = vec2i(round(select(reach, own, (i & 1u) == 1u) * t));
        // the distance to the pixels actually read, so a static pixel only ever reads itself
        let offset = length(vec2f(d));
        let q1 = clamp(ip + d, vec2i(0), size - 1);
        let q2 = clamp(ip - d, vec2i(0), size - 1);
        let m1 = textureLoad(motionTex, q1, 0);
        let m2 = textureLoad(motionTex, q2, 0);
        let r1 = length(m1.xy);
        let r2 = length(m2.xy);
        var w1 = sampleWeight(centerDepth, centerRadius, m1.z, r1, offset);
        var w2 = sampleWeight(centerDepth, centerRadius, m2.z, r2, offset);
        // the pair's nearer, faster sample decides for both
        let mirror = vec2<bool>(m1.z > m2.z, r2 > r1);
        w1 = select(w1, w2, all(mirror));
        w2 = select(w1, w2, any(mirror));
        sum += w1 * textureLoad(colorTex, q1, 0).rgb + w2 * textureLoad(colorTex, q2, 0).rgb;
        weight += w1 + w2;
    }
    let n = f32(2u * steps);
    var rgb = sum / n + saturate(1.0 - weight / n) * center.rgb;
    if (any(rgb != rgb)) { rgb = center.rgb; }
    textureStore(outputTex, ip, vec4f(rgb, center.a));
}
