// SMAA 1x, the algorithm SmaaEffect's three passes run: a port of Jorge Jimenez et al.'s
// SMAA.hlsl (https://github.com/iryoku/smaa, MIT; the notice is in smaa.rs's module docs):
// SMAALumaEdgeDetectionPS, SMAAArea, SMAADetectHorizontalCornerPattern,
// SMAADetectVerticalCornerPattern, the orthogonal part of SMAABlendingWeightCalculationPS and
// SMAANeighborhoodBlendingPS.
//
// It reads pixels through three functions smaa_passes.wgsl defines (from textures):
//   smaaLumaAt(p: vec2i) -> f32     the luma of the input at p (smaaLumaOf), clamped to the frame
//   smaaEdgesAt(p: vec2i) -> vec2f  x the left edge, y the top edge, each 0 or 1; 0 outside the frame
//   smaaColorAt(p: vec2i) -> vec3f  the input as stored, p clamped to the frame
//
// Where iryoku reads two edge flags with one bilinear fetch (the searches' pseudo-gathers, the
// crossing edges at -0.25 px), this reads the two flags and mixes them; the searches walk one
// pixel at a time, so no search texture is needed.

struct SmaaParams {
    size           : vec2u,
    threshold      : f32,
    localContrast  : f32,
    cornerRounding : f32,
    // how far (px) the searches reach along an edge each way
    reach          : u32,
    // SMAA_FLAG_*
    flags          : u32,
    // the film grain's frame (smaa.rs)
    frame          : u32,
    grain          : f32,
    grainSize      : f32,
    pad0           : u32,
    pad1           : u32,
}

@group(0) @binding(0) var<uniform> smaa : SmaaParams;
@group(0) @binding(1) var smaaInput     : texture_2d<f32>;
// SMAA's area texture, its 1x subtexture (160 x 80, RG8)
@group(0) @binding(2) var smaaAreaTex   : texture_2d<f32>;
@group(0) @binding(3) var smaaLinear    : sampler;

// the input is sRGB-encoded (the tonemapper's encode_srgb); otherwise display-linear
const SMAA_FLAG_ENCODED : u32 = 1u;
// a triangular dither after the blend (the tonemapper's, left to SMAA)
const SMAA_FLAG_DITHER  : u32 = 2u;

fn smaaIsEncoded() -> bool {
    return (smaa.flags & SMAA_FLAG_ENCODED) != 0u;
}

fn smaaInFrame(p: vec2i) -> bool {
    return all(p >= vec2i(0)) && all(p < vec2i(smaa.size));
}

fn smaaClampToFrame(p: vec2i) -> vec2i {
    return clamp(p, vec2i(0), vec2i(smaa.size) - vec2i(1));
}

fn smaaSrgbEncode(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

fn smaaSrgbDecode(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(pow((x + 0.055) / 1.055, vec3f(2.4)), x / 12.92, x <= vec3f(0.04045));
}

// The luma edges are found by: Rec. 709 weights on the sRGB-encoded signal (iryoku's luma edge
// detection wants gamma-corrected colour), encoding first when the input is linear.
fn smaaLumaOf(stored: vec3f) -> f32 {
    var c = stored;
    if (!smaaIsEncoded()) {
        c = smaaSrgbEncode(c);
    }
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ── Edge detection (SMAALumaEdgeDetectionPS) ──

// The left (x) and top (y) edges of p: a luma step of at least the threshold, kept where it is
// at least 1 / localContrast of the largest step round it (local contrast adaptation).
fn smaaDetectEdges(p: vec2i) -> vec2f {
    let l = smaaLumaAt(p);
    let lLeft = smaaLumaAt(p + vec2i(-1, 0));
    let lTop = smaaLumaAt(p + vec2i(0, -1));
    let delta = abs(vec2f(l) - vec2f(lLeft, lTop));
    var edges = step(vec2f(smaa.threshold), delta);
    if (dot(edges, vec2f(1.0)) == 0.0) {
        return vec2f(0.0);
    }
    let lRight = smaaLumaAt(p + vec2i(1, 0));
    let lBottom = smaaLumaAt(p + vec2i(0, 1));
    var maxDelta = max(delta, abs(vec2f(l) - vec2f(lRight, lBottom)));
    let lLeftLeft = smaaLumaAt(p + vec2i(-2, 0));
    let lTopTop = smaaLumaAt(p + vec2i(0, -2));
    maxDelta = max(maxDelta, abs(vec2f(lLeft, lTop) - vec2f(lLeftLeft, lTopTop)));
    let finalDelta = max(maxDelta.x, maxDelta.y);
    edges *= step(vec2f(finalDelta), smaa.localContrast * delta);
    return edges;
}

// ── Blending weights (SMAABlendingWeightCalculationPS, orthogonal) ──

fn smaaLeftEdge(p: vec2i) -> f32 {
    return smaaEdgesAt(p).x;
}

fn smaaTopEdge(p: vec2i) -> f32 {
    return smaaEdgesAt(p).y;
}

// SMAAArea: the areas on each side of an edge run `sqrtD` (the square roots of its distances to
// each end) whose crossing edges are e1 and e2 (0, 0.25, 0.75 or 1), from the area texture
// (stored quadratically, so its bilinear filter interpolates the square roots).
fn smaaArea(sqrtD: vec2f, e1: f32, e2: f32) -> vec2f {
    let texel = 16.0 * round(4.0 * vec2f(e1, e2)) + sqrtD;
    let uv = (texel + 0.5) / vec2f(160.0, 80.0);
    return textureSampleLevel(smaaAreaTex, smaaLinear, uv, 0.0).rg;
}

// SMAADetect{Horizontal,Vertical}CornerPattern's factors for a run `d` from its ends: the
// crossing edges one step past each end's pair, on the near side (x) and the far side (y).
fn smaaCornerFactors(d: vec2f, nearStart: f32, nearEnd: f32, farStart: f32, farEnd: f32) -> vec2f {
    let leftRight = step(d.xy, d.yx);
    // less blending for pixels in the middle of a run
    let rounding = (1.0 - smaa.cornerRounding) * leftRight / (leftRight.x + leftRight.y);
    let factor = vec2f(1.0) - rounding.x * vec2f(nearStart, farStart) - rounding.y * vec2f(nearEnd, farEnd);
    return saturate(factor);
}

// A crossing edge at column c of a horizontal run on row y: a left edge on row y or y - 1.
fn smaaCrossesRow(c: i32, y: i32) -> bool {
    return (smaaLeftEdge(vec2i(c, y)) + smaaLeftEdge(vec2i(c, y - 1))) > 0.0;
}

// A crossing edge at row c of a vertical run on column x: a top edge on column x or x - 1.
fn smaaCrossesColumn(x: i32, c: i32) -> bool {
    return (smaaTopEdge(vec2i(x, c)) + smaaTopEdge(vec2i(x - 1, c))) > 0.0;
}

// The blending weights of p's edges, searching at most `reach` pixels along each run: rg for its
// top edge's run (horizontal), ba for its left edge's (vertical). Zero for a pixel with neither.
fn smaaWeights(p: vec2i, reach: i32) -> vec4f {
    var weights = vec4f(0.0);
    let e = smaaEdgesAt(p);
    if (e.y > 0.0) {
        // the run of top edges along row y, and its ends' crossing edges
        let x = p.x;
        let y = p.y;
        var l = x;
        loop {
            if (l <= x - reach || smaaCrossesRow(l, y) || smaaTopEdge(vec2i(l - 1, y)) == 0.0) {
                break;
            }
            l -= 1;
        }
        var r = x;
        loop {
            if (r >= x + reach || smaaCrossesRow(r + 1, y) || smaaTopEdge(vec2i(r + 1, y)) == 0.0) {
                break;
            }
            r += 1;
        }
        let d = vec2f(f32(x - l), f32(r - x));
        let e1 = 0.75 * smaaLeftEdge(vec2i(l, y)) + 0.25 * smaaLeftEdge(vec2i(l, y - 1));
        let e2 = 0.75 * smaaLeftEdge(vec2i(r + 1, y)) + 0.25 * smaaLeftEdge(vec2i(r + 1, y - 1));
        var w = smaaArea(sqrt(d), e1, e2);
        w *= smaaCornerFactors(
            d,
            smaaLeftEdge(vec2i(l, y + 1)),
            smaaLeftEdge(vec2i(r + 1, y + 1)),
            smaaLeftEdge(vec2i(l, y - 2)),
            smaaLeftEdge(vec2i(r + 1, y - 2))
        );
        weights = vec4f(w, weights.zw);
    }
    if (e.x > 0.0) {
        // the run of left edges down column x, and its ends' crossing edges
        let x = p.x;
        let y = p.y;
        var t = y;
        loop {
            if (t <= y - reach || smaaCrossesColumn(x, t) || smaaLeftEdge(vec2i(x, t - 1)) == 0.0) {
                break;
            }
            t -= 1;
        }
        var b = y;
        loop {
            if (b >= y + reach || smaaCrossesColumn(x, b + 1) || smaaLeftEdge(vec2i(x, b + 1)) == 0.0) {
                break;
            }
            b += 1;
        }
        let d = vec2f(f32(y - t), f32(b - y));
        let e1 = 0.75 * smaaTopEdge(vec2i(x, t)) + 0.25 * smaaTopEdge(vec2i(x - 1, t));
        let e2 = 0.75 * smaaTopEdge(vec2i(x, b + 1)) + 0.25 * smaaTopEdge(vec2i(x - 1, b + 1));
        var w = smaaArea(sqrt(d), e1, e2);
        w *= smaaCornerFactors(
            d,
            smaaTopEdge(vec2i(x + 1, t)),
            smaaTopEdge(vec2i(x + 1, b + 1)),
            smaaTopEdge(vec2i(x - 2, t)),
            smaaTopEdge(vec2i(x - 2, b + 1))
        );
        weights = vec4f(weights.xy, w);
    }
    return weights;
}

// ── Neighbourhood blending (SMAANeighborhoodBlendingPS) ──

// The input at p in linear light (decoded when it is encoded), as iryoku's sRGB reads give it.
fn smaaLinearAt(p: vec2i) -> vec3f {
    let c = smaaColorAt(p);
    if (smaaIsEncoded()) {
        return smaaSrgbDecode(c);
    }
    return c;
}

// p blended with its neighbours by the weights at p (`here`), p + (1, 0) (`right`) and
// p + (0, 1) (`below`): toward the side of the stronger direction, mixed in linear light and
// encoded again. A pixel with no weight comes through as stored.
fn smaaBlend(p: vec2i, here: vec4f, right: vec4f, below: vec4f) -> vec3f {
    // right, bottom, left, top
    let a = vec4f(right.a, below.g, here.b, here.r);
    if (dot(a, vec4f(1.0)) < 1e-5) {
        return smaaColorAt(p);
    }
    let horizontal = max(a.x, a.z) > max(a.y, a.w);
    var amount = a.yw;
    var toward = vec2i(0, 1);
    if (horizontal) {
        amount = a.xz;
        toward = vec2i(1, 0);
    }
    let weight = amount / (amount.x + amount.y);
    let c = smaaLinearAt(p);
    let c1 = mix(c, smaaLinearAt(p + toward), amount.x);
    let c2 = mix(c, smaaLinearAt(p - toward), amount.y);
    let blended = weight.x * c1 + weight.y * c2;
    if (smaaIsEncoded()) {
        return smaaSrgbEncode(blended);
    }
    return blended;
}
