// Temporal anti-aliasing resolve (Karis 2014, Salvi 2016, Pedersen 2016), on scene-linear HDR,
// and temporal upsampling when the scene is rendered below the output size (Unreal's TAAU).
//
// Per output pixel:
//   1. the 3x3 neighbourhood of this frame's jittered samples around it (rendered pixels), in
//      YCoCg of Karis' reversible tonemap (so single bright samples can't flicker): a Gaussian
//      reconstruction at the pixel's centre from the samples' positions measured in output
//      pixels, and the mean and deviation of the colours;
//   2. motion from the closest surface in the neighbourhood (thin edges keep their motion): the
//      velocity target where a material wrote one, else the camera's reprojection of its depth;
//   3. the history (output size) there, Catmull-Rom filtered, clipped toward the neighbourhood's
//      variance box (the colours this pixel can plausibly have now), blended with less weight in
//      motion, and dropped where the history saw another surface: its stored view depth
//      disagrees with where this surface was (disocclusion; swaying foliage uncovers and covers
//      itself all the time);
//   4. back to linear light, written to the chain and to next frame's history (with this
//      pixel's view depth in alpha).
//
// Upsampling, rendered pixels are further apart than output pixels, so most output pixels have
// no sample close by in a given frame: the current frame's share of the blend scales with the
// reconstruction's total weight (below 1 only when upsampling), and the history, which the
// jitter fills with samples over the frames, carries the rest. The clip box widens by the
// upsampling factor, and the disocclusion test accepts the surface of any of the four rendered
// pixels around the output pixel (the jitter changes which one lies under it). Features much
// thinner than a rendered pixel are missed by most frames and clipped away, as sub-pixel ones
// are at 1:1.

struct TaaParams {
    invViewProj   : mat4x4f,   // inverse of this frame's jittered view-projection (the depth's)
    viewProj      : mat4x4f,   // this frame, unjittered
    prevViewProj  : mat4x4f,   // last frame, unjittered
    jitterPx      : vec2f,     // this frame's jitter in rendered pixels, +x right, +y down
    size          : vec2f,     // output (and history) size
    feedbackMin   : f32,       // history weight in fast motion
    feedbackMax   : f32,       // history weight when still
    exposure      : f32,       // scene multiplier of the working space (the tonemapper's)
    varianceGamma : f32,       // clip box half-size in standard deviations
    hasHistory    : u32,
    hasVelocity   : u32,
    inputSize     : vec2f,     // rendered size: the colour, depth and velocity textures
}

@group(0) @binding(0) var currentTex    : texture_2d<f32>;
@group(0) @binding(1) var depthTex      : texture_depth_2d;
@group(0) @binding(2) var velocityTex   : texture_2d<f32>;
@group(0) @binding(3) var historyTex    : texture_2d<f32>;
@group(0) @binding(4) var linearSampler : sampler;
@group(0) @binding(5) var outputTex     : texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var historyOut    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> p    : TaaParams;

// velocities at or beyond this are the GBuffer's "none written" clear value
const NO_VELOCITY : f32 = 1000.0;

fn luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn rgbToYCoCg(c: vec3f) -> vec3f {
    return vec3f(dot(c, vec3f(0.25, 0.5, 0.25)), dot(c, vec3f(0.5, 0.0, -0.5)), dot(c, vec3f(-0.25, 0.5, -0.25)));
}

fn yCoCgToRgb(c: vec3f) -> vec3f {
    return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// exposed light -> x / (1 + luma(x)) -> YCoCg, and back
fn toWorking(c: vec3f) -> vec3f {
    let e = max(c, vec3f(0.0)) * p.exposure;
    return rgbToYCoCg(e / (1.0 + luma(e)));
}

fn fromWorking(c: vec3f) -> vec3f {
    let t = max(yCoCgToRgb(c), vec3f(0.0));
    return t / (max(1.0 - luma(t), 1e-4) * p.exposure);
}

// Clip toward the box centre (Playdead): keeps the history's hue where clamping would not.
fn clipToBox(h: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec3f {
    let center = 0.5 * (boxMax + boxMin);
    let extents = 0.5 * (boxMax - boxMin) + 1e-6;
    let v = h - center;
    let a = abs(v / extents);
    let m = max(a.x, max(a.y, a.z));
    return select(h, center + v / m, m > 1.0);
}

// Catmull-Rom history lookup from 5 bilinear taps (the corner taps' weights are negligible).
fn sampleHistory(uv: vec2f) -> vec3f {
    let pos = uv * p.size;
    let t1 = floor(pos - 0.5) + 0.5;
    let f = pos - t1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (t1 - 1.0) / p.size;
    let t3 = (t1 + 2.0) / p.size;
    let t12 = (t1 + w2 / w12) / p.size;
    var c = textureSampleLevel(historyTex, linearSampler, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
    c += textureSampleLevel(historyTex, linearSampler, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
    c += textureSampleLevel(historyTex, linearSampler, t12, 0.0).rgb * (w12.x * w12.y);
    c += textureSampleLevel(historyTex, linearSampler, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
    c += textureSampleLevel(historyTex, linearSampler, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    let w = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / w, vec3f(0.0));
}

// Last frame's view depth (camera motion only) of the surface at screen `uv` and `depth`.
fn previousViewDepth(uv: vec2f, depth: f32) -> f32 {
    let w = p.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return (p.prevViewProj * vec4f(w.xyz / w.w, 1.0)).w;
}

fn depthMismatch(historyDepth: f32, prevViewDepth: f32) -> f32 {
    return abs(historyDepth - prevViewDepth) / max(prevViewDepth, 1e-3);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = vec2i(p.size);
    let ip = vec2i(gid.xy);
    if (any(ip >= size)) { return; }
    let uv = (vec2f(ip) + 0.5) / p.size;
    // this pixel's centre in rendered pixels, the rendered pixel it falls in, and output pixels
    // per rendered pixel
    let inSize = vec2i(p.inputSize);
    let inPos = uv * p.inputSize;
    let base = min(vec2i(floor(inPos)), inSize - 1);
    let scale = p.size / p.inputSize;

    var filtered = vec3f(0.0);
    var filterWeight = 0.0;
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    var lo = vec3f(1e9);
    var hi = vec3f(-1e9);
    var closestDepth = 1.0;
    var closest = base;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = base + vec2i(x, y);
            let q = clamp(o, vec2i(0), inSize - 1);
            let c = toWorking(textureLoad(currentTex, q, 0).rgb);
            // pixel o's sample sits at its centre minus the jitter in the unjittered image; its
            // distance from this pixel's centre in output pixels. exp(-2.29 d^2) fits the
            // Blackman-Harris reconstruction filter
            let d = (vec2f(o) + 0.5 - p.jitterPx - inPos) * scale;
            let w = exp(-2.29 * dot(d, d));
            filtered += c * w;
            filterWeight += w;
            m1 += c;
            m2 += c * c;
            lo = min(lo, c);
            hi = max(hi, c);
            let z = textureLoad(depthTex, q, 0);
            if (z < closestDepth) {
                closestDepth = z;
                closest = q;
            }
        }
    }
    let current = filtered / filterWeight;
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
    // nine samples spread over more output pixels when upsampling, so a feature thinner than a
    // rendered pixel is in few of them: widen the box by as much (it stays within lo..hi)
    let gamma = p.varianceGamma * max(scale.x, scale.y);
    let boxMin = max(lo, mean - gamma * sigma);
    let boxMax = min(hi, mean + gamma * sigma);

    // motion of the closest surface around this pixel
    var velocity = vec2f(0.0);
    var written = false;
    if (p.hasVelocity != 0u) {
        let v = textureLoad(velocityTex, closest, 0).xy;
        if (abs(v.x) < NO_VELOCITY) {
            velocity = v;
            written = true;
        }
    }
    if (!written) {
        let quv = (vec2f(closest) + 0.5) / p.inputSize;
        let w = p.invViewProj * vec4f(quv.x * 2.0 - 1.0, 1.0 - quv.y * 2.0, closestDepth, 1.0);
        let world = vec4f(w.xyz / w.w, 1.0);
        let curr = p.viewProj * world;
        let prev = p.prevViewProj * world;
        velocity = (curr.xy / curr.w - prev.xy / prev.w) * vec2f(0.5, -0.5);
    }
    let prevUV = uv - velocity;

    // this pixel's surface: its view depth now, and (camera motion only) last frame
    let depth = textureLoad(depthTex, base, 0);
    let wp = p.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let surface = vec4f(wp.xyz / wp.w, 1.0);
    let viewDepth = (p.viewProj * surface).w;
    let prevViewDepth = (p.prevViewProj * surface).w;

    var result = current;
    if (p.hasHistory != 0u && all(prevUV >= vec2f(0.0)) && all(prevUV <= vec2f(1.0))) {
        let history = clipToBox(toWorking(sampleHistory(prevUV)), boxMin, boxMax);
        // resampling a moving history blurs it, so trust it less the faster things move
        let motionPx = length(velocity * p.size);
        var feedback = mix(p.feedbackMax, p.feedbackMin, saturate(motionPx / 8.0));
        // disocclusion: the nearest history texel saw a surface at another depth
        let historyDepth = textureLoad(historyTex, clamp(vec2i(prevUV * p.size), vec2i(0), size - 1), 0).a;
        var mismatch = depthMismatch(historyDepth, prevViewDepth);
        if (any(inSize != size)) {
            // upsampling, the jitter moves which surface the (coarser) rendered pixel under
            // this one sees from frame to frame: keep the history if it saw the surface of any
            // of the four rendered pixels around this one
            let corner = vec2i(floor(inPos - 0.5));
            for (var k = 0; k < 4; k++) {
                let q = clamp(corner + vec2i(k & 1, k >> 1), vec2i(0), inSize - 1);
                let quv = (vec2f(q) + 0.5) / p.inputSize;
                mismatch = min(mismatch, depthMismatch(historyDepth, previousViewDepth(quv, textureLoad(depthTex, q, 0))));
            }
        }
        feedback *= 1.0 - smoothstep(0.02, 0.08, mismatch);
        // the current frame's share, less where none of its samples is close to this pixel
        // (the total weight is over 1 at 1:1, whatever the jitter)
        let currentWeight = (1.0 - feedback) * saturate(filterWeight);
        result = mix(current, history, feedback / max(feedback + currentWeight, 1e-8));
    }

    var rgb = fromWorking(result);
    if (any(rgb != rgb)) { rgb = vec3f(0.0); }   // NaN guard: never poison the history
    textureStore(outputTex, ip, vec4f(rgb, 1.0));
    textureStore(historyOut, ip, vec4f(rgb, min(viewDepth, 65000.0)));
}
