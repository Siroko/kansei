// Unreal's local exposure (its bilateral method), the inputs the display transform (tonemap.wgsl)
// reads, per frame from the chain's HDR input, as PostProcessHistogram.usf (BILATERAL_GRID) and
// PostProcessLocalExposure.usf build them:
//   grid:          the bilateral grid: per cell of 64 x 64 half-resolution texels (the mean of
//                  2 x 2 pixels), 32 bins of log2 scene luminance, each holding the sum of its
//                  texels' log2 luminance and their count, split between the two nearest bins;
//                  both divided by the cell's texel count;
//   logLuminance:  log2 of the luminance of the picture at 1/32 of its size (each texel the mean
//                  colour of 32 x 32 pixels);
//   blurX, blurY:  that blurred by Unreal's Gaussian, exp(-16.7 (x/r)^2), mirrored at the edges.
// Luminance is the mean of r, g and b (r.AutoExposure.LuminanceMethod 0), floored.

@group(0) @binding(0) var inputTex      : texture_2d<f32>;
@group(0) @binding(1) var<uniform> p    : ToneMapParams;
@group(0) @binding(2) var linearSampler : sampler;
@group(0) @binding(3) var gridOut       : texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var logOut        : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var logIn         : texture_2d<f32>;

const BINS : u32 = 32u;
// fixed point of the bins' sums (workgroup atomics are integer)
const FIXED : f32 = 4096.0;

var<workgroup> binWeight : array<atomic<u32>, 32>;
var<workgroup> binLog    : array<atomic<i32>, 32>;

fn sceneLogLuminance(c: vec3f) -> f32 {
    return log2(max(dot(c, vec3f(1.0 / 3.0)), exp2(p.localExposure2.w)));
}

@compute @workgroup_size(8, 8)
fn grid(@builtin(workgroup_id) cell : vec3u, @builtin(local_invocation_id) lid : vec3u, @builtin(local_invocation_index) li : u32) {
    if (li < BINS) {
        atomicStore(&binWeight[li], 0u);
        atomicStore(&binLog[li], 0);
    }
    workgroupBarrier();
    let size = vec2u(p.width, p.height);
    let halfSize = (size + 1u) / 2u;
    // each invocation 8 x 8 of the cell's texels
    let first = cell.xy * LOCAL_CELL + lid.xy * 8u;
    for (var k = 0u; k < 64u; k++) {
        let t = first + vec2u(k & 7u, k >> 3u);
        if (any(t >= halfSize)) { continue; }
        // the texel's 2 x 2 pixels, averaged by one bilinear tap at their shared corner
        let c = textureSampleLevel(inputTex, linearSampler, (vec2f(t * 2u) + 1.0) / vec2f(size), 0.0).rgb;
        let logL = sceneLogLuminance(c);
        let f = saturate(logL * p.localExposure2.y + p.localExposure2.z) * f32(BINS - 1u);
        let b0 = min(u32(f), BINS - 1u);
        let w1 = f - f32(b0);
        let b1 = min(b0 + 1u, BINS - 1u);
        let l = clamp(logL, -64.0, 64.0) * FIXED;
        atomicAdd(&binWeight[b0], u32(round((1.0 - w1) * FIXED)));
        atomicAdd(&binLog[b0], i32(round(l * (1.0 - w1))));
        atomicAdd(&binWeight[b1], u32(round(w1 * FIXED)));
        atomicAdd(&binLog[b1], i32(round(l * w1)));
    }
    workgroupBarrier();
    if (li < BINS) {
        let w = f32(atomicLoad(&binWeight[li])) / FIXED;
        let s = f32(atomicLoad(&binLog[li])) / FIXED;
        textureStore(gridOut, vec3u(cell.xy, li), vec4f(s / LOCAL_CELL_TEXELS, w / LOCAL_CELL_TEXELS, 0.0, 1.0));
    }
}

@compute @workgroup_size(8, 8)
fn logLuminance(@builtin(global_invocation_id) gid : vec3u) {
    let dims = textureDimensions(logOut);
    if (any(gid.xy >= dims)) { return; }
    let size = vec2u(p.width, p.height);
    // the mean colour of this texel's 32 x 32 pixels: 16 x 16 bilinear taps of 2 x 2
    var sum = vec3f(0.0);
    var n = 0.0;
    for (var k = 0u; k < 256u; k++) {
        let px = gid.xy * 32u + vec2u(k & 15u, k >> 4u) * 2u;
        if (any(px >= size)) { continue; }
        sum += textureSampleLevel(inputTex, linearSampler, (vec2f(px) + 1.0) / vec2f(size), 0.0).rgb;
        n += 1.0;
    }
    textureStore(logOut, gid.xy, vec4f(sceneLogLuminance(sum / max(n, 1.0)), 0.0, 0.0, 1.0));
}

// Mirrored addressing, as Unreal's AM_Mirror: -1 is 0, size is size - 1.
fn mirror(c : i32, size : i32) -> i32 {
    var m = c;
    if (m < 0) { m = -m - 1; }
    if (m >= size) { m = 2 * size - m - 1; }
    return clamp(m, 0, size - 1);
}

fn blur(gid : vec2u, axis : vec2i) {
    let dims = vec2i(textureDimensions(logIn, 0));
    if (any(vec2i(gid) >= dims)) { return; }
    let radius = p.localExposure3.z;
    let taps = i32(p.localExposure3.w);
    var sum = 0.0;
    var weights = 0.0;
    for (var x = -taps; x <= taps; x++) {
        let w = exp(-16.7 * (f32(x) / radius) * (f32(x) / radius));
        let c = vec2i(gid) + axis * x;
        sum += w * textureLoad(logIn, vec2i(mirror(c.x, dims.x), mirror(c.y, dims.y)), 0).r;
        weights += w;
    }
    textureStore(logOut, gid, vec4f(sum / weights, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn blurX(@builtin(global_invocation_id) gid : vec3u) {
    blur(gid.xy, vec2i(1, 0));
}

@compute @workgroup_size(8, 8)
fn blurY(@builtin(global_invocation_id) gid : vec3u) {
    blur(gid.xy, vec2i(0, 1));
}
