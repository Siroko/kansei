// Splits each 2x2 block of full-resolution pixels between the two layers by CoC: its near-field
// part (colour, mean CoC magnitude, coverage) and its background part (colour, the closest CoC,
// coverage). Colours average only their own layer's pixels, and each layer keeps the fraction
// of the block it covers, so a pixel's energy is counted once and never mixed across depth: the
// background seen through a porous near object (leaves, a fence) stays in the background layer.
// `downsample` builds the coarser levels the same way, for the gather's large discs.
// Each workgroup is one 8x8 tile: it also writes the tile's largest near-field and background
// CoC radius, which bound the gather.

@group(0) @binding(0) var colorTex : texture_2d<f32>;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var nearOut  : texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> p : DofParams;
@group(0) @binding(4) var farOut   : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var tileOut  : texture_storage_2d<rgba16float, write>;   // r near, g far (half px)

// the tile's largest CoCs, as the bits of non-negative floats (which order as integers)
var<workgroup> nearMax : atomic<u32>;
var<workgroup> farMax  : atomic<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u, @builtin(workgroup_id) tile : vec3u, @builtin(local_invocation_index) local : u32) {
    if (local == 0u) {
        atomicStore(&nearMax, 0u);
        atomicStore(&farMax, 0u);
    }
    workgroupBarrier();
    let hs = halfSize();
    if (all(gid.xy < hs)) { prefilter(gid.xy); }
    workgroupBarrier();
    if (local == 0u) {
        let limit = p.maxCoc * 0.5;
        let r = min(bitcast<f32>(atomicLoad(&nearMax)), limit);
        let g = min(bitcast<f32>(atomicLoad(&farMax)), limit);
        textureStore(tileOut, tile.xy, vec4f(r, g, 0.0, 0.0));
    }
}

fn prefilter(gid: vec2u) {
    let lim = vec2i(i32(p.width) - 1, i32(p.height) - 1);
    let base = vec2i(gid) * 2;
    var nearColor = vec3f(0.0);
    var nearCoc = 0.0;
    var nearW = 0.0;
    var farColor = vec3f(0.0);
    var farW = 0.0;
    var farCoc = 1e9;
    var anyCoc = 1e9;
    for (var i = 0u; i < 4u; i++) {
        let fc = min(base + vec2i(i32(i & 1u), i32(i >> 1u)), lim);
        let c = min(textureLoad(colorTex, fc, 0).rgb, vec3f(65000.0));
        let coc = clamp(cocFromDepth(loadDepth(depthTex, fc)), -p.maxCoc, p.maxCoc);
        let n = nearWeight(coc);
        nearColor += c * n;
        nearCoc += -coc * n;
        nearW += n;
        farColor += c * (1.0 - n);
        farW += 1.0 - n;
        if (n < 0.5) { farCoc = min(farCoc, coc); }
        anyCoc = min(anyCoc, coc);
    }
    if (farCoc > 1e8) { farCoc = max(anyCoc, 0.0); }
    let near = select(vec3f(0.0), nearColor / max(nearW, 1e-6), nearW > 1e-6);
    let far = select(vec3f(0.0), farColor / max(farW, 1e-6), farW > 1e-6);
    let nearA = packLayer(nearCoc / max(nearW, 1e-6) * 0.5, nearW * 0.25);
    let farA = packLayer(farCoc * 0.5, farW * 0.25);
    textureStore(nearOut, gid, vec4f(near, nearA));
    textureStore(farOut, gid, vec4f(far, farA));
    if (layerCoverage(nearA) > 0.0) { atomicMax(&nearMax, bitcast<u32>(max(layerCoc(nearA), 0.0))); }
    if (layerCoverage(farA) > 0.0) { atomicMax(&farMax, bitcast<u32>(abs(layerCoc(farA)))); }
}
