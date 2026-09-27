// The next coarser level of a half-resolution layer: colour weighted by coverage, the mean
// coverage, and the CoC weighted by coverage (the near field) or the closest covered one (the
// background, for its no-halo rule).

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> p : DofParams;

fn downsample(gid: vec2u, near: bool) {
    let ds = textureDimensions(dstTex);
    if (gid.x >= ds.x || gid.y >= ds.y) { return; }
    let lim = vec2i(textureDimensions(srcTex, 0)) - 1;
    var color = vec3f(0.0);
    var weight = 0.0;
    var cocSum = 0.0;
    var cocMin = 1e9;
    for (var i = 0u; i < 4u; i++) {
        let s = textureLoad(srcTex, min(vec2i(gid) * 2 + vec2i(i32(i & 1u), i32(i >> 1u)), lim), 0);
        let a = layerCoverage(s.a);
        let c = layerCoc(s.a);
        color += s.rgb * a;
        weight += a;
        cocSum += c * a;
        if (a > 0.0) { cocMin = min(cocMin, c); }
    }
    var coc = cocSum / max(weight, 1e-6);
    if (!near && cocMin < 1e8) { coc = cocMin; }
    textureStore(dstTex, gid, vec4f(color / max(weight, 1e-6), packLayer(coc, weight * 0.25)));
}

@compute @workgroup_size(8, 8)
fn downsampleNear(@builtin(global_invocation_id) gid : vec3u) {
    downsample(gid.xy, true);
}

@compute @workgroup_size(8, 8)
fn downsampleFar(@builtin(global_invocation_id) gid : vec3u) {
    downsample(gid.xy, false);
}
