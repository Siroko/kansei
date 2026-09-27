// Half-resolution colour and CoC: the box average of four full-resolution pixels (which keeps
// every pixel's energy: a one-pixel light becomes a quarter-bright texel, not a four-pixel one)
// with the CoC of the closest of them, so a silhouette texel belongs to the surface in front.
// `downsample` builds the coarser levels the same way, for the gather's large discs.

@group(0) @binding(0) var colorTex : texture_2d<f32>;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var halfOut  : texture_storage_2d<rgba16float, write>;   // rgb, a = signed CoC (half px)
@group(0) @binding(3) var<uniform> p : DofParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    if (gid.x >= hs.x || gid.y >= hs.y) { return; }
    let lim = vec2i(i32(p.width) - 1, i32(p.height) - 1);
    let base = vec2i(gid.xy) * 2;
    var sum = vec3f(0.0);
    var minDepth = 2.0;
    for (var i = 0u; i < 4u; i++) {
        let fc = min(base + vec2i(i32(i & 1u), i32(i >> 1u)), lim);
        sum += min(textureLoad(colorTex, fc, 0).rgb, vec3f(65000.0));
        minDepth = min(minDepth, textureLoad(depthTex, fc, 0));
    }
    textureStore(halfOut, gid.xy, vec4f(sum * 0.25, cocFromDepth(minDepth) * 0.5));
}
