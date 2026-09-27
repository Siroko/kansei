// The next coarser level of the half-resolution chain: box-averaged colour, and the smallest
// (closest) CoC of the four, still in half-resolution pixels.

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p : DofParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let ds = textureDimensions(dstTex);
    if (gid.x >= ds.x || gid.y >= ds.y) { return; }
    let lim = vec2i(textureDimensions(srcTex, 0)) - 1;
    var sum = vec3f(0.0);
    var coc = 1e9;
    for (var i = 0u; i < 4u; i++) {
        let s = textureLoad(srcTex, min(vec2i(gid.xy) * 2 + vec2i(i32(i & 1u), i32(i >> 1u)), lim), 0);
        sum += s.rgb;
        coc = min(coc, s.a);
    }
    textureStore(dstTex, gid.xy, vec4f(sum * 0.25, coc));
}
