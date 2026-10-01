// Planar reflection, mip chain: each level is the one above filtered by a 4x4 tent (four
// bilinear taps), so rough water can read a smoothly blurred reflection.

@group(0) @binding(0) var src        : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;
@group(0) @binding(2) var dst        : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(dst);
    if (any(gid.xy >= size)) { return; }
    let texel = 1.0 / vec2f(textureDimensions(src));
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
    var c = textureSampleLevel(src, srcSampler, uv + vec2f(-0.75, -0.75) * texel, 0.0);
    c += textureSampleLevel(src, srcSampler, uv + vec2f( 0.75, -0.75) * texel, 0.0);
    c += textureSampleLevel(src, srcSampler, uv + vec2f(-0.75,  0.75) * texel, 0.0);
    c += textureSampleLevel(src, srcSampler, uv + vec2f( 0.75,  0.75) * texel, 0.0);
    textureStore(dst, gid.xy, c * 0.25);
}
