// Screen-space ambient occlusion, the blur across (after ssao_common.wgsl and ssao_blur.wgsl).

@group(0) @binding(4) var aoOut : texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= params.size)) { return; }
    let px = vec2i(gid.xy);
    textureStore(aoOut, px, vec4f(ssaoBlur(px, vec2i(1, 0)), 0.0, 0.0));
}
