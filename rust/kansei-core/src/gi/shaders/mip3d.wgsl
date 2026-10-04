// One mip of a 3D chain (gi::Mip3d): WebGPU generates no mips, so each level is a dispatch that
// reads level n (a sampled view of that level alone) and writes level n + 1 (a write-only storage
// view): two subresources of one texture. A 2x2x2 box filter, which is exact for premultiplied
// radiance and opacity; on an odd side the last source voxel is read twice.
@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= textureDimensions(dst))) { return; }
    let last = textureDimensions(src) - 1u;
    var sum = vec4f(0.0);
    for (var k = 0u; k < 8u; k++) {
        let o = vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        sum += textureLoad(src, min(2u * gid + o, last), 0);
    }
    textureStore(dst, gid, sum * 0.125);
}
