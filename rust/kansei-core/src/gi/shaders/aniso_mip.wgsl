// Anisotropic mips (gi::VoxelVolume::set_anisotropic_mips, Crassin et al. 2011): from mip 1 up,
// six chains, one per direction a cone can travel along an axis (+x, +y, +z, -x, -y, -z). Each
// voxel of a chain is its 2x2x2 children composited front to back along that direction (the near
// child over the far one), then averaged across the four pairs. A wall bright on one face and
// dark on the other then shows each cone the face it meets first, where an isotropic mean shows
// both their average, halving the light of a lit room's walls at coarse mips; and a wall facing
// the direction stays opaque at every level, however thin. Cones read the chains through
// voxel_irradiance.wgsl's voxelAnisoSample.
//
// Storage textures are four per stage at most: one dispatch writes the three positive directions,
// another the three negative ones. Level 1 reads the isotropic mip 0 (`first_*`), the levels
// above the same direction's level below (`down_*`).

@group(0) @binding(0) var srcIso : texture_3d<f32>;
@group(0) @binding(1) var srcX : texture_3d<f32>;
@group(0) @binding(2) var srcY : texture_3d<f32>;
@group(0) @binding(3) var srcZ : texture_3d<f32>;
@group(0) @binding(4) var dstX : texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var dstY : texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var dstZ : texture_storage_3d<rgba16float, write>;

// premultiplied `near` over `far`
fn over(near: vec4f, far: vec4f) -> vec4f {
    return near + (1.0 - near.a) * far;
}

// Child voxel `o` of `gid` in a source whose texels are `src` (clamped to its last texel).
fn child(src: texture_3d<f32>, gid: vec3u, o: vec3u) -> vec4f {
    return textureLoad(src, min(2u * gid + o, textureDimensions(src) - 1u), 0);
}

// The four pairs along `axis` (0, 1, 2), each the child at `nearSide` over the other, averaged.
fn composite(src: texture_3d<f32>, gid: vec3u, axis: u32, nearSide: u32) -> vec4f {
    var sum = vec4f(0.0);
    for (var k = 0u; k < 4u; k++) {
        var near = vec3u(0u);
        // the two coordinates across the axis walk the pair; along it, the near side
        let a = k & 1u;
        let b = k >> 1u;
        if (axis == 0u) { near = vec3u(nearSide, a, b); }
        else if (axis == 1u) { near = vec3u(a, nearSide, b); }
        else { near = vec3u(a, b, nearSide); }
        var far = near;
        far[axis] = 1u - nearSide;
        sum += over(child(src, gid, near), child(src, gid, far));
    }
    return sum * 0.25;
}

// A cone travelling toward +axis meets the child at the low side first; toward -axis, the high.
@compute @workgroup_size(4, 4, 4)
fn first_pos(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= textureDimensions(dstX))) { return; }
    textureStore(dstX, gid, composite(srcIso, gid, 0u, 0u));
    textureStore(dstY, gid, composite(srcIso, gid, 1u, 0u));
    textureStore(dstZ, gid, composite(srcIso, gid, 2u, 0u));
}

@compute @workgroup_size(4, 4, 4)
fn first_neg(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= textureDimensions(dstX))) { return; }
    textureStore(dstX, gid, composite(srcIso, gid, 0u, 1u));
    textureStore(dstY, gid, composite(srcIso, gid, 1u, 1u));
    textureStore(dstZ, gid, composite(srcIso, gid, 2u, 1u));
}

@compute @workgroup_size(4, 4, 4)
fn down_pos(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= textureDimensions(dstX))) { return; }
    textureStore(dstX, gid, composite(srcX, gid, 0u, 0u));
    textureStore(dstY, gid, composite(srcY, gid, 1u, 0u));
    textureStore(dstZ, gid, composite(srcZ, gid, 2u, 0u));
}

@compute @workgroup_size(4, 4, 4)
fn down_neg(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= textureDimensions(dstX))) { return; }
    textureStore(dstX, gid, composite(srcX, gid, 0u, 1u));
    textureStore(dstY, gid, composite(srcY, gid, 1u, 1u));
    textureStore(dstZ, gid, composite(srcZ, gid, 2u, 1u));
}
