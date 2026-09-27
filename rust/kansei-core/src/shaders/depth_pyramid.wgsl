// Hierarchical depth pyramid (Hi-Z). Each texel of mip 0 reduces a 2x2 block of the depth buffer,
// each later mip a 2x2 block of the mip before. Mip 0 is a power of two in each axis, so every mip
// halves exactly, and loads past the depth buffer repeat its edge, so texel (x, y) of mip L covers
// exactly the depth pixels [x, x + 1) * 2^(L + 1) (clipped to the buffer): a query maps pixels to
// texels by shifting, whatever the buffer's size.
//
// Substituted by depth_pyramid.rs: MODE_VALUE (0 keeps the maximum, 1 the minimum, 2 both, as
// r = min and g = max) and FORMAT (its storage format).

const MODE : u32 = MODE_VALUE;

@group(0) @binding(0) var depth    : texture_depth_2d;
@group(0) @binding(1) var previous : texture_2d<f32>;
@group(0) @binding(2) var dst      : texture_storage_2d<FORMAT, write>;

fn store(p : vec2u, v : vec2f) {
    switch MODE {
        case 0u: { textureStore(dst, p, vec4f(v.y, 0.0, 0.0, 0.0)); }
        case 1u: { textureStore(dst, p, vec4f(v.x, 0.0, 0.0, 0.0)); }
        default: { textureStore(dst, p, vec4f(v, 0.0, 0.0)); }
    }
}

// (min, max) of four (min, max) pairs
fn reduce4(a : vec2f, b : vec2f, c : vec2f, d : vec2f) -> vec2f {
    return vec2f(min(min(a.x, b.x), min(c.x, d.x)), max(max(a.y, b.y), max(c.y, d.y)));
}

@compute @workgroup_size(8, 8)
fn from_depth(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= textureDimensions(dst))) { return; }
    let last = textureDimensions(depth) - 1u;
    let a = min(gid.xy * 2u, last);
    let b = min(gid.xy * 2u + 1u, last);
    let d = vec4f(textureLoad(depth, a, 0), textureLoad(depth, vec2u(b.x, a.y), 0),
                  textureLoad(depth, vec2u(a.x, b.y), 0), textureLoad(depth, b, 0));
    store(gid.xy, reduce4(d.xx, d.yy, d.zz, d.ww));
}

// a texel of the previous mip as (min, max)
fn texel(p : vec2u) -> vec2f {
    let t = textureLoad(previous, p, 0);
    if (MODE == 2u) { return t.xy; }
    return t.xx;
}

@compute @workgroup_size(8, 8)
fn from_mip(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= textureDimensions(dst))) { return; }
    let last = textureDimensions(previous) - 1u;
    let a = min(gid.xy * 2u, last);
    let b = min(gid.xy * 2u + 1u, last);
    store(gid.xy, reduce4(texel(a), texel(vec2u(b.x, a.y)), texel(vec2u(a.x, b.y)), texel(b)));
}
