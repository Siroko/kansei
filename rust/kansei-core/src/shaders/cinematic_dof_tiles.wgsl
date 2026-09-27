// The largest near-field and background CoC radius of each 8x8 half-resolution tile (written by
// the prefilter), dilated over the tiles within the largest CoC's reach. They bound each layer's
// gather radius: a blurred foreground spills that far over its neighbours.

@group(0) @binding(0) var srcTex  : texture_2d<f32>;   // the prefilter's tiles
@group(0) @binding(1) var tileOut : texture_storage_2d<rgba16float, write>;   // r near, g far (half px)
@group(0) @binding(2) var<uniform> p : DofParams;

const TILE : u32 = 8u;

@compute @workgroup_size(8, 8)
fn dilate(@builtin(global_invocation_id) gid : vec3u) {
    let ts = (halfSize() + TILE - 1u) / TILE;
    if (gid.x >= ts.x || gid.y >= ts.y) { return; }
    let reach = i32(ceil(p.maxCoc * 0.5 / f32(TILE)));
    var m = vec2f(0.0);
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let t = vec2i(gid.xy) + vec2i(dx, dy);
            if (any(t < vec2i(0)) || any(t >= vec2i(ts))) { continue; }
            let v = textureLoad(srcTex, vec2u(t), 0).rg;
            // a tile d tiles away can only reach this one with a CoC above (d - 1) tiles
            let gap = f32(max(max(abs(dx), abs(dy)) - 1, 0) * i32(TILE));
            m = max(m, select(vec2f(0.0), v, v > vec2f(gap)));
        }
    }
    textureStore(tileOut, gid.xy, vec4f(m, 0.0, 0.0));
}
