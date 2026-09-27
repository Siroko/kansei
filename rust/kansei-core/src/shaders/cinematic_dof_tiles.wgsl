// Per 8x8 half-resolution tile: the largest near (in front of focus) and far CoC, then (second
// entry point) the same maxima over the tiles within the largest CoC's reach, which bound the
// gather radius: a blurred foreground spills that far over its neighbours.

@group(0) @binding(0) var srcTex  : texture_2d<f32>;
@group(0) @binding(1) var tileOut : texture_storage_2d<rgba16float, write>;   // r near, g far (half px)
@group(0) @binding(2) var<uniform> p : DofParams;

const TILE : u32 = 8u;

@compute @workgroup_size(8, 8)
fn tiles(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    let ts = (hs + TILE - 1u) / TILE;
    if (gid.x >= ts.x || gid.y >= ts.y) { return; }
    var nearMax = 0.0;
    var farMax = 0.0;
    for (var y = 0u; y < TILE; y++) {
        for (var x = 0u; x < TILE; x++) {
            let c = min(gid.xy * TILE + vec2u(x, y), hs - 1u);
            let coc = textureLoad(srcTex, c, 0).a;
            nearMax = max(nearMax, -coc);
            farMax = max(farMax, coc);
        }
    }
    let limit = p.maxCoc * 0.5;
    textureStore(tileOut, gid.xy, vec4f(min(nearMax, limit), min(farMax, limit), 0.0, 0.0));
}

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
