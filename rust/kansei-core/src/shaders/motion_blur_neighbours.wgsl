// Per tile: the longest blur among the tiles whose streaks reach it (the adjacent ones always,
// farther ones when their streak passes within reach), which bounds the gather: a fast object
// smears that far over its neighbours. And the shortest blur of the adjacent tiles: when it is
// about as long as the longest, everything around moves alike and the gather needs no weights.

@group(0) @binding(0) var tileTex      : texture_2d<f32>;
@group(0) @binding(1) var neighbourOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p   : MotionBlurParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let tiles = tileCount();
    if (any(gid.xy >= tiles)) { return; }
    let reach = max(i32(ceil(p.maxRadius / f32(TILE))), 1);
    var best = vec2f(0.0);
    var bestLen2 = 0.0;
    var shortest = 1e9;
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let t = vec2i(gid.xy) + vec2i(dx, dy);
            if (any(t < vec2i(0)) || any(t >= vec2i(tiles))) { continue; }
            let tv = textureLoad(tileTex, t, 0);
            let v = tv.xy;
            let len2 = dot(v, v);
            if (max(abs(dx), abs(dy)) <= 1) { shortest = min(shortest, tv.z); }
            if (len2 <= bestLen2) { continue; }
            if (max(abs(dx), abs(dy)) > 1) {
                // the streak c +- v of that tile's centre c: does it pass within two tile
                // half-diagonals of this tile's centre (any of its pixels to any of ours)?
                let c = vec2f(f32(dx), f32(dy)) * f32(TILE);
                let h = clamp(-dot(c, v) / len2, -1.0, 1.0);
                if (length(c + v * h) > 1.4143 * f32(TILE)) { continue; }
            }
            best = v;
            bestLen2 = len2;
        }
    }
    textureStore(neighbourOut, gid.xy, vec4f(best, shortest, 0.0));
}
