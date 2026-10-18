// Full-resolution composite: the sharp image where the CoC is under half a pixel, a small
// full-resolution gather where it is under a few (the half-resolution layers' 2-pixel texels
// would stair-step the silhouettes there), the gathered background beyond (upsampled
// bilaterally by CoC, so the half-resolution result does not smear across depth edges), and the
// foreground layer over all.

@group(0) @binding(0) var colorTex  : texture_2d<f32>;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var bgTex     : texture_2d<f32>;
@group(0) @binding(3) var fgTex     : texture_2d<f32>;
@group(0) @binding(4) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> p : DofParams;
@group(0) @binding(6) var tileTex   : texture_2d<f32>;   // r near, g far: the largest CoC (half px) reaching each tile

// The CoC (pixels) up to which the full-resolution gather draws the background layer, fading
// into the half-resolution one over the last pixel.
const SLIGHT: f32 = 3.0;

// The sharp image blurred by the CoC, as the half-resolution gather does it: each sample spreads
// over its own disc (its light shared by the disc's area), but no farther than this pixel's own
// disc when it lies behind this pixel's surface. It reaches as far as the largest CoC near the
// pixel (its tile's), at most SLIGHT.
fn slightBlur(px: vec2i, sharp: vec3f, coc: f32) -> vec3f {
    let tile = textureLoad(tileTex, vec2u(px) / 16u, 0).rg * 2.0;
    let reach = i32(min(max(max(tile.r, tile.g), abs(coc)), SLIGHT) + 0.5);
    if (reach == 0) { return sharp; }
    let absC = abs(coc);
    let ownW = cover(0.0, absC) / max(3.14159 * absC * absC, 1.0);
    var sum = sharp * ownW;
    var weight = ownW;
    let lim = vec2i(i32(p.width), i32(p.height)) - 1;
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let d = length(vec2f(f32(dx), f32(dy)));
            if (d < 0.5 || d > f32(reach) + 0.5) { continue; }
            let q = clamp(px + vec2i(dx, dy), vec2i(0), lim);
            let cs = clamp(cocFromDepth(loadDepth(depthTex, q)), -SLIGHT, SLIGHT);
            let absS = abs(cs);
            let tol = layerTolerance(min(absC, absS));
            let behind = saturate((cs - coc - tol) / tol);
            let spread = mix(absS, min(absS, absC), behind);
            let w = cover(d, spread) / max(3.14159 * absS * absS, 1.0);
            sum += textureLoad(colorTex, q, 0).rgb * w;
            weight += w;
        }
    }
    return sum / max(weight, 1e-6);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let sharp = textureLoad(colorTex, gid.xy, 0);
    let coc = clamp(cocFromDepth(loadDepth(depthTex, vec2i(gid.xy))), -p.maxCoc, p.maxCoc);

    let hs = halfSize();
    let h = (vec2f(gid.xy) + 0.5) * 0.5 - 0.5;
    let b = floor(h);
    let f = h - b;
    var bg = vec3f(0.0);
    var bgW = 0.0;
    var fg = vec4f(0.0);
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let t = vec2u(clamp(vec2i(b + o), vec2i(0), vec2i(hs) - 1));
        let bilinear = (select(1.0 - f.x, f.x, o.x > 0.5)) * (select(1.0 - f.y, f.y, o.y > 0.5));
        let bgT = textureLoad(bgTex, t, 0);
        // a sharp texel must not tint the blurred background next to it, nor the reverse
        let w = bilinear * (1e-4 + layerMatch(coc, bgT.a * 2.0));
        bg += bgT.rgb * w;
        bgW += w;
        fg += textureLoad(fgTex, t, 0) * bilinear;
    }
    bg = select(sharp.rgb, bg / bgW, bgW > 1e-6);
    var base = bg;
    if (abs(coc) < SLIGHT) {
        base = mix(slightBlur(vec2i(gid.xy), sharp.rgb, coc), bg, smoothstep(SLIGHT - 1.0, SLIGHT, abs(coc)));
    }
    var result = base * (1.0 - fg.a) + fg.rgb;
    switch (p.debugView) {
        case 1u: { result = base; }
        case 2u: { result = fg.rgb / max(fg.a, 1e-3); }
        case 3u: { result = vec3f(fg.a); }
        // CoC: red in front of the focus plane, blue behind, a pixel of CoC per 1/16
        case 4u: { result = vec3f(saturate(-coc / 16.0), 1.0 - saturate(abs(coc) / 1.5), saturate(coc / 16.0)); }
        default: {}
    }
    textureStore(outputTex, gid.xy, vec4f(result, sharp.a));
}
