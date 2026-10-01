// A 3x3 tent over the gathered layers at half resolution, which averages the gather's sampling
// noise. The background only mixes texels of similar CoC, so it doesn't blur across depth edges;
// the near field (premultiplied) is filtered as is. The scattered highlights listed in this
// pixel's bin are then added, crisp, to their layers.

@group(0) @binding(0) var bgIn  : texture_2d<f32>;
@group(0) @binding(1) var fgIn  : texture_2d<f32>;
@group(0) @binding(2) var bgOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var fgOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> p : DofParams;
@group(0) @binding(5) var<storage, read> sprites : array<Sprite>;
@group(0) @binding(6) var<storage, read> binCount : array<u32>;
@group(0) @binding(7) var<storage, read> binList : array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    if (gid.x >= hs.x || gid.y >= hs.y) { return; }
    let center = textureLoad(bgIn, gid.xy, 0);
    let lim = vec2i(hs) - 1;
    var bg = vec3f(0.0);
    var bgW = 0.0;
    var fg = vec4f(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let c = vec2u(clamp(vec2i(gid.xy) + vec2i(dx, dy), vec2i(0), lim));
            let tent = f32((2 - abs(dx)) * (2 - abs(dy))) / 16.0;
            let b = textureLoad(bgIn, c, 0);
            // sharp texels keep themselves; blurred ones average with their like
            let w = tent * exp(-abs(b.a - center.a) / layerTolerance(center.a)) * select(1.0, 0.0, abs(center.a) < 0.5 && (dx != 0 || dy != 0));
            bg += b.rgb * w;
            bgW += w;
            fg += textureLoad(fgIn, c, 0) * tent;
        }
    }
    bg /= bgW;
    let bin = (gid.y / BIN) * binGrid().x + gid.x / BIN;
    let listed = min(binCount[bin], BIN_CAPACITY);
    let pixel = vec2f(gid.xy) + 0.5;
    for (var k = 0u; k < listed; k++) {
        let s = sprites[binList[bin * BIN_CAPACITY + k]];
        let light = spriteLight(s, pixel, abs(center.a));
        if (s.layer > 0.5) { fg += vec4f(light, 0.0); } else { bg += light; }
    }
    textureStore(bgOut, gid.xy, vec4f(bg, center.a));
    textureStore(fgOut, gid.xy, fg);
}
