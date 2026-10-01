// Highlight extraction for scattered bokeh, per layer. A half-resolution texel much brighter than
// its surroundings in its own layer (a lamp, a headlight, a glint) keeps only `contrast` times
// their luminance for the gather; the rest of its energy is appended as a sprite, drawn later as
// a crisp, aperture-shaped disc of its CoC (a gather only hits such a small source with a few
// samples per pixel, which leaves its bokeh grainy). The sprite is listed in the bins it
// reaches (cinematic_dof_sprites.wgsl). Luminance is weighted by the layer's
// coverage, and the surroundings are the darker of the 3x3 neighbours and a ring 4 texels out,
// so a source a few texels wide scatters from its whole area while a large bright area (the sky)
// does not. Texels with little blur are left alone.

@group(0) @binding(0) var nearRaw : texture_2d<f32>;
@group(0) @binding(1) var farRaw  : texture_2d<f32>;
@group(0) @binding(2) var nearOut : texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var farOut  : texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> p : DofParams;
@group(0) @binding(5) var<storage, read_write> sprites : array<Sprite>;
@group(0) @binding(6) var<storage, read_write> spriteCount : atomic<u32>;
@group(0) @binding(7) var<uniform> hp : HighlightParams;
@group(0) @binding(8) var<storage, read_write> binCount : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> binList : array<u32>;

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Premultiplied luminance of a layer texel.
fn energy(t: vec4f) -> f32 {
    return luminance(t.rgb) * layerCoverage(t.a);
}

fn extract(s: vec4f, c: vec2i, near: bool) -> vec4f {
    let coc = layerCoc(s.a);
    let coverage = layerCoverage(s.a);
    if (hp.enabled == 0u || abs(coc) < hp.minCoc || coverage <= 0.0) { return s; }
    let lim = vec2i(halfSize()) - 1;
    var neighbours = 0.0;
    var ring = 0.0;
    let ringOffsets = array<vec2i, 8>(vec2i(4, 0), vec2i(-4, 0), vec2i(0, 4), vec2i(0, -4), vec2i(3, 3), vec2i(-3, 3), vec2i(3, -3), vec2i(-3, -3));
    for (var i = 0u; i < 8u; i++) {
        let n = clamp(c + vec2i(i32(i % 3u) - 1, i32(i / 3u) - 1) + select(vec2i(0), vec2i(1, 1), i >= 4u), vec2i(0), lim);
        let r = clamp(c + ringOffsets[i], vec2i(0), lim);
        if (near) {
            neighbours += energy(textureLoad(nearRaw, n, 0));
            ring += energy(textureLoad(nearRaw, r, 0));
        } else {
            neighbours += energy(textureLoad(farRaw, n, 0));
            ring += energy(textureLoad(farRaw, r, 0));
        }
    }
    let keep = hp.contrast * min(neighbours, ring) / 8.0;
    let e = luminance(s.rgb) * coverage;
    if (e <= keep || e <= 0.0) { return s; }
    let index = atomicAdd(&spriteCount, 1u);
    if (index >= hp.maxSprites) { return s; }
    // list it in every bin its disc reaches; if one is full, the sprite carries no light and the
    // texel is gathered after all
    let center = vec2f(c) + 0.5;
    let reach = abs(coc) + 1.0;
    let last = vec2i(binGrid()) - 1;
    let lo = clamp(vec2i(floor((center - reach) / f32(BIN))), vec2i(0), last);
    let hi = clamp(vec2i(floor((center + reach) / f32(BIN))), vec2i(0), last);
    var listed = true;
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let bin = u32(y) * binGrid().x + u32(x);
            let slot = atomicAdd(&binCount[bin], 1u);
            if (slot < BIN_CAPACITY) { binList[bin * BIN_CAPACITY + slot] = index; } else { listed = false; }
        }
    }
    if (!listed) {
        sprites[index] = Sprite(center, abs(coc), select(0.0, 1.0, near), vec4f(0.0));
        return s;
    }
    let kept = keep / e;
    sprites[index] = Sprite(center, abs(coc), select(0.0, 1.0, near), vec4f(s.rgb * (1.0 - kept) * coverage, 0.0));
    return vec4f(s.rgb * kept, s.a);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let hs = halfSize();
    if (gid.x >= hs.x || gid.y >= hs.y) { return; }
    let c = vec2i(gid.xy);
    textureStore(nearOut, gid.xy, extract(textureLoad(nearRaw, c, 0), c, true));
    textureStore(farOut, gid.xy, extract(textureLoad(farRaw, c, 0), c, false));
}
