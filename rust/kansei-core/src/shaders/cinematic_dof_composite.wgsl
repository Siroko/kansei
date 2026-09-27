// Full-resolution composite: the sharp image where the CoC is under a pixel, the gathered
// background elsewhere (upsampled bilaterally by CoC, so the half-resolution result does not
// smear across depth edges), and the foreground layer over both.

@group(0) @binding(0) var colorTex  : texture_2d<f32>;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var bgTex     : texture_2d<f32>;
@group(0) @binding(3) var fgTex     : texture_2d<f32>;
@group(0) @binding(4) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> p : DofParams;

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
        // only texels of about this pixel's CoC: a sharp foreground texel must not tint the
        // blurred background next to it, nor the reverse
        let w = bilinear * (1e-4 + exp(-abs(bgT.a * 2.0 - coc) / layerTolerance(coc)));
        bg += bgT.rgb * w;
        bgW += w;
        fg += textureLoad(fgTex, t, 0) * bilinear;
    }
    bg = select(sharp.rgb, bg / bgW, bgW > 1e-6);
    let base = mix(sharp.rgb, bg, smoothstep(0.5, 1.5, abs(coc)));
    textureStore(outputTex, gid.xy, vec4f(base * (1.0 - fg.a) + fg.rgb, sharp.a));
}
