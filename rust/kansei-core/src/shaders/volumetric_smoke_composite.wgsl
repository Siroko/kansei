// The fog's composite with smoke (VolumetricFogEffect::set_smoke), appended to
// volumetric_fog_composite.wgsl: its bindings 0-6, and the smoke's march (or its history), upsampled
// by depth as the shafts are:
//   out = Ts * (scene * fog.a + fog.rgb + shafts) + R
// (volumetric_smoke_march.wgsl). With no smoke on a pixel (Ts = 1, R = 0) it is `main`'s.

@group(0) @binding(7) var smokeLightTex : texture_2d<f32>;   // (R, Ts), half resolution
@group(0) @binding(8) var smokeAuxTex   : texture_2d<f32>;   // x: the texel's linear depth

// The half-resolution smoke at a pixel: the four texels around it, weighted bilinearly and by how
// close their depth is to the pixel's, so the smoke stops at the edges of what stands in it.
fn upsampleSmoke(coord: vec2u, linearDepth: f32) -> vec4f {
    let size = vec2i(textureDimensions(smokeLightTex));
    let pos = (vec2f(coord) + 0.5) * 0.5 - 0.5;
    let base = floor(pos);
    let f = pos - base;
    var sum = vec4f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let c = clamp(vec2i(base + o), vec2i(0), size - 1);
        let s = textureLoad(smokeLightTex, c, 0);
        let d = textureLoad(smokeAuxTex, c, 0).x;
        let bilinear = select(1.0 - f.x, f.x, o.x > 0.5) * select(1.0 - f.y, f.y, o.y > 0.5);
        let w = bilinear * (exp(-abs(d - linearDepth) / (0.05 * linearDepth + 0.1)) + 1e-4);
        sum += s * w;
        weight += w;
    }
    return sum / max(weight, 1e-6);
}

@compute @workgroup_size(8, 8)
fn compositeSmoke(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    if (f32(coord.x) >= cp.screenWidth || f32(coord.y) >= cp.screenHeight) { return; }

    let sceneColor = textureLoad(inputTex, coord, 0);
    let depth      = textureLoad(depthTex, coord, 0);

    let linearDepth = ndcToLinearDepth(depth, cp.cameraNear, cp.cameraFar);
    let sliceFloat = depthToSlice(linearDepth, cp.gridNear, cp.gridFar, cp.gridD);

    let jitter = screenHash(vec2f(coord)) - 0.5;
    let w = clamp((sliceFloat + jitter) / cp.gridD, 0.0, 1.0);
    let uv = (vec2f(coord) + 0.5) / vec2f(cp.screenWidth, cp.screenHeight);

    let fog = textureSampleLevel(accumTex, accumSampler, vec3f(uv, w), 0.0);
    var shafts = vec3f(0.0);
    if (cp.shafts > 0.5) { shafts = upsampleShafts(coord, linearDepth); }
    let smoke = upsampleSmoke(coord, linearDepth);
    textureStore(outputTex, coord, vec4f(smoke.a * (sceneColor.rgb * fog.a + fog.rgb + shafts) + smoke.rgb, sceneColor.a));
}
