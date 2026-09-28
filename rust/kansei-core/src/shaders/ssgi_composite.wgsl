// Screen-space global illumination, composite: the accumulated bounce upsampled by depth (the
// traced texels whose surface matches this pixel's), then added to the scene as albedo / pi
// times its irradiance. With the sky's lighting bound, the part of the sky the point cannot see
// is taken out of the ambient light it would otherwise get (albedo / pi times the sky's
// irradiance around its normal), so the bounce replaces the sky light it blocks. The debug view
// shows the bounce alone.

@group(0) @binding(0) var<uniform> sp : SsgiParams;
@group(0) @binding(1) var colorTex  : texture_2d<f32>;
@group(0) @binding(2) var depthTex  : texture_depth_2d;
@group(0) @binding(3) var giTex     : texture_2d<f32>;
@group(0) @binding(4) var albedoTex : texture_2d<f32>;
@group(0) @binding(5) var normalTex : texture_2d<f32>;
@group(0) @binding(6) var<uniform> sky : SkyLighting;
@group(0) @binding(7) var outTex    : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.fullSize)) { return; }
    let color = textureLoad(colorTex, gid.xy, 0);
    let px = vec2i(gid.xy);
    let depth = ssgiDepth(px);
    let albedo = textureLoad(albedoTex, px, 0).rgb;
    if (depth >= 1.0 || all(albedo <= vec3f(0.0))) {
        textureStore(outTex, gid.xy, select(color, vec4f(0.0, 0.0, 0.0, color.a), sp.debug != 0u));
        return;
    }
    let uv = (vec2f(gid.xy) + 0.5) / sp.fullSize;
    let z = -ssgiViewPos(uv, depth).z;
    let size = vec2i(sp.traceSize);
    let pos = uv * sp.traceSize - 0.5;
    let base = floor(pos);
    let f = pos - base;
    var gi = vec4f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let t = clamp(vec2i(base + o), vec2i(0), size - 1);
        let tuv = (vec2f(t) + 0.5) / sp.traceSize;
        let tz = -ssgiViewPos(tuv, ssgiDepth(ssgiPixel(tuv))).z;
        let bilinear = select(1.0 - f.x, f.x, o.x > 0.5) * select(1.0 - f.y, f.y, o.y > 0.5);
        let w = bilinear * exp(-abs(tz - z) / (0.02 * z + 0.05)) + 1e-5 * bilinear;
        gi += textureLoad(giTex, t, 0) * w;
        weight += w;
    }
    gi /= max(weight, 1e-6);
    let bounce = albedo * gi.rgb * (sp.intensity / SSGI_PI);
    if (sp.debug != 0u) {
        textureStore(outTex, gid.xy, vec4f(bounce, color.a));
        return;
    }
    var result = color.rgb + bounce;
    let n = ssgiWorldNormal(px);
    if (sp.hasSky != 0u && sp.aoStrength > 0.0 && n.w > 0.0) {
        let ambient = albedo * skyIrradiance(sky, n.xyz) / SSGI_PI;
        result = max(result - ambient * ((1.0 - gi.a) * sp.aoStrength), vec3f(0.0));
    }
    textureStore(outTex, gid.xy, vec4f(result, color.a));
}
