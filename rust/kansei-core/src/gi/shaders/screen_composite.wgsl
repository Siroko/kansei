// Voxel GI on screen, composite: the accumulated irradiance upsampled by depth (the traced
// texels whose surface matches this pixel's), then added to the scene as albedo / pi times it.
//
// With a near field (screen-space GI traced first, ssgi_trace.wgsl), the screen supplies the
// light of what it saw hide each direction and the voxels the rest: E = E_screen + open * E_voxels,
// `open` the share of the hemisphere the screen found no occluder in (Lumen's split: screen
// traces first, the scene representation for what they miss).
//
// The voxels' irradiance holds the sky they see past the volume, so with the sky's lighting
// bound the material's own sky ambient (albedo / pi times the sky's irradiance around its normal)
// is taken out, `ambient` times: it is what the GI replaces. The debug view shows the GI alone.

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var colorTex  : texture_2d<f32>;
@group(0) @binding(2) var depthTex  : texture_depth_2d;
@group(0) @binding(3) var giTex     : texture_2d<f32>;
@group(0) @binding(4) var nearTex   : texture_2d<f32>;
@group(0) @binding(5) var albedoTex : texture_2d<f32>;
@group(0) @binding(6) var normalTex : texture_2d<f32>;
@group(0) @binding(7) var<uniform> sky : SkyLighting;
@group(0) @binding(8) var outTex    : texture_storage_2d<rgba16float, write>;

// `tex` (traced at `size`) at `uv`, bilinear among the texels whose surface lies near this
// pixel's view depth `z`.
fn upsample(tex: texture_2d<f32>, size: vec2f, uv: vec2f, z: f32) -> vec4f {
    let pos = uv * size - 0.5;
    let base = floor(pos);
    let f = pos - base;
    var sum = vec4f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let t = clamp(vec2i(base + o), vec2i(0), vec2i(size) - 1);
        let tuv = (vec2f(t) + 0.5) / size;
        let tz = -gpViewPos(tuv, gpDepth(gpPixel(tuv))).z;
        let bilinear = select(1.0 - f.x, f.x, o.x > 0.5) * select(1.0 - f.y, f.y, o.y > 0.5);
        let w = bilinear * exp(-abs(tz - z) / (0.02 * z + 0.05)) + 1e-5 * bilinear;
        sum += textureLoad(tex, t, 0) * w;
        weight += w;
    }
    return sum / max(weight, 1e-6);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.fullSize)) { return; }
    let color = textureLoad(colorTex, gid.xy, 0);
    let px = vec2i(gid.xy);
    let depth = gpDepth(px);
    let albedo = textureLoad(albedoTex, px, 0).rgb;
    if (depth >= 1.0 || all(albedo <= vec3f(0.0))) {
        textureStore(outTex, gid.xy, select(color, vec4f(0.0, 0.0, 0.0, color.a), gp.debug != 0u));
        return;
    }
    let uv = (vec2f(gid.xy) + 0.5) / gp.fullSize;
    let z = -gpViewPos(uv, depth).z;
    var e = upsample(giTex, gp.traceSize, uv, z).rgb;
    if (gp.nearField != 0u) {
        let near = upsample(nearTex, gp.nearSize, uv, z);
        e = near.rgb + near.a * e;
    }
    let bounce = albedo * e * (gp.intensity / 3.14159265);
    if (gp.debug != 0u) {
        textureStore(outTex, gid.xy, vec4f(bounce, color.a));
        return;
    }
    var result = color.rgb + bounce;
    let n = gpWorldNormal(px);
    if (gp.hasSky != 0u && gp.ambient > 0.0 && n.w > 0.0) {
        result = max(result - albedo * skyIrradiance(sky, n.xyz) / 3.14159265 * gp.ambient, vec3f(0.0));
    }
    textureStore(outTex, gid.xy, vec4f(result, color.a));
}
