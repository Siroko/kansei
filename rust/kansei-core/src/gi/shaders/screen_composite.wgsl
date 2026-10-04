// Voxel GI on screen, composite: the accumulated irradiance upsampled and filtered by depth (the
// traced texels whose surface matches this pixel's), then added to the scene as albedo / pi
// times it.
//
// With a near field (screen-space GI traced first, ssgi_trace.wgsl), the screen supplies the
// light of what it saw hide each direction and the voxels the rest: E = E_screen + open * E_voxels,
// `open` the share of the hemisphere the screen found no occluder in (Lumen's split: screen
// traces first, the scene representation for what they miss). The screen's occluders carry only
// their direct light, though (the frame's lit colour before GI), so where they are lit by
// bounces alone it would leave those directions dark: the voxels' light from the same share of
// the hemisphere, (1 - open) * E_voxels, is the least the screen's part gives.
//
// The voxels' irradiance holds the sky they see past the volume, so with the sky's lighting
// bound the material's own sky ambient (albedo / pi times the sky's irradiance around its normal)
// is taken out, `ambient` times: it is what the GI replaces. The debug views show the GI alone, or
// the volume's mip 0 as the camera sees it (the voxelized scene and its light).

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var colorTex  : texture_2d<f32>;
@group(0) @binding(2) var depthTex  : texture_depth_2d;
@group(0) @binding(3) var giTex     : texture_2d<f32>;
@group(0) @binding(4) var nearTex   : texture_2d<f32>;
@group(0) @binding(5) var albedoTex : texture_2d<f32>;
@group(0) @binding(6) var normalTex : texture_2d<f32>;
@group(0) @binding(7) var<uniform> sky : SkyLighting;
@group(0) @binding(8) var outTex    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<uniform> vol : VoxelVolume;
@group(0) @binding(10) var radiance : texture_3d<f32>;
@group(0) @binding(11) var linearClamp : sampler;

// The voxels' light along the camera ray through `uv`: mip 0 marched front to back, half a voxel
// a step (from a start jittered per pixel, so the steps don't band), from where the ray enters
// the volume.
fn marchVoxels(uv: vec2f) -> vec3f {
    let eye = (gp.invView * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let far = (gp.invView * vec4f(gpViewPos(uv, 1.0), 1.0)).xyz;
    let dir = normalize(far - eye);
    let lo = vol.origin;
    let hi = vol.origin + vec3f(vol.dims) * vol.voxelSize;
    let t0 = (lo - eye) / dir;
    let t1 = (hi - eye) / dir;
    let enter = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
    let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    var color = vec3f(0.0);
    var transmittance = 1.0;
    let jitter = fract(52.9829189 * fract(dot(uv * gp.fullSize, vec2f(0.06711056, 0.00583715))));
    var t = enter + jitter * 0.5 * vol.voxelSize;
    for (var i = 0u; i < 1024u; i++) {
        if (t >= exit || transmittance < 0.01) { break; }
        let s = textureSampleLevel(radiance, linearClamp, voxelUvw(vol, eye + dir * t), 0.0);
        let a = 1.0 - sqrt(max(1.0 - s.a, 0.0));
        color += transmittance * s.rgb * select(0.5, a / s.a, s.a > 1e-4);
        transmittance *= 1.0 - a;
        t += 0.5 * vol.voxelSize;
    }
    return color * vol.radianceScale;
}

// `tex` (traced at `size`) at `uv`, filtered over the 4x4 traced texels around it with a tent
// two texels wide (a wider bilinear: the irradiance is smooth, the cones' per-pixel rotation is
// not), among the texels whose surface lies near this pixel's view depth `z`.
fn upsample(tex: texture_2d<f32>, size: vec2f, uv: vec2f, z: f32) -> vec4f {
    let pos = uv * size - 0.5;
    let base = floor(pos) - 1.0;
    var sum = vec4f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let o = vec2f(f32(i & 3u), f32(i >> 2u));
        let t = clamp(vec2i(base + o), vec2i(0), vec2i(size) - 1);
        let tuv = (vec2f(t) + 0.5) / size;
        let tz = -gpViewPos(tuv, gpDepth(gpPixel(tuv))).z;
        let d = abs(base + o - pos) * 0.5;
        let tent = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
        let w = tent * exp(-abs(tz - z) / (0.02 * z + 0.05)) + 1e-5 * tent;
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
    if (gp.debug == 2u) {
        textureStore(outTex, gid.xy, vec4f(marchVoxels((vec2f(gid.xy) + 0.5) / gp.fullSize), color.a));
        return;
    }
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
        e = max(near.rgb, (1.0 - near.a) * e) + near.a * e;
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
