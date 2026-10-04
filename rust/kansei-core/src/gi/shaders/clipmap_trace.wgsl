// Voxel GI on screen through a voxel clipmap (gi::VoxelGIEffect::with_clipmap), trace:
// screen_trace.wgsl's, with clipmap.wgsl's cones. Per traced pixel, the irradiance its surface
// receives through the clipmap (six cones over the hemisphere around its normal, turned per pixel
// and frame), from a point a little out along the normal, by the voxels of the finest level that
// holds the surface: a distant surface reads only coarse levels. Output: rgb the irradiance, a the
// share of the hemisphere that sees past the clipmap.
//
// `show_voxels`: the debug view, the clipmap's light as the camera sees it, each step of the ray
// reading the finest level that holds it, at full resolution in place of the lit image.

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(6) var<uniform> sky : SkyLighting;
@group(0) @binding(7) var outTex : texture_storage_2d<rgba16float, write>;

fn ign(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let px = gpPixel((vec2f(gid.xy) + 0.5) / gp.traceSize);
    let depth = gpDepth(px);
    if (depth >= 1.0) {
        textureStore(outTex, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let p = gpViewPos((vec2f(px) + 0.5) / gp.fullSize, depth);
    let world = (gp.invView * vec4f(p, 1.0)).xyz;
    let n = surfaceNormal(px, p);
    let angle = fract(ign(vec2f(gid.xy)) + f32(gp.frame % 64u) * 0.618034) * 6.2831853;
    // the voxels of the finest level holding the surface (the coarsest past them all)
    let size = clipVoxelSize(min(clipLevelAt(world, 0u, 1.0), clipmap.levelCount - 1u));
    let origin = world + n * (gp.startVoxels * size);
    let e = clipIrradiance(sky, gp.skyScale, origin, n, angle, size, size, gp.maxDistance, gp.maxSteps);
    textureStore(outTex, gid.xy, vec4f(min(e.rgb, vec3f(60000.0)), e.a));
}

// The clipmap's light along the camera ray through `uv`: front to back, half a voxel of the
// finest level holding each step a step, from a start jittered per pixel.
fn marchClipmap(uv: vec2f) -> vec3f {
    let eye = (gp.invView * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let far = (gp.invView * vec4f(gpViewPos(uv, 1.0), 1.0)).xyz;
    let dir = normalize(far - eye);
    var color = vec3f(0.0);
    var transmittance = 1.0;
    let jitter = ign(uv * gp.fullSize);
    var t = jitter * 0.5 * clipmap.voxelSize;
    for (var i = 0u; i < 2048u; i++) {
        if (transmittance < 0.01) { break; }
        let p = eye + dir * t;
        let k = clipLevelAt(p, 0u, 0.5);
        if (k >= clipmap.levelCount) { break; }
        let s = clipSample(k, p);
        let a = 1.0 - sqrt(max(1.0 - s.a, 0.0));
        color += transmittance * s.rgb * select(0.5, a / s.a, s.a > 1e-4);
        transmittance *= 1.0 - a;
        t += 0.5 * clipVoxelSize(k);
    }
    return color * clipmap.radianceScale;
}

@compute @workgroup_size(8, 8)
fn show_voxels(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.fullSize)) { return; }
    textureStore(outTex, gid.xy, vec4f(marchClipmap((vec2f(gid.xy) + 0.5) / gp.fullSize), 1.0));
}
