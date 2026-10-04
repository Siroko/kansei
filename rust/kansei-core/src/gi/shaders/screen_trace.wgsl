// Voxel GI on screen, trace: per traced pixel, the irradiance its surface receives through the
// voxel volume and its anisotropic mips (voxel_irradiance.wgsl's six cones over the hemisphere around its normal, turned
// per pixel and frame for the temporal filter to integrate), from a point a little out along the
// normal so the surface's own voxels don't hide it. Output: rgb the irradiance (E: a diffuse
// surface adds albedo / pi times it), a the share of the hemisphere that sees past the volume.

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var<uniform> vol : VoxelVolume;
@group(0) @binding(4) var radiance : texture_3d<f32>;
@group(0) @binding(5) var linearClamp : sampler;
@group(0) @binding(6) var<uniform> sky : SkyLighting;
@group(0) @binding(7) var outTex : texture_storage_2d<rgba16float, write>;

fn ign(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

// A world normal: the GBuffer's, or one rebuilt from the depth around the pixel.
fn surfaceNormal(px: vec2i, p: vec3f) -> vec3f {
    let n = gpWorldNormal(px);
    if (n.w > 0.0) { return n.xyz; }
    let uv = (vec2f(px) + 0.5) / gp.fullSize;
    let du = vec2f(1.0 / gp.fullSize.x, 0.0);
    let dv = vec2f(0.0, 1.0 / gp.fullSize.y);
    let r = gpViewPos(uv + du, gpDepth(px + vec2i(1, 0)));
    let l = gpViewPos(uv - du, gpDepth(px - vec2i(1, 0)));
    let d = gpViewPos(uv + dv, gpDepth(px + vec2i(0, 1)));
    let u = gpViewPos(uv - dv, gpDepth(px - vec2i(0, 1)));
    // the side with the smaller depth step, so edges don't bend the normal
    let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
    let dy = select(p - d, u - p, abs(u.z - p.z) < abs(p.z - d.z));
    var nv = normalize(cross(dx, dy));
    if (dot(nv, -p) < 0.0) { nv = -nv; }
    return normalize((gp.invView * vec4f(nv, 0.0)).xyz);
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
    let origin = world + n * (gp.startVoxels * vol.voxelSize);
    let e = voxelIrradiance(vol, radiance, linearClamp, sky, gp.skyScale, origin, n, angle, vol.voxelSize, gp.maxDistance, gp.maxSteps, 1.0);
    textureStore(outTex, gid.xy, vec4f(min(e.rgb, vec3f(60000.0)), e.a));
}
