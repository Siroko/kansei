// Voxel GI on screen through a voxel clipmap with probes (VoxelGIEffect::set_clipmap_probes),
// trace: per traced pixel, the irradiance its surface receives from the clipmap's irradiance
// probes (clipmap_probes.wgsl's kansei_clipmap_light), in place of cones traced per pixel; the
// temporal filter and the composite follow as for the cones. Where no probe holds a pixel (past
// the probes, or before they are traced), the sky's irradiance there, which the composite adds
// back for the material's own sky light it takes out. Output: rgb the irradiance, a the share of
// the hemisphere that sees the sky.

@group(0) @binding(0) var<uniform> gp : VoxelGiParams;
@group(0) @binding(1) var depthTex  : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(6) var<uniform> sky : SkyLighting;
@group(0) @binding(7) var outTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(60) var<uniform> kansei_clip_probe_grid : ClipProbeGrid;
@group(0) @binding(61) var<storage, read> kansei_clip_probes : array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.traceSize)) { return; }
    let px = gpPixel((vec2f(gid.xy) + 0.5) / gp.traceSize);
    let depth = gpDepth(px);
    if (depth >= 1.0) {
        textureStore(outTex, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let view = gpViewPos((vec2f(px) + 0.5) / gp.fullSize, depth);
    let world = (gp.invView * vec4f(view, 1.0)).xyz;
    let n = surfaceNormal(px, view);
    let light = kansei_clipmap_light(world, n);
    var e = vec4f(light.rgb, light.a);
    if (light.a < 0.0) {
        // no probe here: the sky the material's own light assumes
        e = vec4f(skyIrradiance(sky, n) * gp.skyScale, 1.0);
    }
    textureStore(outTex, gid.xy, vec4f(min(e.rgb, vec3f(60000.0)), e.a));
}
