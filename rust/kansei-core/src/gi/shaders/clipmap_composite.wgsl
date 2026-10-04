// Voxel GI on screen through a voxel clipmap with probes (VoxelGIEffect::set_clipmap_probes),
// composite: screen_composite.wgsl's main_probes, with each pixel's far field from the clipmap's
// irradiance probes (clipmap_probes.wgsl's kansei_clipmap_light) in place of cones traced per
// pixel. Where no probe holds a pixel (past the probes, or before they are traced), it keeps the
// material's own sky light. Concatenated after screen_composite.wgsl (its bindings and helpers).

@group(0) @binding(60) var<uniform> kansei_clip_probe_grid : ClipProbeGrid;
@group(0) @binding(61) var<storage, read> kansei_clip_probes : array<vec4f>;

@compute @workgroup_size(8, 8)
fn main_clipmap_probes(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= gp.fullSize)) { return; }
    let color = textureLoad(colorTex, gid.xy, 0);
    let px = vec2i(gid.xy);
    let uv = (vec2f(gid.xy) + 0.5) / gp.fullSize;
    let depth = gpDepth(px);
    let albedo = textureLoad(albedoTex, px, 0).rgb;
    if (depth >= 1.0 || all(albedo <= vec3f(0.0))) {
        textureStore(outTex, gid.xy, select(color, vec4f(0.0, 0.0, 0.0, color.a), gp.debug != 0u));
        return;
    }
    let view = gpViewPos(uv, depth);
    let ns = surfaceNormal(px, view);
    let world = (gp.invView * vec4f(view, 1.0)).xyz;
    let light = kansei_clipmap_light(world, ns);
    if (light.a < 0.0) {
        // no probe here: the material's own sky light stays
        textureStore(outTex, gid.xy, select(color, vec4f(0.0, 0.0, 0.0, color.a), gp.debug != 0u));
        return;
    }
    var e = light.rgb;
    if (gp.nearField != 0u) {
        let near = upsample(nearTex, gp.nearSize, uv, -view.z);
        e = max(near.rgb, (1.0 - near.a) * e) + near.a * e;
    }
    let n = gpWorldNormal(px);
    let bounce = albedo * e * (gp.intensity / 3.14159265);
    if (gp.debug != 0u) {
        textureStore(outTex, gid.xy, vec4f(bounce, color.a));
        return;
    }
    var result = color.rgb + bounce;
    if (gp.hasSky != 0u && gp.ambient > 0.0 && n.w > 0.0) {
        result = max(result - albedo * skyIrradiance(sky, n.xyz) / 3.14159265 * gp.ambient, vec3f(0.0));
    }
    textureStore(outTex, gid.xy, vec4f(result, color.a));
}
