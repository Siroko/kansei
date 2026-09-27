// Spot lights in the fog: the renderer's spot-light buffer and shadow atlas. Each light scatters
// only inside its cone, falls off with the inverse square of distance, and is shadowed by its
// atlas layer, so headlights draw beams with the shadows of what stands in them.

@group(0) @binding(7) var<storage, read> spotLights : KanseiSpotLights;
@group(0) @binding(8) var spotShadowAtlas : texture_depth_2d_array;
@group(0) @binding(9) var spotShadowSampler : sampler_comparison;

// `minDist`: a froxel is a volume, not a point; closer than about its size to the light, the
// inverse square is capped (its average over the froxel stays finite).
fn spotInScatter(worldPos: vec3f, viewDir: vec3f, minDist: f32) -> vec3f {
    var scatter = vec3f(0.0);
    for (var i = 0u; i < spotLights.count; i++) {
        let light = spotLights.lights[i];
        if (light.volumetricScale <= 0.0) { continue; }
        var s = kansei_spot_sample(light, worldPos);
        let d = light.position - worldPos;
        let dist2 = max(dot(d, d), 1e-4);
        s.illuminance *= dist2 / max(dist2, minDist * minDist);
        if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { continue; }

        var visibility = 1.0;
        if (light.shadowLayer >= 0) {
            let coord = kansei_spot_shadow_coord(light, worldPos);
            if (coord.w > 0.0) {
                visibility = textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, coord.xy, light.shadowLayer, coord.z);
            }
        }
        let phase = henyeyGreenstein(dot(viewDir, s.toLight), params.anisotropy);
        scatter += s.illuminance * (light.volumetricScale * visibility * phase);
    }
    return scatter;
}
