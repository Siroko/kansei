// Spot lights in the fog: the renderer's spot-light buffer and shadow atlas. Each light scatters
// only inside its cone, falls off with the inverse square of distance, and is shadowed by its
// atlas layer, so headlights draw beams with the shadows of what stands in them. The lights and
// the atlas are compute_shadows.wgsl's.

// The i-th spot light at worldPos as a FogLight (volumetric_fog_inject.wgsl): its illuminance in
// its cone, times its `volumetricScale` and its atlas layer's visibility. `minDist`: a froxel is a
// volume, not a point; closer than about its size to the light, the inverse square is capped (its
// average over the froxel stays finite).
fn fogSpotLight(i: u32, worldPos: vec3f, minDist: f32) -> FogLight {
    let light = spotLights.lights[i];
    if (light.volumetricScale <= 0.0) { return FogLight(vec3f(0.0, 1.0, 0.0), vec3f(0.0)); }
    var s = kansei_spot_sample(light, worldPos);
    let d = light.position - worldPos;
    let dist2 = max(dot(d, d), 1e-4);
    s.illuminance *= dist2 / max(dist2, minDist * minDist);
    if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { return FogLight(s.toLight, vec3f(0.0)); }

    var visibility = 1.0;
    if (light.shadowLayer >= 0) {
        let coord = kansei_spot_shadow_coord(light, worldPos);
        if (coord.w > 0.0) {
            visibility = textureSampleCompareLevel(spotShadowAtlas, spotShadowSampler, coord.xy, light.shadowLayer, coord.z);
        }
    }
    return FogLight(s.toLight, s.illuminance * (light.volumetricScale * visibility));
}

fn spotInScatter(worldPos: vec3f, viewDir: vec3f, minDist: f32) -> vec3f {
    var scatter = vec3f(0.0);
    for (var i = 0u; i < spotLights.count; i++) {
        let l = fogSpotLight(i, worldPos, minDist);
        scatter += l.radiance * henyeyGreenstein(dot(viewDir, l.toLight), params.anisotropy);
    }
    return scatter;
}
