// Kansei spot lights for materials: the renderer's spot lights and their shadow atlas, bound in
// the shared shadow group (group 3, bindings 5-9), clustered so each fragment only visits the
// lights that reach it, with contact-hardening shadows and a GGX / Lambert BRDF. Prepend
// `lights::SPOT_LIGHTS_WGSL` to a material shader (it includes spot_light_types.wgsl), then
// either call
//
//     kansei_spot_lights_radiance(worldPos, N, V, baseColor, roughness, metallic, fragCoord.xy)
//
// for the light all spots reflect toward the viewer (cd/m², in the same units as the lights),
// or loop over the fragment's lights yourself: kansei_light_cluster, kansei_cluster_light_count
// and kansei_cluster_light give the indices into kansei_spot_lights.lights, and
// kansei_spot_sample/kansei_spot_shadow evaluate one.

@group(3) @binding(5) var kansei_spot_shadow_atlas   : texture_depth_2d_array;
@group(3) @binding(6) var<storage, read> kansei_spot_lights : KanseiSpotLights;
@group(3) @binding(7) var kansei_spot_shadow_sampler : sampler_comparison;
@group(3) @binding(8) var<uniform> kansei_clusters : KanseiClusterParams;
@group(3) @binding(9) var<storage, read> kansei_cluster_lights : array<u32>;

const KANSEI_NO_CLUSTER : u32 = 0xffffffffu;

// The light cluster a fragment is in (its screen tile and view depth), or KANSEI_NO_CLUSTER in
// views the clusters weren't built for.
fn kansei_light_cluster(worldPos: vec3f, pixel: vec2f) -> u32 {
    if (kansei_clusters.enabled == 0u) { return KANSEI_NO_CLUSTER; }
    let g = kansei_clusters.grid;
    let depth = -(kansei_clusters.view * vec4f(worldPos, 1.0)).z;
    let slice = log(max(depth, kansei_clusters.near) / kansei_clusters.near) / log(kansei_clusters.far / kansei_clusters.near);
    let z = min(u32(max(slice, 0.0) * f32(g.z)), g.z - 1u);
    let tile = min(vec2u(pixel / kansei_clusters.screen * vec2f(g.xy)), g.xy - 1u);
    return (z * g.y + tile.y) * g.x + tile.x;
}

// How many lights shade a fragment in `cluster`, and the `k`-th of them.
fn kansei_cluster_light_count(cluster: u32) -> u32 {
    if (cluster == KANSEI_NO_CLUSTER) { return kansei_spot_lights.count; }
    return kansei_cluster_lights[cluster * KANSEI_CLUSTER_SLOTS];
}

fn kansei_cluster_light(cluster: u32, k: u32) -> u32 {
    if (cluster == KANSEI_NO_CLUSTER) { return k; }
    return kansei_cluster_lights[cluster * KANSEI_CLUSTER_SLOTS + 1u + k];
}

const KANSEI_PI : f32 = 3.14159265;
const KANSEI_GOLDEN_ANGLE : f32 = 2.39996323;

// Interleaved gradient noise (Jimenez 2014), rotates the shadow kernels per pixel.
fn kansei_ign(pixel: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
}

// Point i of an n-point Vogel disk (unit radius), rotated by phi.
fn kansei_vogel(i: u32, n: u32, phi: f32) -> vec2f {
    let r = sqrt((f32(i) + 0.5) / f32(n));
    let theta = f32(i) * KANSEI_GOLDEN_ANGLE + phi;
    return r * vec2f(cos(theta), sin(theta));
}

// Visibility of a spot light (1 lit, 0 shadowed), PCSS: a blocker search sized by the emitter,
// then a 16-tap PCF whose radius is the penumbra width at the receiver, so shadows are sharp
// where casters touch the receiver and soften with distance.
fn kansei_spot_shadow(light: KanseiSpotLight, worldPos: vec3f, worldNormal: vec3f, pixel: vec2f) -> f32 {
    if (light.shadowLayer < 0) { return 1.0; }

    // receiver offset along the normal by a few texels of the map at this depth, more at grazing
    // angles, against acne
    let toLight = normalize(light.position - worldPos);
    let depthAlong = max(dot(worldPos - light.position, light.direction), light.shadowNear);
    let texelWorld = 2.0 * depthAlong * light.tanHalfFov * light.texelSize;
    let cosTheta = saturate(dot(worldNormal, toLight));
    let slope = sqrt(1.0 - cosTheta * cosTheta);
    let biased = worldPos + worldNormal * (light.normalBias * texelWorld * max(slope, 0.2));

    let coord = kansei_spot_shadow_coord(light, biased);
    if (coord.w == 0.0) { return 1.0; }
    let layer = light.shadowLayer;
    let zReceiver = kansei_spot_linear_depth(light, coord.z);
    let uvPerMetre = 1.0 / (2.0 * zReceiver * light.tanHalfFov);
    let phi = kansei_ign(pixel) * 6.2831853;
    let dim = vec2f(textureDimensions(kansei_spot_shadow_atlas));

    var filterUV = light.texelSize * 1.5;
    if (light.sourceRadius > 0.0) {
        // blocker search over the region where occluders could hide part of the emitter
        let searchUV = clamp(light.sourceRadius * uvPerMetre * 2.0, light.texelSize * 2.0, light.texelSize * 16.0);
        var blockerSum = 0.0;
        var blockers = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let uv = coord.xy + kansei_vogel(i, 8u, phi) * searchUV;
            let texel = vec2i(clamp(uv * dim, vec2f(0.0), dim - 1.0));
            let zb = kansei_spot_linear_depth(light, textureLoad(kansei_spot_shadow_atlas, texel, layer, 0));
            if (zb < zReceiver * 0.995) {
                blockerSum += zb;
                blockers += 1.0;
            }
        }
        if (blockers == 0.0) { return 1.0; }
        let zBlocker = blockerSum / blockers;
        // penumbra radius at the receiver: emitter radius scaled by (receiver - blocker) / blocker
        filterUV = clamp(light.sourceRadius * (zReceiver - zBlocker) / zBlocker * uvPerMetre,
                         light.texelSize, light.texelSize * 24.0);
    }

    var lit = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let uv = coord.xy + kansei_vogel(i, 16u, phi) * filterUV;
        lit += textureSampleCompareLevel(kansei_spot_shadow_atlas, kansei_spot_shadow_sampler, uv, layer, coord.z);
    }
    return lit / 16.0;
}

// GGX specular (height-correlated Smith) + Lambert diffuse, times N·L.
fn kansei_brdf(N: vec3f, V: vec3f, L: vec3f, baseColor: vec3f, roughness: f32, metallic: f32) -> vec3f {
    let NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0) { return vec3f(0.0); }
    let H = normalize(V + L);
    let NdotV = max(dot(N, V), 1e-4);
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));
    let a = max(roughness * roughness, 2e-3);
    let a2 = a * a;
    let dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    let D = a2 / (KANSEI_PI * dd * dd);
    let vis = 0.5 / (NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2) + NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2));
    let F0 = mix(vec3f(0.04), baseColor, metallic);
    let F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    let diffuse = (1.0 - metallic) * (1.0 - F) * baseColor / KANSEI_PI;
    return (diffuse + D * vis * F) * NdotL;
}

// Light reflected toward the viewer from every spot light, with shadows.
fn kansei_spot_lights_radiance(worldPos: vec3f, N: vec3f, V: vec3f, baseColor: vec3f, roughness: f32,
                               metallic: f32, pixel: vec2f) -> vec3f {
    var radiance = vec3f(0.0);
    // only the lights of this fragment's cluster (every light in views without clusters)
    let cluster = kansei_light_cluster(worldPos, pixel);
    let count = kansei_cluster_light_count(cluster);
    for (var k = 0u; k < count; k++) {
        let light = kansei_spot_lights.lights[kansei_cluster_light(cluster, k)];
        let s = kansei_spot_sample(light, worldPos);
        if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { continue; }
        let brdf = kansei_brdf(N, V, s.toLight, baseColor, roughness, metallic);
        if (max(brdf.r, max(brdf.g, brdf.b)) <= 0.0) { continue; }
        radiance += brdf * s.illuminance * kansei_spot_shadow(light, worldPos, N, pixel);
    }
    return radiance;
}
