// Clustered light culling (Olsson et al. 2012): the view frustum is cut into 16 x 9 screen
// tiles x 24 exponential depth slices, and each cluster lists the spot lights whose range (and,
// for spots, cone) touches it. Materials then shade with their cluster's lights only
// (kansei_spot_lights_radiance), so many lights stay cheap. One thread per cluster.
//
// Per cluster, KANSEI_CLUSTER_SLOTS u32: the count, then the light indices.

@group(0) @binding(0) var<uniform> kansei_clusters : KanseiClusterParams;
@group(0) @binding(1) var<storage, read> lights : KanseiSpotLights;
@group(0) @binding(2) var<storage, read_write> clusterLights : array<u32>;

// A point on the view-space ray through the screen uv (y down), at view depth `depth` (> 0).
fn clusterRayPoint(uv: vec2f, depth: f32) -> vec3f {
    let p = kansei_clusters.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let d = p.xyz / p.w;
    return d * (depth / max(-d.z, 1e-6));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let g = kansei_clusters.grid;
    let index = gid.x;
    if (index >= g.x * g.y * g.z) { return; }
    let cx = index % g.x;
    let cy = (index / g.x) % g.y;
    let cz = index / (g.x * g.y);

    // the cluster's view-space bounding box: its tile's corner rays between its slice depths
    let ratio = kansei_clusters.far / kansei_clusters.near;
    let z0 = kansei_clusters.near * pow(ratio, f32(cz) / f32(g.z));
    let z1 = kansei_clusters.near * pow(ratio, f32(cz + 1u) / f32(g.z));
    let uv0 = vec2f(f32(cx), f32(cy)) / vec2f(g.xy);
    let uv1 = vec2f(f32(cx + 1u), f32(cy + 1u)) / vec2f(g.xy);
    var lo = vec3f(1e30);
    var hi = vec3f(-1e30);
    for (var k = 0u; k < 8u; k++) {
        let uv = vec2f(select(uv0.x, uv1.x, (k & 1u) != 0u), select(uv0.y, uv1.y, (k & 2u) != 0u));
        let p = clusterRayPoint(uv, select(z0, z1, (k & 4u) != 0u));
        lo = min(lo, p);
        hi = max(hi, p);
    }
    let center = 0.5 * (lo + hi);
    let radius = 0.5 * length(hi - lo);

    let base = index * KANSEI_CLUSTER_SLOTS;
    var count = 0u;
    for (var i = 0u; i < lights.count; i++) {
        let light = lights.lights[i];
        let pos = (kansei_clusters.view * vec4f(light.position, 1.0)).xyz;
        // the light's range sphere against the box
        let q = clamp(pos, lo, hi);
        if (dot(q - pos, q - pos) > light.range * light.range) { continue; }
        // spots: the cone against the cluster's bounding sphere (Wronski, "Cull that cone")
        if (light.cosOuter > -1.0) {
            let axis = normalize((kansei_clusters.view * vec4f(light.direction, 0.0)).xyz);
            let v = center - pos;
            let along = dot(v, axis);
            let across = sqrt(max(dot(v, v) - along * along, 0.0));
            let sinOuter = sqrt(max(1.0 - light.cosOuter * light.cosOuter, 0.0));
            let closest = light.cosOuter * across - along * sinOuter;
            if (closest > radius || along > radius + light.range || along < -radius) { continue; }
        }
        if (count == KANSEI_CLUSTER_SLOTS - 1u) { break; }
        count++;
        clusterLights[base + count] = i;
    }
    clusterLights[base] = count;
}
