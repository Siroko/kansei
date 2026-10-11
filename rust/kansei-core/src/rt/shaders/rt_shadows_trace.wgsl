// rt::RtShadowsEffect, trace: a shadow ray a light at each texel of the trace resolution (and a
// screen-space contact ray), toward this frame's point of the light's emitter. Writes 1 lit, 0
// shadowed (1 where the light doesn't reach or there is no lit surface), and the guide.

@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var normalTex : texture_2d<f32>;
@group(0) @binding(4) var emissiveTex : texture_2d<f32>;
@group(0) @binding(5) var visOutA : texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var visOutB : texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var guideOut : texture_storage_2d<rgba32float, write>;

// Off a surface by a little of a cell (more far from the eye: the depth's precision).
fn shBias(p: vec3f) -> f32 {
    return 0.06 * kansei_rt_grid.cell + 2e-4 * distance(sp.invView[3].xyz, p);
}

// Whether a short ray from `o` along `d` passes in front of what the depth buffer holds: 0 where
// it goes behind a surface within `contactThickness` (a contact the grid is too coarse for).
fn shContact(o: vec3f, d: vec3f, px: vec2u) -> f32 {
    let steps = 10u;
    let jitter = shHash(px.x, px.y, sp.frame * 3u + 1u);
    for (var i = 0u; i < steps; i++) {
        let t = sp.contactLength * (f32(i) + jitter) / f32(steps);
        let q = o + d * t;
        let clip = sp.viewProj * vec4f(q, 1.0);
        if (clip.w <= 0.0) { return 1.0; }
        let ndc = clip.xyz / clip.w;
        let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return 1.0; }
        let texel = vec2u(uv * sp.fullSize);
        let sceneDepth = textureLoad(depthTex, texel, 0);
        if (sceneDepth >= 1.0) { continue; }
        // compare view depths: the ray's (its clip w) and the surface's there
        let rayZ = clip.w;
        let sv = sp.invProj * vec4f(ndc.xy, sceneDepth, 1.0);
        let sceneZ = -sv.z / sv.w;
        let behind = rayZ - sceneZ;
        if (behind > 0.01 && behind < sp.contactThickness) {
            return 0.0;
        }
    }
    return 1.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let t = id.xy;
    if (any(vec2f(t) >= sp.traceSize)) { return; }
    let px = min(t * sp.downscale, vec2u(sp.fullSize) - 1u);
    let depth = textureLoad(depthTex, px, 0);
    let ea = textureLoad(emissiveTex, px, 0).a;
    var vis = array<f32, 8>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    if (depth >= 1.0 || !shLitHere(ea)) {
        textureStore(visOutA, t, vec4f(1.0));
        textureStore(visOutB, t, vec4f(1.0));
        textureStore(guideOut, t, vec4f(0.0));
        return;
    }
    let uv = (vec2f(px) + 0.5) / sp.fullSize;
    let p = shWorldPos(uv, depth);
    let n = normalize(textureLoad(normalTex, px, 0).xyz * 2.0 - 1.0);
    let o = p + n * shBias(p);
    let count = min(sp.numLights, RT_SHADOW_MAX_LIGHTS);
    for (var k = 0u; k < count; k++) {
        let light = shadowLights[k];
        let inc = shIncoming(light, p);
        if (dot(n, inc.toLight) <= 0.0 || max(inc.illuminance.r, max(inc.illuminance.g, inc.illuminance.b)) <= 0.0) {
            continue;
        }
        let xi = shSample2(px, sp.frame, k);
        var d = inc.toLight;
        var tMax = sp.maxDistance;
        if (light.kind == 0u) {
            // a direction within the sun's disk
            let b = shBasis(d);
            let r = tan(light.radius) * sqrt(xi.x);
            let a = 6.2831853 * xi.y;
            d = normalize(d + (b[0] * cos(a) + b[1] * sin(a)) * r);
        } else {
            var aim = light.position;
            if (dot(light.axisU, light.axisU) > 0.0) {
                // a point of the rectangle
                aim += light.axisU * (xi.x * 2.0 - 1.0) + light.axisV * (xi.y * 2.0 - 1.0);
            } else if (light.radius > 0.0) {
                // a point of the disk facing the surface
                let b = shBasis(-inc.toLight);
                let r = light.radius * sqrt(xi.x);
                let a = 6.2831853 * xi.y;
                aim += (b[0] * cos(a) + b[1] * sin(a)) * r;
            }
            let toT = aim - o;
            tMax = length(toT) - 0.02;
            d = toT / max(length(toT), 1e-4);
            if (dot(d, n) <= 0.0) { vis[k] = 0.0; continue; }
        }
        var v = 1.0;
        if (kansei_rt_contains(o)) {
            let h = kansei_rt_trace(o, d, 0.0, tMax, KANSEI_RT_ANY_HIT | KANSEI_RT_SOLID);
            if (h.found) { v = 0.0; }
        }
        if (v > 0.0 && (sp.flags & RT_SHADOW_CONTACT) != 0u && sp.contactLength > 0.0) {
            v = shContact(o, d, px);
        }
        vis[k] = v;
    }
    textureStore(visOutA, t, vec4f(vis[0], vis[1], vis[2], vis[3]));
    textureStore(visOutB, t, vec4f(vis[4], vis[5], vis[6], vis[7]));
    let viewZ = -shViewPos(uv, depth).z;
    textureStore(guideOut, t, vec4f(shEncodeNormal(n), viewZ, 0.0));
}
