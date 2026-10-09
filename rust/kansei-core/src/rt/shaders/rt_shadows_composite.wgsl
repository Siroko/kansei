// rt::RtShadowsEffect, composite: at each pixel the effect lights, every light's BRDF times its
// illuminance times its visibility (upsampled from the trace resolution with depth and normal
// weights), added to the lit colour. The diffuse is Lambert on the GBuffer's albedo; the specular
// (GGX) only on surfaces no ray-traced reflection covers (theirs is traced), toward the point of
// the emitter closest to the reflected ray.

@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var normalTex : texture_2d<f32>;
@group(0) @binding(4) var albedoTex : texture_2d<f32>;
@group(0) @binding(5) var emissiveTex : texture_2d<f32>;
@group(0) @binding(6) var inputTex : texture_2d<f32>;
@group(0) @binding(7) var visA : texture_2d<f32>;
@group(0) @binding(8) var visB : texture_2d<f32>;
@group(0) @binding(9) var guideTex : texture_2d<f32>;
@group(0) @binding(10) var outputTex : texture_storage_2d<rgba16float, write>;

// The visibility at a full-resolution pixel: the four trace texels round it, each by its bilinear
// weight and how alike its surface is (depth and normal).
fn shUpsample(px: vec2u, z: f32, n: vec3f) -> array<f32, 8> {
    let tp = (vec2f(px) + 0.5) / f32(sp.downscale) - 0.5;
    let t0 = vec2i(floor(tp));
    let f = tp - floor(tp);
    let last = vec2i(sp.traceSize) - 1;
    var a = vec4f(0.0);
    var b = vec4f(0.0);
    var wsum = 0.0;
    var fa = vec4f(1.0);
    var fb = vec4f(1.0);
    var best = -1.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1u);
        let q = clamp(t0 + o, vec2i(0), last);
        let g = textureLoad(guideTex, q, 0);
        if (g.z <= 0.0) { continue; }
        let similar = exp(-abs(g.z - z) / (0.02 * z + 1e-3)) * pow(max(dot(shDecodeNormal(g.xy), n), 0.0), 16.0);
        let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * similar;
        let qa = textureLoad(visA, q, 0);
        let qb = textureLoad(visB, q, 0);
        a += w * qa;
        b += w * qb;
        wsum += w;
        if (similar > best) { best = similar; fa = qa; fb = qb; }
    }
    if (wsum > 1e-4) {
        a /= wsum;
        b /= wsum;
    } else {
        a = fa;
        b = fb;
    }
    return array<f32, 8>(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
}

fn shGgx(n: vec3f, v: vec3f, l: vec3f, roughness: f32, f0: f32) -> f32 {
    let h = normalize(v + l);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let a = max(roughness * roughness, 2e-3);
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (RT_SHADOW_PI * dd * dd);
    let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
    let fr = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    return d * vis * fr * nl;
}

// The emitter's point closest to the reflected ray (Karis 2013's representative point), for the
// specular of a disk or a rectangle.
fn shRepresentative(light: RtShadowLight, p: vec3f, r: vec3f) -> vec3f {
    let c = light.position - p;
    if (dot(light.axisU, light.axisU) > 0.0) {
        let nrm = normalize(cross(light.axisU, light.axisV));
        let denom = dot(r, nrm);
        var hit = c;
        if (abs(denom) > 1e-4) {
            hit = r * (dot(c, nrm) / denom);
        }
        let local = hit - c;
        let u = clamp(dot(local, light.axisU) / dot(light.axisU, light.axisU), -1.0, 1.0);
        let v = clamp(dot(local, light.axisV) / dot(light.axisV, light.axisV), -1.0, 1.0);
        return c + light.axisU * u + light.axisV * v;
    }
    let centre = dot(c, r) * r - c;
    return c + centre * clamp(light.radius / max(length(centre), 1e-4), 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let px = id.xy;
    if (any(vec2f(px) >= sp.fullSize)) { return; }
    let color = textureLoad(inputTex, px, 0);
    let depth = textureLoad(depthTex, px, 0);
    let ea = textureLoad(emissiveTex, px, 0).a;
    if (sp.view == 3u) {
        // the mask: red where the effect lights, the emissive alpha in green, depth in blue
        textureStore(outputTex, px, vec4f(select(0.0, 1.0, shLitHere(ea)), ea, select(0.0, 1.0, depth < 1.0), 1.0));
        return;
    }
    if (depth >= 1.0 || !shLitHere(ea)) {
        textureStore(outputTex, px, select(color, vec4f(0.0, 0.0, 0.0, color.a), sp.view == 2u));
        return;
    }
    let uv = (vec2f(px) + 0.5) / sp.fullSize;
    let p = shWorldPos(uv, depth);
    let rawN = textureLoad(normalTex, px, 0);
    let n = normalize(rawN.xyz * 2.0 - 1.0);
    let albedo = textureLoad(albedoTex, px, 0);
    let eye = sp.invView[3].xyz;
    let v = normalize(eye - p);
    let roughness = shRoughness(ea);
    // reflective surfaces (their F0 in the normal's alpha) take their specular from the traced
    // reflections
    let reflective = rawN.w < 0.999;
    let z = -shViewPos(uv, depth).z;
    let vis = shUpsample(px, z, n);
    let r = reflect(-v, n);
    var direct = vec3f(0.0);
    let count = min(sp.numLights, RT_SHADOW_MAX_LIGHTS);
    for (var k = 0u; k < count; k++) {
        let light = shadowLights[k];
        let inc = shIncoming(light, p);
        let nl = dot(n, inc.toLight);
        if (nl <= 0.0 || vis[k] <= 0.0) { continue; }
        var brdf = albedo.rgb / RT_SHADOW_PI * nl;
        if (!reflective) {
            var l = inc.toLight;
            if (light.kind == 1u) {
                l = normalize(shRepresentative(light, p, r));
            }
            brdf += vec3f(shGgx(n, v, l, roughness, 0.04));
        }
        direct += brdf * inc.illuminance * vis[k];
    }
    direct *= sp.intensity;
    var out = vec4f(color.rgb + direct, color.a);
    if (sp.view == 1u) {
        let k = min(sp.debugLight, RT_SHADOW_MAX_LIGHTS - 1u);
        out = vec4f(vec3f(vis[k]), color.a);
    } else if (sp.view == 2u) {
        out = vec4f(direct, color.a);
    }
    textureStore(outputTex, px, out);
}
