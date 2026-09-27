// Screen-space global illumination, trace: one bounce of the light on screen and how much of
// the sky each point sees, after Therrien, Levesque and Gilet 2023, "Screen Space Indirect
// Lighting with Visibility Bitmasks" (GTAO's slices, with each depth sample a slab `thickness`
// deep instead of an infinite wall).
//
// Around each point, a few slices through the view vector are searched on both sides. Every
// depth sample hides the part of the slice's hemisphere between its front and back; the
// hemisphere is cut into 32 sectors of equal cosine-weighted area, and a sector the first time
// it is hidden takes the sample's colour (its outgoing radiance, the lit scene). The sectors
// left open are the sky the point sees. Output: rgb the irradiance the bounce brings (as E, so
// a diffuse surface adds albedo / pi times it), a the share of the hemisphere left open.

@group(0) @binding(0) var<uniform> sp : SsgiParams;
@group(0) @binding(1) var colorTex  : texture_2d<f32>;
@group(0) @binding(2) var depthTex  : texture_depth_2d;
@group(0) @binding(3) var normalTex : texture_2d<f32>;
@group(0) @binding(4) var outTex    : texture_storage_2d<rgba16float, write>;

// The sectors between lo and hi (0..1 of the hemisphere). A sample grazing the edge of the
// hemisphere (the point's own plane) marks none.
fn sectors(lo: f32, hi: f32) -> u32 {
    let a = u32(clamp(floor(lo * 32.0 + 0.05), 0.0, 32.0));
    let b = u32(clamp(ceil(hi * 32.0 - 0.05), 0.0, 32.0));
    if (b <= a) { return 0u; }
    let width = b - a;
    return select((1u << width) - 1u, 0xffffffffu, width >= 32u) << a;
}

// Share of the cosine-weighted hemisphere below an angle a from the normal. Angles within a few
// degrees of the horizon count as the horizon itself: depth-rebuilt normals are that far off, and
// a surface's own neighbours would otherwise hide the edge sectors (which span 14 degrees each,
// the cosine weighting being flat there) while carrying almost none of the light.
const HORIZON_BIAS : f32 = 0.09;
fn hemisphereShare(a: f32) -> f32 {
    let c = clamp(a, -0.5 * SSGI_PI, 0.5 * SSGI_PI);
    let biased = sign(c) * min(abs(c) + HORIZON_BIAS, 0.5 * SSGI_PI);
    return (sin(biased) + 1.0) * 0.5;
}

fn ign(c: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(c, vec2f(0.06711056, 0.00583715))));
}

// A view-space normal: the GBuffer's, or one rebuilt from the depth around the pixel.
fn viewNormal(px: vec2i, p: vec3f) -> vec3f {
    let n = ssgiWorldNormal(px);
    if (n.w > 0.0) { return normalize((sp.view * vec4f(n.xyz, 0.0)).xyz); }
    let uv = (vec2f(px) + 0.5) / sp.fullSize;
    let du = vec2f(1.0 / sp.fullSize.x, 0.0);
    let dv = vec2f(0.0, 1.0 / sp.fullSize.y);
    let r = ssgiViewPos(uv + du, ssgiDepth(px + vec2i(1, 0)));
    let l = ssgiViewPos(uv - du, ssgiDepth(px - vec2i(1, 0)));
    let d = ssgiViewPos(uv + dv, ssgiDepth(px + vec2i(0, 1)));
    let u = ssgiViewPos(uv - dv, ssgiDepth(px - vec2i(0, 1)));
    // the side with the smaller depth step, so edges don't bend the normal
    let dx = select(p - l, r - p, abs(r.z - p.z) < abs(p.z - l.z));
    let dy = select(p - d, u - p, abs(u.z - p.z) < abs(p.z - d.z));
    var n2 = normalize(cross(dx, dy));
    if (dot(n2, -p) < 0.0) { n2 = -n2; }
    return n2;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= sp.traceSize)) { return; }
    let px = ssgiPixel((vec2f(gid.xy) + 0.5) / sp.traceSize);
    let depth = ssgiDepth(px);
    if (depth >= 1.0) {
        textureStore(outTex, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }
    // positions are unprojected at the centre of the depth pixel they were read from: off it, a
    // surface seen at a grazing angle would lift off its own plane and hide itself
    let uv = (vec2f(px) + 0.5) / sp.fullSize;
    let p = ssgiViewPos(uv, depth);
    let n = viewNormal(px, p);
    let v = normalize(-p);
    // the search radius on screen, and the step jitter that the history averages
    let radiusPx = min(sp.radius * sp.proj[1][1] * 0.5 * sp.fullSize.y / max(-p.z, 1e-3), sp.maxRadiusPx);
    let noise = fract(ign(vec2f(gid.xy)) + f32(sp.frame % 64u) * 0.618034);
    let stepNoise = fract(ign(vec2f(gid.yx) + 13.0) + f32(sp.frame % 64u) * 0.754877);

    var light = vec3f(0.0);
    var open = 0.0;
    var weight = 0.0;
    let slices = max(sp.slices, 1u);
    let steps = max(sp.steps, 1u);
    for (var s = 0u; s < slices; s++) {
        let phi = (f32(s) + noise) * SSGI_PI / f32(slices);
        let dir = vec2f(cos(phi), sin(phi));
        // the slice's plane holds the view vector and dir; the normal's angle within it
        let dir3 = vec3f(dir, 0.0);
        let ortho = dir3 - dot(dir3, v) * v;
        let axis = normalize(cross(dir3, v));
        let projN = n - axis * dot(n, axis);
        let projLen = length(projN);
        if (projLen < 1e-4) { continue; }
        let cosN = clamp(dot(projN, v) / projLen, -1.0, 1.0);
        let nAngle = sign(dot(ortho, projN)) * acos(cosN);
        var hidden = 0u;
        for (var side = 0u; side < 2u; side++) {
            let sgn = select(-1.0, 1.0, side == 1u);
            for (var k = 0u; k < steps; k++) {
                // denser near the point, where occluders matter most
                var t = (f32(k) + stepNoise) / f32(steps);
                t = t * t;
                let offset = dir * (sgn * max(t * radiusPx, 1.0 + f32(k)));
                let suv = uv + vec2f(offset.x, -offset.y) / sp.fullSize;
                if (any(suv < vec2f(0.0)) || any(suv > vec2f(1.0))) { break; }
                let spx = ssgiPixel(suv);
                let sd = ssgiDepth(spx);
                // the sky is not an occluder (its light is the ambient term)
                if (sd >= 1.0) { continue; }
                let q = ssgiViewPos((vec2f(spx) + 0.5) / sp.fullSize, sd);
                let front = q - p;
                let dist = length(front);
                if (dist > sp.radius || dist < 1e-4) { continue; }
                // on or under the point's own plane: no occluder. Tested in 3D, where rounding the
                // sample to its pixel (off the slice line) cannot lift a plane's own points above it
                if (dot(front, n) < 0.05 * dist) { continue; }
                let back = front + normalize(q) * sp.thickness;
                // the sample's front and back, as angles from the view vector in the slice,
                // then as shares of the cosine-weighted hemisphere around the normal
                let hf = sgn * acos(clamp(dot(front / dist, v), -1.0, 1.0));
                let hb = sgn * acos(clamp(dot(normalize(back), v), -1.0, 1.0));
                let tf = hemisphereShare(hf - nAngle);
                let tb = hemisphereShare(hb - nAngle);
                let bits = sectors(min(tf, tb), max(tf, tb));
                let fresh = bits & ~hidden;
                if (fresh != 0u) {
                    // the light the sample sends back toward this point (none from its back)
                    var facing = 1.0;
                    let sn = ssgiWorldNormal(spx);
                    if (sn.w > 0.0) {
                        facing = saturate(dot(normalize((sp.view * vec4f(sn.xyz, 0.0)).xyz), -front / dist) * 4.0);
                    }
                    let radiance = textureLoad(colorTex, spx, 0).rgb;
                    light += radiance * (facing * f32(countOneBits(fresh)) / 32.0 * projLen);
                }
                hidden |= bits;
            }
        }
        open += (1.0 - f32(countOneBits(hidden)) / 32.0) * projLen;
        weight += projLen;
    }
    // each slice's sectors average the radiance over the cosine-weighted hemisphere: pi times it
    // is the irradiance
    let e = light * (SSGI_PI / max(weight, 1e-4));
    textureStore(outTex, gid.xy, vec4f(min(e, vec3f(60000.0)), open / max(weight, 1e-4)));
}
