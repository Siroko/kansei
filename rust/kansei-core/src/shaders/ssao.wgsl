// Screen-space ambient occlusion, the occlusion pass (after ssao_common.wgsl): a hemisphere of
// samples around each point's normal, turned about the normal by a per-pixel angle, projected
// back to the screen and compared with the depth there. It writes the share left open (1: none
// occluded) and the point's distance from the camera (0 on the sky), for the blur.

@group(0) @binding(3) var<uniform> kernel : array<vec4f, 32>;
@group(0) @binding(4) var aoOut : texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= params.size)) { return; }
    let px = vec2i(gid.xy);
    let depth = ssaoDepth(px);
    if (depth >= 1.0) {
        textureStore(aoOut, px, vec4f(1.0, 0.0, 0.0, 0.0));
        return;
    }
    let p = ssaoViewPosAt(px);
    let n = ssaoViewNormal(px, p);
    let dist = -p.z;

    // the radius held to `maxRadiusPx` on screen; under a pixel, its samples would only find the
    // point's own surface
    let pxPerMetre = params.proj[1][1] * 0.5 * params.size.y / max(dist, 1e-4);
    let radius = min(params.radius, params.maxRadiusPx / pxPerMetre);
    if (radius * pxPerMetre < 1.0) {
        textureStore(aoOut, px, vec4f(1.0, dist, 0.0, 0.0));
        return;
    }
    // depth steps between neighbouring pixels grow with the distance
    let bias = params.bias * dist;

    // a tangent basis about the normal, turned by the pixel's angle
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t0 = normalize(cross(helper, n));
    let b0 = cross(n, t0);
    let angle = ssaoNoise(vec2f(px)) * 2.0 * SSAO_PI;
    let t = t0 * cos(angle) + b0 * sin(angle);
    let b = cross(n, t);

    var occlusion = 0.0;
    for (var i = 0u; i < params.kernelSize; i++) {
        let k = kernel[i].xyz;
        let s = p + (t * k.x + b * k.y + n * k.z) * radius;
        let clip = params.proj * vec4f(s, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc = clip.xy / clip.w;
        let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (any(suv < vec2f(0.0)) || any(suv >= vec2f(1.0))) { continue; }
        let spx = vec2i(suv * params.size);
        let sd = ssaoDepth(spx);
        if (sd >= 1.0) { continue; }
        // the surface seen along the sample's ray, and the point of it at its pixel's centre
        let surface = ssaoViewPos(suv, sd);
        let q = ssaoViewPosAt(spx);
        // occluding where it is in front of the sample and rises above the point's tangent
        // plane (a plane's own pixels lie in it, however coarse its depth steps at a grazing
        // angle), unless it is that far in front that it is something else (a range check
        // against haloes over depth discontinuities)
        let range = smoothstep(0.0, 1.0, radius / abs(p.z - q.z));
        occlusion += select(0.0, range, surface.z >= s.z + bias && dot(q - p, n) > bias);
    }
    let open = 1.0 - occlusion / f32(params.kernelSize);
    textureStore(aoOut, px, vec4f(open, dist, 0.0, 0.0));
}
