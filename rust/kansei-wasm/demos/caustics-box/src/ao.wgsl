// The studio's ambient occlusion (`ao::AoEffect`): a game's contact darkening over the lit image.
// Per pixel, `samples` points in the hemisphere round the GBuffer's normal, up to `radius` metres
// out (most of them near the surface), each projected to the screen and compared with the depth
// there; an occluder counts by how near the point lies (`falloff`: higher, tighter contact
// shadows) and fades past `radius` in depth; the unoccluded share to the power `intensity`. The rotation of the pattern changes every pixel and
// frame (interleaved gradient noise), for TAA to resolve. The sky, glass (its albedo's alpha
// 0.05-0.45) and the fluid's screen-space surface pass through.

struct AoParams {
    view        : mat4x4f,
    proj        : mat4x4f,
    invProj     : mat4x4f,
    size        : vec2f,
    radius      : f32,
    intensity   : f32,
    falloff     : f32,
    samples     : u32,
    frame       : u32,
    // 1: the occlusion alone, on white of radiance `white`
    debug       : u32,
    white       : f32,
    bias        : f32,
    _pad0       : f32,
    _pad1       : f32,
}

@group(0) @binding(0) var<uniform> ap : AoParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var emissiveTex : texture_2d<f32>;
@group(0) @binding(5) var inputTex : texture_2d<f32>;
@group(0) @binding(6) var outputTex : texture_storage_2d<rgba16float, write>;

fn depthAt(uv: vec2f) -> f32 {
    let dims = textureDimensions(depthTex);
    let q = clamp(vec2i(uv * vec2f(dims)), vec2i(0), vec2i(dims) - 1);
    return textureLoad(depthTex, q, 0);
}

fn viewPos(uv: vec2f, depth: f32) -> vec3f {
    let v = ap.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return v.xyz / v.w;
}

// Jorge Jimenez's interleaved gradient noise.
fn ign(p: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(vec2f(gid.xy) >= ap.size)) { return; }
    let px = gid.xy;
    let color = textureLoad(inputTex, px, 0);
    let uv = (vec2f(px) + 0.5) / ap.size;
    let depth = depthAt(uv);
    let albedo = textureLoad(albedoTex, px, 0);
    let skip = depth >= 1.0 || (albedo.a > 0.025 && albedo.a < 0.475) || textureLoad(emissiveTex, px, 0).a > 0.5;
    if (skip) {
        textureStore(outputTex, px, select(color, vec4f(vec3f(ap.white), color.a), ap.debug == 1u));
        return;
    }
    let p = viewPos(uv, depth);
    let nWorld = normalize(textureLoad(normalTex, px, 0).xyz * 2.0 - 1.0);
    let n = normalize((ap.view * vec4f(nWorld, 0.0)).xyz);
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    let noise = ign(vec2f(px) + f32(ap.frame % 64u) * vec2f(5.588238, 5.588238));
    let count = max(ap.samples, 1u);
    var occluded = 0.0;
    var total = 0.0;
    for (var i = 0u; i < count; i++) {
        // a spiral over the hemisphere (cosine-weighted), turned by the noise; the points' reach
        // from the surface squared toward it
        let f = (f32(i) + noise) / f32(count);
        let phi = 6.2831853 * (f32(i) * 0.618034 + noise);
        let r2 = fract(f * 7.0 + noise * 0.5);
        let sinT = sqrt(r2);
        let dir = t * (cos(phi) * sinT) + b * (sin(phi) * sinT) + n * sqrt(1.0 - r2);
        let reach = ap.radius * mix(0.05, 1.0, f * f);
        let q = p + dir * reach;
        let clip = ap.proj * vec4f(q, 1.0);
        if (clip.w <= 0.0) { continue; }
        let quv = vec2f(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
        // the nearer the point, the more an occluder there counts
        let w = pow(1.0 - f * f * 0.95, ap.falloff);
        total += w;
        if (any(quv < vec2f(0.0)) || any(quv > vec2f(1.0))) { continue; }
        let s = viewPos(quv, depthAt(quv));
        // (view space looks down -z: the scene in front of the point is nearer, larger z)
        let gap = s.z - q.z;
        if (gap > ap.bias) {
            occluded += w * smoothstep(0.0, 1.0, ap.radius / max(abs(p.z - s.z), 1e-4));
        }
    }
    // the unoccluded share to the power `intensity`: a game's AO, dark in creases and contacts
    let visibility = pow(clamp(1.0 - occluded / max(total, 1e-4), 0.0, 1.0), ap.intensity);
    if (ap.debug == 1u) {
        textureStore(outputTex, px, vec4f(vec3f(ap.white * visibility), color.a));
        return;
    }
    textureStore(outputTex, px, vec4f(color.rgb * visibility, color.a));
}
