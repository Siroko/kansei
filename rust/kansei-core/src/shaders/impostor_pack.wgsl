// Impostor bake: pack one frame's render (the GBuffer targets and depth, `supersample` times the
// frame's size) into the atlases' frame, mip 0. Per atlas texel: coverage (the share of its
// render texels the parts drew), and over those the albedo (the colour where a material wrote no
// albedo), the object-space normal (from the depth where a material wrote none) and the depth.
// Empty texels take the values of the nearest covered render texel within DILATE atlas texels,
// so filtering and parallax near the silhouette read the object, not the background.

struct Pack {
    right      : vec3f,   // the frame's view axes, object space (dir points at the viewer)
    radius     : f32,
    up         : vec3f,
    frameSize  : u32,     // atlas texels per frame side
    dir        : vec3f,
    supersample: u32,     // render texels per atlas texel side
    frame      : vec2u,   // column, row
    _pad       : vec2u,
}

@group(0) @binding(0) var colorTex : texture_2d<f32>;
@group(0) @binding(1) var normalTex : texture_2d<f32>;
@group(0) @binding(2) var albedoTex : texture_2d<f32>;
@group(0) @binding(3) var depthTex : texture_depth_2d;
@group(0) @binding(4) var albedoOut : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var normalDepthOut : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<uniform> pack : Pack;

const DILATE : i32 = 4;

fn renderSize() -> i32 {
    return i32(pack.frameSize * pack.supersample);
}

fn covered(p : vec2i) -> bool {
    let n = renderSize();
    if (any(p < vec2i(0)) || any(p >= vec2i(n))) { return false; }
    return textureLoad(depthTex, p, 0) < 1.0;
}

// View-space position of render texel p (x right, y up, z towards the viewer; depth over [0, 2r]).
fn viewPos(p : vec2i) -> vec3f {
    let n = f32(renderSize());
    let ndc = (vec2f(p) + 0.5) / n * 2.0 - 1.0;
    let depth = textureLoad(depthTex, p, 0);
    return vec3f(ndc.x * pack.radius, -ndc.y * pack.radius, -depth * 2.0 * pack.radius);
}

fn toObject(v : vec3f) -> vec3f {
    return pack.right * v.x + pack.up * v.y + pack.dir * v.z;
}

// The object-space normal at covered render texel p: the material's, or the depth's slope.
fn normalAt(p : vec2i) -> vec3f {
    let raw = textureLoad(normalTex, p, 0).xyz;
    if (dot(raw, raw) > 1e-4) {
        return normalize(raw * 2.0 - 1.0);
    }
    let c = viewPos(p);
    let step = 2.0 * pack.radius / f32(renderSize());
    var dx = vec3f(step, 0.0, 0.0);
    var dy = vec3f(0.0, -step, 0.0);
    if (covered(p + vec2i(1, 0))) { dx = viewPos(p + vec2i(1, 0)) - c; } else if (covered(p - vec2i(1, 0))) { dx = c - viewPos(p - vec2i(1, 0)); }
    if (covered(p + vec2i(0, 1))) { dy = viewPos(p + vec2i(0, 1)) - c; } else if (covered(p - vec2i(0, 1))) { dy = c - viewPos(p - vec2i(0, 1)); }
    var n = normalize(cross(dy, dx));
    if (n.z < 0.0) { n = -n; }
    return toObject(n);
}

fn albedoAt(p : vec2i) -> vec3f {
    let a = textureLoad(albedoTex, p, 0);
    if (a.a > 0.0) { return a.rgb; }
    return textureLoad(colorTex, p, 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= vec2u(pack.frameSize))) { return; }
    let ss = i32(pack.supersample);
    let base = vec2i(gid.xy) * ss;
    var albedo = vec3f(0.0);
    var normal = vec3f(0.0);
    var depth = 0.0;
    var count = 0;
    for (var y = 0; y < ss; y++) {
        for (var x = 0; x < ss; x++) {
            let p = base + vec2i(x, y);
            if (covered(p)) {
                albedo += albedoAt(p);
                normal += normalAt(p);
                depth += textureLoad(depthTex, p, 0);
                count++;
            }
        }
    }
    var coverage = f32(count) / f32(ss * ss);
    if (count > 0) {
        albedo /= f32(count);
        depth /= f32(count);
        normal = normalize(normal + vec3f(0.0, 1e-6, 0.0));
    } else {
        // the nearest covered render texel, or the frame facing the viewer at the centre plane
        normal = pack.dir;
        depth = 0.5;
        var best = 1e9;
        let centre = base + vec2i(ss / 2);
        let reach = DILATE * ss;
        for (var y = -reach; y <= reach; y++) {
            for (var x = -reach; x <= reach; x++) {
                let p = centre + vec2i(x, y);
                let d2 = f32(x * x + y * y);
                if (d2 < best && covered(p)) {
                    best = d2;
                    albedo = albedoAt(p);
                    normal = normalAt(p);
                    depth = textureLoad(depthTex, p, 0);
                }
            }
        }
    }
    let texel = vec2i(pack.frame * pack.frameSize + gid.xy);
    textureStore(albedoOut, texel, vec4f(albedo, coverage));
    textureStore(normalDepthOut, texel, vec4f(normal * 0.5 + 0.5, depth));
}
