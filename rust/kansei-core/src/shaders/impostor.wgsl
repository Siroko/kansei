// Octahedral impostors in materials (impostors::IMPOSTOR_WGSL). An `Impostor` is baked from
// renderables into two atlases of N x N frames, each an orthographic view of the object from a
// direction of the octahedral (or hemi-octahedral) grid: albedo and coverage, and the
// object-space normal and depth. Bind `Impostor::params()` as a uniform `KanseiImpostor`, its two
// atlases as texture_2d<f32> and a trilinear, clamping sampler, draw `impostors::billboard_geometry`
// instanced like the meshes it stands in for, and:
//
//     // vertex: the camera in the instance's object space, then the billboard corner
//     let objectEye = inverse_placement(eye_world);
//     let local = kansei_impostor_corner(impostor, v.position.xy, objectEye);
//     out.clip = projection_matrix * view_matrix * placement(local);
//     out.local = local;  out.eye = objectEye;
//
//     // fragment (from uniform control flow: it takes derivatives)
//     let s = kansei_impostor_sample(impostor, albedo_atlas, normal_depth_atlas, samp, in.local, in.eye);
//     if (s.alpha < 0.5) { discard; }
//     // s.albedo, s.normal and s.position (object space): place them as the instance, and shade
//
// The billboard's own depth is that of a plane through the bounds' centre, facing the camera. To
// intersect other geometry as the mesh would, write @builtin(frag_depth) from placement(s.position);
// it costs: fragments that write depth are depth-tested only after they run, so every covered
// pixel of every impostor behind others is shaded (a few times the cost on tile-based GPUs).
//
// A shadow pass draws the same billboard facing the light (group 1 is its view): cut it out with
// kansei_impostor_sample(...).alpha in `shadow_fragment_entry`.

struct KanseiImpostor {
    center : vec3f,   // the bounds' centre, object space
    radius : f32,     // the bounding sphere's radius
    extent : vec3f,   // the bounding box's half size
    frames : u32,     // frames per side of the atlases
    hemi   : u32,     // 1: hemi-octahedral (views from above the horizon only)
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,
}

struct KanseiImpostorSample {
    albedo   : vec3f,   // the material's albedo (its colour if it writes no albedo, see Impostor)
    alpha    : f32,     // coverage: cut out below 0.5
    normal   : vec3f,   // object space, unit length
    position : vec3f,   // the surface point, object space
}

fn kansei_impostor_signs(v : vec2f) -> vec2f {
    return select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

// The grid position ([-1, 1] squared) of an object-space direction, y up. Hemi-octahedral
// grids hold the upper hemisphere only: directions below the horizon map to its edge.
fn kansei_impostor_encode(dir : vec3f, hemi : bool) -> vec2f {
    let v = dir / (abs(dir.x) + abs(dir.y) + abs(dir.z));
    if (hemi) {
        let h = v.xz / max(abs(v.x) + abs(v.z), select(1e-12, 1.0, v.y > 0.0));
        return vec2f(h.x + h.y, h.x - h.y);
    }
    if (v.y >= 0.0) {
        return v.xz;
    }
    return (1.0 - abs(v.zx)) * kansei_impostor_signs(v.xz);
}

// The unit direction at a grid position.
fn kansei_impostor_decode(g : vec2f, hemi : bool) -> vec3f {
    if (hemi) {
        let x = (g.x + g.y) * 0.5;
        let z = (g.x - g.y) * 0.5;
        return normalize(vec3f(x, 1.0 - abs(x) - abs(z), z));
    }
    let y = 1.0 - abs(g.x) - abs(g.y);
    if (y < 0.0) {
        let xz = (1.0 - abs(g.yx)) * kansei_impostor_signs(g);
        return normalize(vec3f(xz.x, y, xz.y));
    }
    return normalize(vec3f(g.x, y, g.y));
}

// The right and up axes of a view from direction `dir` (unit, pointing at the viewer): up is y,
// or z looking straight down or up. The bake's frames use the same.
fn kansei_impostor_basis(dir : vec3f) -> mat2x3f {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(dir.y) > 0.999);
    let right = normalize(cross(up, dir));
    return mat2x3f(right, cross(dir, right));
}

// The direction frame `frame` (column, row) was baked from.
fn kansei_impostor_frame_dir(imp : KanseiImpostor, frame : vec2f) -> vec3f {
    return kansei_impostor_decode((frame + 0.5) / f32(imp.frames) * 2.0 - 1.0, imp.hemi != 0u);
}

// Billboard corner `corner` (each coordinate -1 or 1), object space, for an eye at `objectEye`
// (object space): a quad through the bounds' centre facing the eye, just covering the bounds seen
// from there (the box's outline, or the sphere's close up).
fn kansei_impostor_corner(imp : KanseiImpostor, corner : vec2f, objectEye : vec3f) -> vec3f {
    let toEye = objectEye - imp.center;
    let dist = length(toEye);
    let v = toEye / max(dist, 1e-6);
    let b = kansei_impostor_basis(v);
    let r2 = imp.radius * imp.radius;
    let half = imp.radius * dist / sqrt(max(dist * dist - r2, 0.01 * r2));
    var lo = vec2f(-half);
    var hi = vec2f(half);
    // the box's corners projected from the eye onto the plane, while they are all in front of it
    if (dot(abs(v), imp.extent) < dist * 0.99) {
        var boxLo = vec2f(1e30);
        var boxHi = vec2f(-1e30);
        for (var k = 0u; k < 8u; k++) {
            let o = (vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u)) * 2.0 - 1.0) * imp.extent;
            let p = vec2f(dot(o, b[0]), dot(o, b[1])) * (dist / (dist - dot(o, v)));
            boxLo = min(boxLo, p);
            boxHi = max(boxHi, p);
        }
        lo = max(lo, boxLo);
        hi = min(hi, boxHi);
    }
    let q = mix(lo, hi, corner * 0.5 + 0.5);
    return imp.center + b[0] * q.x + b[1] * q.y;
}

// Frame-local uv ([0, 1], y down) of object-space point `p`, in a frame with axes `b`.
fn kansei_impostor_frame_uv(imp : KanseiImpostor, b : mat2x3f, p : vec3f) -> vec2f {
    let q = p - imp.center;
    return vec2f(dot(q, b[0]), -dot(q, b[1])) / (2.0 * imp.radius) + 0.5;
}

// Atlas uv of frame-local `uv` in `frame`, kept half a texel of mip `lod` inside the frame.
fn kansei_impostor_atlas_uv(imp : KanseiImpostor, atlas : texture_2d<f32>, frame : vec2f, uv : vec2f, lod : f32) -> vec2f {
    let n = f32(imp.frames);
    let texels = max(f32(textureDimensions(atlas).x) / n / exp2(floor(lod)), 1.0);
    let inset = 0.5 / texels;
    return (frame + clamp(uv, vec2f(inset), vec2f(1.0 - inset))) / n;
}

// One frame along the view ray from `eye` in direction `ray` (object space): where the ray meets
// the frame's plane, then one step of parallax to the depth stored there.
fn kansei_impostor_frame(imp : KanseiImpostor, albedoAtlas : texture_2d<f32>, normalDepthAtlas : texture_2d<f32>,
                         samp : sampler, frame : vec2f, eye : vec3f, ray : vec3f, lod : f32) -> KanseiImpostorSample {
    let d = kansei_impostor_frame_dir(imp, frame);
    let b = kansei_impostor_basis(d);
    // the ray's progress along -d (the frame's view direction), and the eye's height over the centre
    let rate = min(dot(ray, d), -1e-4);
    let eyeHeight = dot(eye - imp.center, d);
    var uv = kansei_impostor_frame_uv(imp, b, eye + ray * (eyeHeight / -rate));
    let h = textureSampleLevel(normalDepthAtlas, samp, kansei_impostor_atlas_uv(imp, normalDepthAtlas, frame, uv, lod), lod).a;
    // the stored surface's height over the centre along d: depth 0 is the frame's near plane (radius)
    let height = imp.radius * (1.0 - 2.0 * h);
    uv = kansei_impostor_frame_uv(imp, b, eye + ray * ((eyeHeight - height) / -rate));
    let atlasUV = kansei_impostor_atlas_uv(imp, albedoAtlas, frame, uv, lod);
    let a = textureSampleLevel(albedoAtlas, samp, atlasUV, lod);
    let nd = textureSampleLevel(normalDepthAtlas, samp, atlasUV, lod);
    var s : KanseiImpostorSample;
    s.albedo = a.rgb;
    s.alpha = select(0.0, a.a, all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)));
    s.normal = nd.xyz * 2.0 - 1.0;
    let q = (uv - 0.5) * 2.0 * imp.radius;
    s.position = imp.center + b[0] * q.x - b[1] * q.y + d * (imp.radius * (1.0 - 2.0 * nd.a));
    return s;
}

// The impostor seen along the ray from `objectEye` through billboard point `objectPos` (both
// object space): the three frames nearest the direction to the eye, blended by their barycentric
// weights in the grid. Call from uniform control flow (it takes derivatives for the mip).
fn kansei_impostor_sample(imp : KanseiImpostor, albedoAtlas : texture_2d<f32>, normalDepthAtlas : texture_2d<f32>,
                          samp : sampler, objectPos : vec3f, objectEye : vec3f) -> KanseiImpostorSample {
    // the mip: object-space size of a pixel against that of a frame texel
    let texel = 2.0 * imp.radius * f32(imp.frames) / f32(textureDimensions(albedoAtlas).x);
    let footprint = max(length(dpdx(objectPos)), length(dpdy(objectPos)));
    let lod = log2(max(footprint / texel, 1e-6));
    return kansei_impostor_sample_lod(imp, albedoAtlas, normalDepthAtlas, samp, objectPos, objectEye, lod);
}

// kansei_impostor_sample at mip `lod` (clamped to the atlases' mips), from any stage.
fn kansei_impostor_sample_lod(imp : KanseiImpostor, albedoAtlas : texture_2d<f32>, normalDepthAtlas : texture_2d<f32>,
                              samp : sampler, objectPos : vec3f, objectEye : vec3f, lod : f32) -> KanseiImpostorSample {
    let n = f32(imp.frames);
    let level = clamp(lod, 0.0, f32(textureNumLevels(albedoAtlas) - 1u));
    let ray = normalize(objectPos - objectEye);
    let g = kansei_impostor_encode(normalize(objectEye - imp.center), imp.hemi != 0u);
    let f = clamp((g * 0.5 + 0.5) * n - 0.5, vec2f(0.0), vec2f(n - 1.0));
    let f0 = floor(f);
    let t = f - f0;
    // the grid cell's triangle holding f
    var frames = array<vec2f, 3>(f0, f0 + vec2f(1.0, 0.0), f0 + vec2f(0.0, 1.0));
    var weights = vec3f(1.0 - t.x - t.y, t.x, t.y);
    if (t.x + t.y > 1.0) {
        frames[0] = f0 + vec2f(1.0);
        weights = vec3f(t.x + t.y - 1.0, 1.0 - t.y, 1.0 - t.x);
    }
    var out : KanseiImpostorSample;
    var normal = vec3f(0.0);
    for (var k = 0; k < 3; k++) {
        let s = kansei_impostor_frame(imp, albedoAtlas, normalDepthAtlas, samp, min(frames[k], vec2f(n - 1.0)), objectEye, ray, level);
        let w = weights[k];
        out.albedo += s.albedo * w;
        out.alpha += s.alpha * w;
        out.position += s.position * w;
        normal += s.normal * w;
    }
    out.normal = normalize(normal + vec3f(0.0, 1e-6, 0.0));
    return out;
}
