// Planar reflection from the screen (PlanarReflection::screen_space), as pixel-projected
// reflections do (Cichocki, "Optimized pixel-projected reflections for planar reflectors", 2017):
// last frame's lit colour and depth (the GBuffer, before this frame's pass overwrites them), each
// pixel above the plane mirrored across it and projected with this frame's camera into the
// reflection texture. Where several pixels land on one texel the mirror sees the lowest above the
// plane (the first along the reflected ray): `project` keeps each texel's least height (atomicMin
// on its bits, which order as positive floats do), `own` the least source index among the pixels
// at that height, and `resolve` writes their colour and the reflected path length as the rendered
// resolve does, closing the gaps of a texel or two the projection's stretch leaves. What the
// screen did not see reads as sky (SKY_DISTANCE), which materials fill from their environment.

struct Params {
    prevInvViewProj : mat4x4f,   // last frame's camera: its pixels -> world
    viewProj        : mat4x4f,   // this frame's camera
    plane           : vec4f,     // n, d with n·p + d = 0, n toward the reflected side
    cameraPos       : vec3f,     // this frame's
    minHeight       : f32,       // metres above the plane below which a pixel is the surface itself
    srcSize         : vec2u,
    dstSize         : vec2u,
    rect            : vec4f,     // screen uv the reflection is needed in (x0, y0, x1, y1; y down)
}

@group(0) @binding(0) var srcColor : texture_2d<f32>;
@group(0) @binding(1) var srcDepth : texture_depth_2d;
@group(0) @binding(2) var<storage, read_write> heights : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> owners  : array<atomic<u32>>;
@group(0) @binding(4) var dst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> p : Params;

struct ReflectionFogParams {
    viewProj : mat4x4f,   // of the mirrored camera the volume was built from
    gridNear : f32,
    gridFar  : f32,
    gridD    : f32,
    enabled  : u32,
}

@group(0) @binding(6) var fogVolume  : texture_3d<f32>;
@group(0) @binding(7) var fogSampler : sampler;
@group(0) @binding(8) var<uniform> rf : ReflectionFogParams;

const NONE : u32 = 0xffffffffu;
const SKY_DISTANCE : f32 = 65504.0;   // largest f16
// A pixel closer to the plane than this share of its distance is taken for the surface itself:
// its height is known from the depth buffer to much better than that (the depth's error times the
// view ray's slope, which is shallow where the distance is large)
const SURFACE_SLOPE : f32 = 0.0002;

struct Source {
    world  : vec3f,   // where the pixel's surface is
    height : f32,     // above the plane
    texel  : u32,     // the reflection texel its mirror image lands in (NONE: none)
}

fn world_at(pixel : vec2u) -> vec4f {
    let depth = textureLoad(srcDepth, pixel, 0);
    // the far plane (sky), and depth never written (a GBuffer just created)
    if (depth >= 1.0 || depth <= 0.0) { return vec4f(0.0, 0.0, 0.0, -1.0); }
    let uv = (vec2f(pixel) + 0.5) / vec2f(p.srcSize);
    let h = p.prevInvViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return vec4f(h.xyz / h.w, 1.0);
}

fn source(pixel : vec2u) -> Source {
    var s = Source(vec3f(0.0), 0.0, NONE);
    let w = world_at(pixel);
    if (w.w < 0.0) { return s; }
    s.world = w.xyz;
    s.height = dot(p.plane.xyz, s.world) + p.plane.w;
    if (s.height <= max(p.minHeight, SURFACE_SLOPE * length(s.world - p.cameraPos))) { return s; }
    let mirrored = s.world - 2.0 * s.height * p.plane.xyz;
    let clip = p.viewProj * vec4f(mirrored, 1.0);
    if (clip.w <= 1e-4) { return s; }
    let ndc = clip.xy / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < p.rect.xy) || any(uv >= p.rect.zw)) { return s; }
    let texel = min(vec2u(uv * vec2f(p.dstSize)), p.dstSize - 1u);
    s.texel = texel.y * p.dstSize.x + texel.x;
    return s;
}

@compute @workgroup_size(8, 8)
fn clear(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= p.dstSize)) { return; }
    let i = gid.y * p.dstSize.x + gid.x;
    atomicStore(&heights[i], NONE);
    atomicStore(&owners[i], NONE);
}

@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= p.srcSize)) { return; }
    let s = source(gid.xy);
    if (s.texel != NONE) { atomicMin(&heights[s.texel], bitcast<u32>(s.height)); }
}

@compute @workgroup_size(8, 8)
fn own(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= p.srcSize)) { return; }
    let s = source(gid.xy);
    if (s.texel != NONE && atomicLoad(&heights[s.texel]) == bitcast<u32>(s.height)) {
        atomicMin(&owners[s.texel], gid.y * p.srcSize.x + gid.x);
    }
}

// Whether the camera saw the surface itself at this texel's place on screen (last frame).
fn on_surface(t : vec2u) -> bool {
    let w = world_at(min(vec2u((vec2f(t) + 0.5) / vec2f(p.dstSize) * vec2f(p.srcSize)), p.srcSize - 1u));
    if (w.w < 0.0) { return false; }
    let height = dot(p.plane.xyz, w.xyz) + p.plane.w;
    return abs(height) <= max(p.minHeight, SURFACE_SLOPE * length(w.xyz - p.cameraPos));
}

// Whether reflections lie within `reach` of `t` up, down, left and right: a hole inside what is
// reflected (sky between reflected things opens onto more sky). Where the camera saw something
// else than the surface counts as reflected: the nearest reflection goes on into it.
fn enclosed(t : vec2i, reach : i32) -> bool {
    var dirs = array<vec2i, 4>(vec2i(0, 1), vec2i(0, -1), vec2i(-1, 0), vec2i(1, 0));
    for (var d = 0; d < 4; d++) {
        var found = false;
        for (var k = 1; k <= reach; k++) {
            let n = t + dirs[d] * k;
            if (any(n < vec2i(0)) || any(n >= vec2i(p.dstSize))) { break; }
            if (atomicLoad(&owners[u32(n.y) * p.dstSize.x + u32(n.x)]) != NONE || !on_surface(vec2u(n))) {
                found = true;
                break;
            }
        }
        if (!found) { return false; }
    }
    return true;
}

// The nearest reflected texel within `reach`, below first (up the reflection), then above, then
// beside.
fn nearest(t : vec2i, reach : i32) -> u32 {
    var dirs = array<vec2i, 4>(vec2i(0, 1), vec2i(0, -1), vec2i(-1, 0), vec2i(1, 0));
    for (var k = 1; k <= reach; k++) {
        for (var d = 0; d < 4; d++) {
            let n = t + dirs[d] * k;
            if (any(n < vec2i(0)) || any(n >= vec2i(p.dstSize))) { continue; }
            let o = atomicLoad(&owners[u32(n.y) * p.dstSize.x + u32(n.x)]);
            if (o != NONE) { return o; }
        }
    }
    return NONE;
}

// The nearest reflected texel within `reach` along +step and along -step: (the lower one's source
// index, its height bits), or NONE when either side has none. The screen's border counts as a
// side (within three times the reach), and then so may the other be: what lies beyond it was never
// on screen, so the reflection beside it goes on to it.
fn bracket(t : vec2i, step : vec2i, reach : i32) -> vec2u {
    var found = array<vec3u, 2>(vec3u(NONE), vec3u(NONE));   // source index, height bits, distance
    var border = array<bool, 2>(false, false);
    for (var side = 0; side < 2; side++) {
        let dir = select(step, -step, side == 1);
        for (var k = 1; k <= 3 * reach; k++) {
            let n = t + dir * k;
            if (any(n < vec2i(0)) || any(n >= vec2i(p.dstSize))) {
                border[side] = true;
                break;
            }
            let j = u32(n.y) * p.dstSize.x + u32(n.x);
            let o = atomicLoad(&owners[j]);
            if (o != NONE) {
                found[side] = vec3u(o, atomicLoad(&heights[j]), u32(k));
                break;
            }
        }
    }
    let near = vec2<bool>(found[0].z <= u32(reach), found[1].z <= u32(reach));
    if (near.x && near.y) {
        return select(found[1].xy, found[0].xy, found[0].y <= found[1].y);
    }
    if (border[1] && found[0].x != NONE) { return found[0].xy; }
    if (border[0] && found[1].x != NONE) { return found[1].xy; }
    return vec2u(NONE);
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= p.dstSize)) { return; }
    // materials read the reflection mirrored left-right (as the rendered mirror is drawn)
    let out = vec2u(p.dstSize.x - 1u - gid.x, gid.y);
    let i = gid.y * p.dstSize.x + gid.x;
    var owner = atomicLoad(&owners[i]);
    if (owner == NONE) {
        // a gap the projection's stretch left: reflections on both sides of it (within 3 texels
        // up and down, or 2 left and right; or the screen's border), closed with the lower. With
        // one side only it is the edge of what is reflected, and sky beyond it.
        let v = bracket(vec2i(gid.xy), vec2i(0, 1), 3);
        let h = bracket(vec2i(gid.xy), vec2i(1, 0), 2);
        owner = select(h.x, v.x, v.y <= h.y);
    }
    if (owner == NONE && enclosed(vec2i(gid.xy), 24)) {
        // a hole inside what is reflected: what the mirror sees from below and the camera did not
        // from above (under a canopy, under eaves), taken as the reflection nearest it
        owner = nearest(vec2i(gid.xy), 24);
    }
    if (owner == NONE && !on_surface(gid.xy)) {
        // where the camera saw something else than the surface, the reflection is only read by
        // lookups displaced by ripples or blurred by roughness near the surface's edge: the
        // nearest reflection goes on into it (the rendered mirror has what it sees there)
        owner = nearest(vec2i(gid.xy), 8);
    }
    if (owner == NONE) {
        textureStore(dst, out, vec4f(0.0, 0.0, 0.0, SKY_DISTANCE));
        return;
    }
    let pixel = vec2u(owner % p.srcSize.x, owner / p.srcSize.x);
    var color = textureLoad(srcColor, pixel, 0).rgb;
    let world = world_at(pixel).xyz;
    let height = dot(p.plane.xyz, world) + p.plane.w;
    let mirrored = world - 2.0 * height * p.plane.xyz;
    // the reflected path from the camera: to the mirror image, as long
    let dist = min(length(mirrored - p.cameraPos), SKY_DISTANCE);
    if (rf.enabled != 0u) {
        let clip = rf.viewProj * vec4f(world, 1.0);
        let ndc = clip.xy / clip.w;
        let slice = depthToSlice(max(clip.w, rf.gridNear), rf.gridNear, rf.gridFar, rf.gridD);
        let fog = textureSampleLevel(fogVolume, fogSampler, vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, saturate(slice / rf.gridD)), 0.0);
        color = color * fog.a + fog.rgb;
    }
    textureStore(dst, out, vec4f(color, dist));
}
