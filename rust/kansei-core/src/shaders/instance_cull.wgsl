// Per-view GPU instance culling: one thread per instance of a renderable's full instance list, and
// one dispatch for all the views it is drawn in (y: the view). An instance survives when its
// bounds (a sphere, or a box) are inside the view's frustum and its distance from the LOD origin
// (the main camera) is inside the renderable's LOD band; survivors are copied, word by word, into
// the view's region of the compacted instances, and counted into its indirect draw.
//
// Crossfades (`crossfade` > 0, InstanceCulling::with_crossfade): each band's edges widen by half
// the crossfade width, and a survivor's record is followed by one more word, its fade: 1 inside
// the band, in (0, 1) where it fades out past its far edge (the share of its pixels it keeps), in
// (-1, 0) where it fades in before its near edge (it keeps the complementary pixels). Two LODs
// meeting at a distance with the same width fade by complementary shares there, so a material
// that drops pixels by the fade (culling::LOD_FADE_WGSL) draws each pixel from one of them.
//
// Two-phase occlusion culling (the views that ask for it, FLAG_OCCLUSION: the camera, planar
// reflections; for renderables that opt in, FLAG_TWO_PHASE):
// - `early`: of the survivors, keep those that were visible last frame (`visibility`).
// - The renderer draws them with the rest of the opaque scene and builds a depth pyramid (Hi-Z)
//   from that depth.
// - `late`: test every survivor against the pyramid. Keep the visible ones `early` did not
//   (disoccluded, or new in view) for a second draw, and record each instance's visibility for
//   the next frame's `early`.
// Nothing visible is missed: what `early` leaves out, `late` tests against this frame's depth.

// The renderable's instances: written when they change (rarely).
struct CullInstances {
    world        : mat4x4f,           // the renderable's world matrix
    shift        : vec3f,             // object-space offset of the bounds from the instance centre (x scale)
    lodNear      : f32,
    boxHalf      : vec3f,             // object-space half extents of the bounding box (x scale), FLAG_BOX
    lodFar       : f32,
    radius       : f32,               // object-space bounding radius (times maxScale)
    maxScale     : f32,               // the world matrix's largest axis scale
    count        : u32,
    strideWords  : u32,
    centerWord   : u32,               // word offset of the instance centre (3 x f32)
    scaleWord    : u32,               // word offset of an f32 the bounds scale by, or NO_WORD
    flags        : u32,               // FLAG_BOX, FLAG_CASTS_SHADOW, FLAG_TWO_PHASE
    indexCount   : u32,               // the indirect draw's (its args are cleared every frame)
    firstView    : u32,               // `main`: the view of the dispatch's first row (y = 0);
                                      // `early`: the first view of occlusionView's chunk
    capacity     : u32,               // instances per view region in `dst`
    lateSlot     : u32,               // `late`: occlusionView's second phase's draw in `args`
    layers       : u32,               // the renderable's (`Renderable::layers`)
    shadowLod    : vec2f,             // the LOD band in shadow maps (FLAG_CASTERS_ONLY views)
    reflectionLod: vec2f,             // the LOD band in planar reflections (FLAG_REFLECTION views)
    occlusionView: u32,               // `early` and `late`: the view culled in two phases
    crossfade    : f32,               // the width of the bands' crossfades (LOD distance); 0: none
    _pad1        : u32,
    _pad2        : u32,
}

// A view: all of them in one buffer, written once a frame.
struct CullView {
    planes       : array<vec4f, 6>,   // world-space frustum planes, normalized, inside >= 0
    view         : mat4x4f,           // occlusion: the view's view matrix
    proj         : mat4x4f,           // occlusion: its projection, as rasterized (jittered)
    lodOrigin    : vec3f,             // the main camera, for every view
    flags        : u32,               // FLAG_VIEW, FLAG_CASTERS_ONLY, FLAG_REFLECTION, FLAG_LAYERED, FLAG_OCCLUSION, FLAG_STATS, FLAG_REVERSE_Z
    depthSize    : vec2f,             // occlusion: the depth buffer's size in pixels
    lodScale     : f32,               // the view's LOD distance scale (reflections pick finer LODs)
    layerMask    : u32,               // FLAG_LAYERED: the layers it draws (a reflection's)
}

struct DrawArgs {
    indexCount    : u32,
    instanceCount : atomic<u32>,
    firstIndex    : u32,
    baseVertex    : i32,
    firstInstance : u32,
    culled        : array<atomic<u32>, 3>,   // FLAG_STATS: culled by the LOD band, the frustum, occlusion
}

@group(0) @binding(0) var<uniform> ci : CullInstances;
@group(0) @binding(1) var<storage, read> src : array<u32>;
// a region of `capacity` instances per view, from view `firstView` (`late`: its own buffer)
@group(0) @binding(2) var<storage, read_write> dst : array<u32>;
// every view's draw, then the second phase's (`lateSlot`)
@group(0) @binding(3) var<storage, read_write> args : array<DrawArgs>;
// `early` and `late`: 1 where the instance was visible in occlusionView last frame
@group(0) @binding(4) var<storage, read_write> visibility : array<u32>;
@group(1) @binding(0) var<storage, read> views : array<CullView>;
// `late`: the depth pyramid (farthest depth per texel, see culling::DepthPyramid)
@group(2) @binding(0) var pyramid : texture_2d<f32>;

const NO_WORD : u32 = 0xffffffffu;
const FLAG_STATS : u32 = 1u;
const FLAG_BOX : u32 = 2u;
const FLAG_REVERSE_Z : u32 = 4u;
const FLAG_CASTS_SHADOW : u32 = 8u;
const FLAG_TWO_PHASE : u32 = 16u;
const FLAG_VIEW : u32 = 32u;
const FLAG_CASTERS_ONLY : u32 = 64u;
const FLAG_LAYERED : u32 = 128u;
const FLAG_REFLECTION : u32 = 256u;
const FLAG_OCCLUSION : u32 = 512u;
const FLAG_LINEAR_DEPTH : u32 = 1024u;

// what became of an instance
const KEPT : u32 = 0u;
const LOD_CULLED : u32 = 1u;
const FRUSTUM_CULLED : u32 = 2u;
const OCCLUDED : u32 = 3u;
const UNCOUNTED : u32 = 4u;

struct Bounds {
    local  : vec3f,   // centre, object space
    center : vec3f,   // centre, world space
    radius : f32,     // sphere radius, world space
    half   : vec3f,   // box half extents, object space
}

fn bounds(i : u32) -> Bounds {
    let base = i * ci.strideWords;
    var scale = 1.0;
    if (ci.scaleWord != NO_WORD) {
        scale = abs(bitcast<f32>(src[base + ci.scaleWord]));
    }
    let c = vec3f(bitcast<f32>(src[base + ci.centerWord]),
                  bitcast<f32>(src[base + ci.centerWord + 1u]),
                  bitcast<f32>(src[base + ci.centerWord + 2u]));
    var b : Bounds;
    b.local = c + ci.shift * scale;
    b.center = (ci.world * vec4f(b.local, 1.0)).xyz;
    b.radius = ci.radius * ci.maxScale * scale;
    b.half = ci.boxHalf * scale;
    return b;
}

// The view's kind's LOD band: a shadow map's, a reflection's, or the camera's.
fn lodBand(v : u32) -> vec2f {
    if ((views[v].flags & FLAG_CASTERS_ONLY) != 0u) {
        return ci.shadowLod;
    } else if ((views[v].flags & FLAG_REFLECTION) != 0u) {
        return ci.reflectionLod;
    }
    return vec2f(ci.lodNear, ci.lodFar);
}

fn lodDistance(b : Bounds, v : u32) -> f32 {
    return distance(b.center, views[v].lodOrigin) * views[v].lodScale;
}

// The fade of a survivor in view v (see the header): 1 inside its band; past the far edge the
// share it keeps as it fades out, before the near edge minus the share it keeps as it fades in.
// A band from 0 does not fade in, nor one to infinity (f32 max) out.
fn lodFade(b : Bounds, v : u32) -> f32 {
    if (ci.crossfade <= 0.0) { return 1.0; }
    let d = lodDistance(b, v);
    let band = lodBand(v);
    let h = 0.5 * ci.crossfade;
    if (band.x > 0.0 && d < band.x + h) {
        return -saturate((d - (band.x - h)) / ci.crossfade);
    }
    if (band.y < 3.0e38 && d > band.y - h) {
        return saturate(((band.y + h) - d) / ci.crossfade);
    }
    return 1.0;
}

// KEPT, LOD_CULLED or FRUSTUM_CULLED
// (view v)
fn cull(b : Bounds, v : u32) -> u32 {
    let d = lodDistance(b, v);
    // the view's kind's band, its edges widened by half the crossfade
    let band = lodBand(v);
    let h = 0.5 * max(ci.crossfade, 0.0);
    if (d < select(band.x, band.x - h, band.x > 0.0) || d >= band.y + h) { return LOD_CULLED; }
    let isBox = (ci.flags & FLAG_BOX) != 0u;
    for (var k = 0u; k < 6u; k++) {
        let plane = views[v].planes[k];
        let n = plane.xyz;
        var r = b.radius;
        if (isBox) {
            // the box's reach along the normal: its axes are the world matrix's columns
            r = dot(abs(vec3f(dot(n, ci.world[0].xyz), dot(n, ci.world[1].xyz), dot(n, ci.world[2].xyz))), b.half);
        }
        if (dot(n, b.center) + plane.w < -r) { return FRUSTUM_CULLED; }
    }
    return KEPT;
}

// Whether the bounds are hidden: the rectangle they cover on screen, from their nearest depth,
// is behind the farthest depth under it in the pyramid (with FLAG_LINEAR_DEPTH, from their
// nearest view distance, against a pyramid of view distances).
fn occluded(b : Bounds, view : u32) -> bool {
    let reverse = (views[view].flags & FLAG_REVERSE_Z) != 0u;
    let linear = (views[view].flags & FLAG_LINEAR_DEPTH) != 0u;
    let isBox = (ci.flags & FLAG_BOX) != 0u;
    let modelView = views[view].view * ci.world;
    let centerView = (modelView * vec4f(b.local, 1.0)).xyz;
    var lo = vec2f(1e30);
    var hi = vec2f(-1e30);
    var nearest = select(1e30, -1e30, reverse);
    var nearestDistance = 1e30;
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u)) * 2.0 - 1.0;
        // the box's corners, or those of a view-aligned cube round the sphere
        var v = vec4f(centerView + corner * b.radius, 1.0);
        if (isBox) {
            v = modelView * vec4f(b.local + corner * b.half, 1.0);
        }
        let clip = views[view].proj * v;
        if (clip.w <= 0.0) { return false; }   // reaching behind the eye
        let ndc = clip.xyz / clip.w;
        lo = min(lo, ndc.xy);
        hi = max(hi, ndc.xy);
        nearest = select(min(nearest, ndc.z), max(nearest, ndc.z), reverse);
        nearestDistance = min(nearestDistance, -v.z / v.w);
    }
    if (linear) {
        nearest = nearestDistance;
    } else if ((!reverse && nearest < 0.0) || (reverse && nearest > 1.0)) {
        // reaching in front of the near plane
        return false;
    }
    // the rectangle in depth pixels (y down), clamped to the buffer
    let last = views[view].depthSize - 1.0;
    let q0 = vec2u(clamp((vec2f(lo.x, -hi.y) * 0.5 + 0.5) * views[view].depthSize, vec2f(0.0), last));
    let q1 = vec2u(clamp((vec2f(hi.x, -lo.y) * 0.5 + 0.5) * views[view].depthSize, vec2f(0.0), last));
    // the finest mip where it spans at most 2 x 2 texels: texel t of mip L covers the pixels
    // [t, t + 1) * 2^(L + 1)
    let span = max(q1.x - q0.x, q1.y - q0.y);
    let shift = clamp(32u - countLeadingZeros(span), 1u, textureNumLevels(pyramid));
    let level = i32(shift - 1u);
    let t0 = q0 >> vec2u(shift);
    let t1 = q1 >> vec2u(shift);
    let d = vec4f(textureLoad(pyramid, t0, level).x, textureLoad(pyramid, vec2u(t1.x, t0.y), level).x,
                  textureLoad(pyramid, vec2u(t0.x, t1.y), level).x, textureLoad(pyramid, t1, level).x);
    if (reverse) {
        return nearest < min(min(d.x, d.y), min(d.z, d.w));
    }
    return nearest > max(max(d.x, d.y), max(d.z, d.w));
}

// The draw's index count, from its first invocation (the renderer clears the args).
fn begin(i : u32, draw : u32) {
    if (i == 0u) { args[draw].indexCount = ci.indexCount; }
}

// Copy instance i into region `region` of `dst`, counted into draw `draw`; with crossfades its
// fade after it.
fn emit(i : u32, region : u32, draw : u32, fade : f32) {
    let slot = atomicAdd(&args[draw].instanceCount, 1u);
    let base = i * ci.strideWords;
    let fading = ci.crossfade > 0.0;
    let outStride = ci.strideWords + select(0u, 1u, fading);
    let out = (region * ci.capacity + slot) * outStride;
    for (var w = 0u; w < ci.strideWords; w++) {
        dst[out + w] = src[base + w];
    }
    if (fading) {
        dst[out + ci.strideWords] = bitcast<u32>(fade);
    }
}

var<workgroup> tallies : array<atomic<u32>, 3>;

// FLAG_STATS: add the workgroup's culled instances to draw `draw`'s `culled`, one atomic per kind.
// Every invocation calls it, from uniform control flow.
fn tally(outcome : u32, lid : u32, draw : u32, viewFlags : u32) {
    if ((viewFlags & FLAG_STATS) == 0u) { return; }
    if (outcome >= LOD_CULLED && outcome <= OCCLUDED) {
        atomicAdd(&tallies[outcome - 1u], 1u);
    }
    workgroupBarrier();
    if (lid < 3u) {
        let n = atomicLoad(&tallies[lid]);
        if (n > 0u) { atomicAdd(&args[draw].culled[lid], n); }
    }
}

// Whether the renderable is drawn in view `v` by this pass: the view is in use (a shadowed
// light's), the renderable casts shadows if only casters draw there and is on a layer the view
// draws (as `CullView::draws`), and the view does not cull it in two phases.
fn drawnIn(v : u32) -> bool {
    let flags = views[v].flags;
    return (flags & FLAG_VIEW) != 0u
        && ((flags & FLAG_CASTERS_ONLY) == 0u || (ci.flags & FLAG_CASTS_SHADOW) != 0u)
        && ((flags & FLAG_LAYERED) == 0u || (views[v].layerMask & ci.layers) != 0u)
        && !((flags & FLAG_OCCLUSION) != 0u && (ci.flags & FLAG_TWO_PHASE) != 0u);
}

// Frustum and LOD culling, a row of workgroups per view.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u, @builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) lid : u32) {
    let v = ci.firstView + wg.y;
    if (!drawnIn(v)) { return; }
    begin(gid.x, v);
    var outcome = UNCOUNTED;
    if (gid.x < ci.count) {
        let b = bounds(gid.x);
        outcome = cull(b, v);
        if (outcome == KEPT) { emit(gid.x, wg.y, v, lodFade(b, v)); }
    }
    tally(outcome, lid, v, views[v].flags);
}

// Occlusion, first phase in occlusionView: the instances in view that were visible last frame.
@compute @workgroup_size(64)
fn early(@builtin(global_invocation_id) gid : vec3u, @builtin(local_invocation_index) lid : u32) {
    let i = gid.x;
    let v = ci.occlusionView;
    begin(i, v);
    var outcome = UNCOUNTED;
    if (i < ci.count) {
        let b = bounds(i);
        outcome = cull(b, v);
        if (outcome == KEPT) {
            if (visibility[i] != 0u) {
                emit(i, v - ci.firstView, v, lodFade(b, v));
            } else {
                outcome = UNCOUNTED;   // left to `late`
            }
        }
    }
    tally(outcome, lid, v, views[v].flags);
}

// Occlusion, second phase: the instances in view and not hidden in the pyramid of the first
// phase's depth that `early` did not draw; and every instance's visibility, for the next frame.
@compute @workgroup_size(64)
fn late(@builtin(global_invocation_id) gid : vec3u, @builtin(local_invocation_index) lid : u32) {
    let i = gid.x;
    let v = ci.occlusionView;
    begin(i, ci.lateSlot);
    var outcome = UNCOUNTED;
    if (i < ci.count) {
        let b = bounds(i);
        let inView = cull(b, v) == KEPT;
        let visible = inView && !occluded(b, v);
        if (inView && visibility[i] == 0u) {
            if (visible) {
                emit(i, 0u, ci.lateSlot, lodFade(b, v));
            } else {
                outcome = OCCLUDED;
            }
        }
        visibility[i] = select(0u, 1u, visible);
    }
    tally(outcome, lid, ci.lateSlot, views[v].flags);
}
