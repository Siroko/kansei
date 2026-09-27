// Per-view GPU instance culling: one thread per instance of a renderable's full instance list.
// An instance survives when its bounding sphere is inside the view's frustum and its distance
// from the LOD origin (the main camera) is inside the renderable's LOD band; survivors are copied,
// word by word, into the view's compacted instance buffer, and counted into its indirect draw.

struct CullParams {
    planes       : array<vec4f, 6>,   // world-space frustum planes, normalized, inside >= 0
    world        : mat4x4f,           // the renderable's world matrix
    lodOrigin    : vec3f,
    lodNear      : f32,
    lodFar       : f32,
    radius       : f32,               // object-space bounding radius (times maxScale)
    maxScale     : f32,               // the world matrix's largest axis scale
    count        : u32,
    strideWords  : u32,
    centerWord   : u32,               // word offset of the sphere centre (3 x f32)
    scaleWord    : u32,               // word offset of an f32 the radius scales by, or NO_WORD
    _pad         : u32,
}

struct DrawIndexedArgs {
    indexCount    : u32,
    instanceCount : atomic<u32>,
    firstIndex    : u32,
    baseVertex    : i32,
    firstInstance : u32,
}

@group(0) @binding(0) var<uniform> cp : CullParams;
@group(0) @binding(1) var<storage, read> src : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst : array<u32>;
@group(0) @binding(3) var<storage, read_write> args : DrawIndexedArgs;

const NO_WORD : u32 = 0xffffffffu;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    if (i >= cp.count) { return; }
    let base = i * cp.strideWords;
    let local = vec3f(bitcast<f32>(src[base + cp.centerWord]),
                      bitcast<f32>(src[base + cp.centerWord + 1u]),
                      bitcast<f32>(src[base + cp.centerWord + 2u]));
    var radius = cp.radius * cp.maxScale;
    if (cp.scaleWord != NO_WORD) {
        radius *= abs(bitcast<f32>(src[base + cp.scaleWord]));
    }
    let center = (cp.world * vec4f(local, 1.0)).xyz;

    let d = distance(center, cp.lodOrigin);
    if (d < cp.lodNear || d >= cp.lodFar) { return; }
    for (var k = 0u; k < 6u; k++) {
        if (dot(cp.planes[k].xyz, center) + cp.planes[k].w < -radius) { return; }
    }

    let slot = atomicAdd(&args.instanceCount, 1u);
    let out = slot * cp.strideWords;
    for (var w = 0u; w < cp.strideWords; w++) {
        dst[out + w] = src[base + w];
    }
}
