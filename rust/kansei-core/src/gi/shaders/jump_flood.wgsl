// The distance field (gi::JumpFloodSdf): miaumiau.cat/?p=1457's last step, without its 2D-atlas
// workaround. Seeds are the occupied voxels; each flood pass reads the 27 voxels `step` apart
// around a voxel and keeps the nearest seed any of them knows (Rong and Tan 2006), the step
// halving from half the volume to 1, then 2 and 1 again (JFA+2), which leaves almost every voxel
// its exact nearest seed. Seeds ping-pong between a sampled texture_3d<u32> and a write-only
// r32uint storage view (read-write storage textures need adapter-specific features on native
// wgpu 24). The last pass writes the distance in metres to an r32float field, which Kansei can
// filter (it requires FLOAT32_FILTERABLE). Unsigned, like the article's: a surface is where the
// distance falls under half a voxel.

struct SdfParams {
    dims      : vec3u,
    step      : u32,      // this flood pass's offset, voxels
    voxelSize : f32,
    threshold : f32,      // opacity seeds: a voxel this opaque or more is occupied
    seedMode  : u32,      // 0: the voxelizer's surfaces (4 u32 a voxel), 1: the volume's opacity
    hasDynamic: u32,      // distance pass: also the dynamic seeds
}

const SDF_NONE : u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> jp : SdfParams;
@group(0) @binding(1) var<storage, read> surfaces : array<u32>;
@group(0) @binding(2) var opacity : texture_3d<f32>;
@group(0) @binding(3) var seedsIn : texture_3d<u32>;
@group(0) @binding(4) var seedsOut : texture_storage_3d<r32uint, write>;
@group(0) @binding(5) var dynamicSeeds : texture_3d<u32>;
@group(0) @binding(6) var distanceOut : texture_storage_3d<r32float, write>;

fn packSeed(v: vec3u) -> u32 { return v.x | (v.y << 10u) | (v.z << 20u); }
fn unpackSeed(s: u32) -> vec3f { return vec3f(f32(s & 1023u), f32((s >> 10u) & 1023u), f32(s >> 20u)); }

fn seedDistance2(s: u32, here: vec3f) -> f32 {
    if (s == SDF_NONE) { return 1e30; }
    let d = unpackSeed(s) - here;
    return dot(d, d);
}

@compute @workgroup_size(4, 4, 4)
fn seed(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= jp.dims)) { return; }
    var occupied = false;
    if (jp.seedMode == 0u) {
        // gi::SURFACE_WORDS_PER_VOXEL words: a surface counted (albedo's count) or emitting
        let idx = 4u * ((gid.z * jp.dims.y + gid.y) * jp.dims.x + gid.x);
        occupied = (surfaces[idx] >> 24u) != 0u || surfaces[idx + 3u] != 0u;
    } else {
        occupied = textureLoad(opacity, gid, 0).a >= jp.threshold;
    }
    textureStore(seedsOut, gid, vec4u(select(SDF_NONE, packSeed(gid), occupied), 0u, 0u, 0u));
}

@compute @workgroup_size(4, 4, 4)
fn flood(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= jp.dims)) { return; }
    let here = vec3f(gid);
    var best = textureLoad(seedsIn, gid, 0).x;
    var bestD = seedDistance2(best, here);
    let step = i32(jp.step);
    for (var k = 0u; k < 27u; k++) {
        if (k == 13u) { continue; }
        let o = vec3i(i32(k % 3u) - 1, i32((k / 3u) % 3u) - 1, i32(k / 9u) - 1) * step;
        let q = vec3i(gid) + o;
        if (any(q < vec3i(0)) || any(q >= vec3i(jp.dims))) { continue; }
        let s = textureLoad(seedsIn, vec3u(q), 0).x;
        let d = seedDistance2(s, here);
        if (d < bestD) { bestD = d; best = s; }
    }
    textureStore(seedsOut, gid, vec4u(best, 0u, 0u, 0u));
}

// metres from the voxel's centre to the nearest occupied voxel's surface (its centre less half a
// voxel), 0 in an occupied voxel; 1e4 with no seed at all
@compute @workgroup_size(4, 4, 4)
fn distance(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= jp.dims)) { return; }
    let here = vec3f(gid);
    var d2 = seedDistance2(textureLoad(seedsIn, gid, 0).x, here);
    if (jp.hasDynamic != 0u) {
        d2 = min(d2, seedDistance2(textureLoad(dynamicSeeds, gid, 0).x, here));
    }
    var d = 1e4;
    if (d2 < 1e29) { d = max(sqrt(d2) - 0.5, 0.0) * jp.voxelSize; }
    textureStore(distanceOut, gid, vec4f(d, 0.0, 0.0, 0.0));
}
