// Gathers a source's world triangles into the grid's (rt/grid.rs). A source is a mesh (`RtMesh`
// words: where its vertices and indices start, how many vertices and triangles it has, then the
// vertices, 4 words each, and the indices) placed once per record: workgroups of (64 triangles, record), the
// records past 65535 in z. A thread
// places its triangle (`kansei_rt_place`, prepended, puts a mesh point where a record puts it,
// then `src.world`), drops it when its box misses the grid's (widened by epsilon) or it has no
// area, and the workgroup's survivors take their slots with one atomic. Included after
// rt_types.wgsl.

struct RtSource {
    world       : mat4x4f,
    triangles   : u32,
    strideWords : u32,
    firstRecord : u32,
    // the records, unless `countWord` names the word of `args` that holds how many
    records     : u32,
    countWord   : u32,
    surface     : u32,
    albedo      : u32,
    // the source's id, shifted to bits 20 and up
    source      : u32,
}

const KANSEI_RT_NO_WORD : u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> grid : KanseiRtGrid;
@group(0) @binding(1) var<storage, read_write> triangles : array<vec4f>;
// [0]: triangles claimed (more than the capacity when it overflowed)
@group(0) @binding(2) var<storage, read_write> counters : array<atomic<u32>>;
@group(1) @binding(0) var<uniform> src : RtSource;
@group(1) @binding(1) var<storage, read> mesh : array<u32>;
@group(1) @binding(2) var<storage, read> records : array<u32>;
@group(1) @binding(3) var<storage, read> args : array<u32>;

fn kansei_rt_record_f32(record: u32, word: u32) -> f32 {
    return bitcast<f32>(records[record * src.strideWords + word]);
}

fn kansei_rt_record_vec3(record: u32, word: u32) -> vec3f {
    return vec3f(kansei_rt_record_f32(record, word), kansei_rt_record_f32(record, word + 1u), kansei_rt_record_f32(record, word + 2u));
}

fn kansei_rt_record_vec4(record: u32, word: u32) -> vec4f {
    return vec4f(kansei_rt_record_vec3(record, word), kansei_rt_record_f32(record, word + 3u));
}

// Mesh vertex `v` (`RtMesh`: x, y, z, then the uv packed as two f16).
fn meshPosition(v: u32) -> vec3f {
    let at = mesh[0] + v * 4u;
    return vec3f(bitcast<f32>(mesh[at]), bitcast<f32>(mesh[at + 1u]), bitcast<f32>(mesh[at + 2u]));
}

fn meshUv(v: u32) -> u32 {
    return mesh[mesh[0] + v * 4u + 3u];
}

fn recordCount() -> u32 {
    if (src.countWord == KANSEI_RT_NO_WORD) {
        return src.records;
    }
    return args[src.countWord];
}

var<workgroup> kept : atomic<u32>;
var<workgroup> keptBase : u32;

@compute @workgroup_size(64)
fn gather(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
    if (lane == 0u) {
        atomicStore(&kept, 0u);
    }
    workgroupBarrier();
    let slot = wid.z * 65535u + wid.y;
    let t = wid.x * 64u + lane;
    var keep = false;
    var v0 = vec3f(0.0);
    var v1 = vec3f(0.0);
    var v2 = vec3f(0.0);
    var uv = vec3u(0u);
    if (slot < recordCount() && t < src.triangles) {
        let record = src.firstRecord + slot;
        let i = mesh[1] + t * 3u;
        let a = mesh[i];
        let b = mesh[i + 1u];
        let c = mesh[i + 2u];
        v0 = (src.world * vec4f(kansei_rt_place(record, meshPosition(a)), 1.0)).xyz;
        v1 = (src.world * vec4f(kansei_rt_place(record, meshPosition(b)), 1.0)).xyz;
        v2 = (src.world * vec4f(kansei_rt_place(record, meshPosition(c)), 1.0)).xyz;
        uv = vec3u(meshUv(a), meshUv(b), meshUv(c));
        let lo = min(min(v0, v1), v2);
        let hi = max(max(v0, v1), v2);
        let gmin = grid.origin - grid.epsilon;
        let gmax = grid.origin + vec3f(grid.dims) * grid.cell + grid.epsilon;
        let n = cross(v1 - v0, v2 - v0);
        keep = all(hi >= gmin) && all(lo <= gmax) && dot(n, n) > 1e-30;
    }
    var local = 0u;
    if (keep) {
        local = atomicAdd(&kept, 1u);
    }
    workgroupBarrier();
    if (lane == 0u) {
        keptBase = atomicAdd(&counters[0], atomicLoad(&kept));
    }
    workgroupBarrier();
    let id = keptBase + local;
    if (keep && id < grid.triangleCapacity) {
        triangles[id * 4u] = vec4f(v0, bitcast<f32>(src.surface));
        triangles[id * 4u + 1u] = vec4f(v1 - v0, bitcast<f32>(src.albedo));
        triangles[id * 4u + 2u] = vec4f(v2 - v0, bitcast<f32>(src.source | min(slot, 0xfffffu)));
        triangles[id * 4u + 3u] = vec4f(bitcast<f32>(uv.x), bitcast<f32>(uv.y), bitcast<f32>(uv.z), 0.0);
    }
}
