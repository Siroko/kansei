// Gathers a source's world triangles into the grid's (rt/grid.rs). A thread places a triangle
// (`kansei_rt_place`, prepended, puts a mesh point where a record puts it, then `src.world`),
// drops it when its box misses the grid's (widened by epsilon) or it has no area, and the
// workgroup's survivors take their slots with one atomic. Included after rt_types.wgsl.
//
// - `gather`: a mesh (`RtMesh` words: where its vertices and indices start, how many vertices and
//   triangles it has, then the vertices, 4 words each, and the indices) placed once per record:
//   workgroups of (64 triangles, record), the records past 65535 in z.
// - `gather_clusters`: a cluster LOD cut (cluster_mesh.wgsl's words, `ClusterMesh::gpu_words`):
//   a workgroup per entry of the cut's draw list (record, cluster), a thread for each of the
//   cluster's first 128 triangles and the next 128.
// - `prepare`: a source's indirect dispatch, from the records (or entries) it has, which the
//   GPU may have counted (a culled view's draw, a cut's claimed clusters).

struct RtSource {
    world       : mat4x4f,
    // the mesh's triangles (`gather`)
    triangles   : u32,
    strideWords : u32,
    firstRecord : u32,
    // the records (or draw-list entries) at most
    records     : u32,
    // KANSEI_RT_NO_WORD: `records` of them; else the word of `args` that holds how many
    countWord   : u32,
    surface     : u32,
    albedo      : u32,
    // the source's id, shifted to bits 20 and up
    source      : u32,
    // its indirect dispatch's slot in `dispatchOut`
    slot        : u32,
    // 0: a mesh (`gather`), 1: a cut (`gather_clusters`)
    kind        : u32,
    pad0        : u32,
    pad1        : u32,
}

const KANSEI_RT_NO_WORD : u32 = 0xffffffffu;
const KANSEI_CLUSTER_WORDS : u32 = 28u;

@group(0) @binding(0) var<uniform> grid : KanseiRtGrid;
@group(0) @binding(1) var<storage, read_write> triangles : array<vec4f>;
// [0]: triangles claimed (more than the capacity when it overflowed)
@group(0) @binding(2) var<storage, read_write> counters : array<atomic<u32>>;
// `prepare`: every source's dispatch, 4 words each
@group(0) @binding(5) var<storage, read_write> dispatchOut : array<u32>;
@group(1) @binding(0) var<uniform> src : RtSource;
@group(1) @binding(1) var<storage, read> mesh : array<u32>;
@group(1) @binding(2) var<storage, read> records : array<u32>;
@group(1) @binding(3) var<storage, read> args : array<u32>;
@group(1) @binding(4) var<storage, read> draws : array<vec2u>;

fn kansei_rt_record_f32(record: u32, word: u32) -> f32 {
    return bitcast<f32>(records[record * src.strideWords + word]);
}

fn kansei_rt_record_vec3(record: u32, word: u32) -> vec3f {
    return vec3f(kansei_rt_record_f32(record, word), kansei_rt_record_f32(record, word + 1u), kansei_rt_record_f32(record, word + 2u));
}

fn kansei_rt_record_vec4(record: u32, word: u32) -> vec4f {
    return vec4f(kansei_rt_record_vec3(record, word), kansei_rt_record_f32(record, word + 3u));
}

fn recordCount() -> u32 {
    if (src.countWord == KANSEI_RT_NO_WORD) {
        return src.records;
    }
    return min(args[src.countWord], src.records);
}

@compute @workgroup_size(1)
fn prepare() {
    let n = recordCount();
    var x = 1u;
    if (src.kind == 0u) {
        x = (src.triangles + 63u) / 64u;
    }
    let at = src.slot * 4u;
    dispatchOut[at] = select(0u, x, n > 0u);
    dispatchOut[at + 1u] = min(n, 65535u);
    dispatchOut[at + 2u] = (n + 65534u) / 65535u;
}

struct Placed {
    v0 : vec3f,
    v1 : vec3f,
    v2 : vec3f,
    // whether it meets the grid's box and has an area
    keep : bool,
}

fn place(record: u32, a: vec3f, b: vec3f, c: vec3f) -> Placed {
    var p : Placed;
    p.v0 = (src.world * vec4f(kansei_rt_place(record, a), 1.0)).xyz;
    p.v1 = (src.world * vec4f(kansei_rt_place(record, b), 1.0)).xyz;
    p.v2 = (src.world * vec4f(kansei_rt_place(record, c), 1.0)).xyz;
    let lo = min(min(p.v0, p.v1), p.v2);
    let hi = max(max(p.v0, p.v1), p.v2);
    let gmin = grid.origin - grid.epsilon;
    let gmax = grid.origin + vec3f(grid.dims) * grid.cell + grid.epsilon;
    let n = cross(p.v1 - p.v0, p.v2 - p.v0);
    p.keep = all(hi >= gmin) && all(lo <= gmax) && dot(n, n) > 1e-30;
    return p;
}

// A world normal of the vertex at object position `a` with object normal `n`, as the record's
// placement and the source's world matrix carry it (rigid or uniformly scaled), packed.
fn placeNormal(record: u32, a: vec3f, n: vec3f) -> u32 {
    let p0 = (src.world * vec4f(kansei_rt_place(record, a), 1.0)).xyz;
    let p1 = (src.world * vec4f(kansei_rt_place(record, a + n * 0.01), 1.0)).xyz;
    return kansei_rt_pack_normal(normalize(p1 - p0));
}

// What a triangle's last four words carry: its vertices' uvs, or with KANSEI_RT_SMOOTH their
// world normals.
fn extra(record: u32, a: vec3f, b: vec3f, c: vec3f, na: vec3f, nb: vec3f, nc: vec3f, uv: vec3u) -> vec3u {
    if ((src.surface & KANSEI_RT_SMOOTH) == 0u) {
        return uv;
    }
    return vec3u(placeNormal(record, a, na), placeNormal(record, b, nb), placeNormal(record, c, nc));
}

fn write(id: u32, p: Placed, uv: vec3u, record: u32) {
    if (id >= grid.triangleCapacity) {
        return;
    }
    triangles[id * 4u] = vec4f(p.v0, bitcast<f32>(src.surface));
    triangles[id * 4u + 1u] = vec4f(p.v1 - p.v0, bitcast<f32>(src.albedo));
    triangles[id * 4u + 2u] = vec4f(p.v2 - p.v0, bitcast<f32>(src.source | min(record, 0xfffffu)));
    triangles[id * 4u + 3u] = vec4f(bitcast<f32>(uv.x), bitcast<f32>(uv.y), bitcast<f32>(uv.z), 0.0);
}

var<workgroup> kept : atomic<u32>;
var<workgroup> keptBase : u32;

// ---- a mesh, once per record ----

// Mesh vertex `v` (`RtMesh`: x, y, z, the uv packed as two f16, the normal octahedral).
fn meshPosition(v: u32) -> vec3f {
    let at = mesh[0] + v * 5u;
    return vec3f(bitcast<f32>(mesh[at]), bitcast<f32>(mesh[at + 1u]), bitcast<f32>(mesh[at + 2u]));
}

fn meshUv(v: u32) -> u32 {
    return mesh[mesh[0] + v * 5u + 3u];
}

fn meshNormal(v: u32) -> vec3f {
    return kansei_rt_unpack_normal(mesh[mesh[0] + v * 5u + 4u]);
}

@compute @workgroup_size(64)
fn gather(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
    if (lane == 0u) {
        atomicStore(&kept, 0u);
    }
    workgroupBarrier();
    let slot = wid.z * 65535u + wid.y;
    let t = wid.x * 64u + lane;
    var p : Placed;
    var uv = vec3u(0u);
    if (slot < recordCount() && t < src.triangles) {
        let i = mesh[1] + t * 3u;
        let a = mesh[i];
        let b = mesh[i + 1u];
        let c = mesh[i + 2u];
        p = place(src.firstRecord + slot, meshPosition(a), meshPosition(b), meshPosition(c));
        uv = extra(src.firstRecord + slot, meshPosition(a), meshPosition(b), meshPosition(c), meshNormal(a), meshNormal(b), meshNormal(c), vec3u(meshUv(a), meshUv(b), meshUv(c)));
    }
    var local = 0u;
    if (p.keep) {
        local = atomicAdd(&kept, 1u);
    }
    workgroupBarrier();
    if (lane == 0u) {
        keptBase = atomicAdd(&counters[0], atomicLoad(&kept));
    }
    workgroupBarrier();
    if (p.keep) {
        write(keptBase + local, p, uv, slot);
    }
}

// ---- a cluster LOD cut: a workgroup per draw-list entry ----

fn clusterWord(c: u32, word: u32) -> u32 {
    return mesh[mesh[3] + c * KANSEI_CLUSTER_WORDS + word];
}

// The mesh vertex that is vertex `local` of cluster `c`.
fn clusterVertex(c: u32, local: u32) -> u32 {
    return mesh[mesh[1] + clusterWord(c, 0u) + local];
}

// A mesh vertex (`Vertex`: position 0-3, normal 4-6, uv 7-8).
fn vertexPosition(v: u32) -> vec3f {
    let at = mesh[0] + v * 9u;
    return vec3f(bitcast<f32>(mesh[at]), bitcast<f32>(mesh[at + 1u]), bitcast<f32>(mesh[at + 2u]));
}

fn vertexNormal(v: u32) -> vec3f {
    let at = mesh[0] + v * 9u + 4u;
    return vec3f(bitcast<f32>(mesh[at]), bitcast<f32>(mesh[at + 1u]), bitcast<f32>(mesh[at + 2u]));
}

fn vertexUv(v: u32) -> u32 {
    let at = mesh[0] + v * 9u + 7u;
    return pack2x16float(vec2f(bitcast<f32>(mesh[at]), bitcast<f32>(mesh[at + 1u])));
}

@compute @workgroup_size(128)
fn gather_clusters(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
    if (lane == 0u) {
        atomicStore(&kept, 0u);
    }
    workgroupBarrier();
    let entry = wid.z * 65535u + wid.y;
    var placed : array<Placed, 2>;
    var uvs : array<vec3u, 2>;
    var record = 0u;
    var count = 0u;
    if (entry < recordCount()) {
        let e = draws[entry];
        // (an entry whose triangles found no room in the cut's index buffer)
        if (e.x != KANSEI_RT_NO_WORD) {
            record = e.x;
            let c = e.y;
            let first = mesh[2] + clusterWord(c, 1u);
            let n = clusterWord(c, 2u);
            for (var k = 0u; k < 2u; k++) {
                let t = lane + k * 128u;
                if (t < n) {
                    let packed = mesh[first + t];
                    let a = clusterVertex(c, packed & 255u);
                    let b = clusterVertex(c, (packed >> 8u) & 255u);
                    let d = clusterVertex(c, (packed >> 16u) & 255u);
                    let p = place(record, vertexPosition(a), vertexPosition(b), vertexPosition(d));
                    if (p.keep) {
                        placed[count] = p;
                        uvs[count] = extra(record, vertexPosition(a), vertexPosition(b), vertexPosition(d), vertexNormal(a), vertexNormal(b), vertexNormal(d), vec3u(vertexUv(a), vertexUv(b), vertexUv(d)));
                        count += 1u;
                    }
                }
            }
        }
    }
    var local = 0u;
    if (count > 0u) {
        local = atomicAdd(&kept, count);
    }
    workgroupBarrier();
    if (lane == 0u) {
        keptBase = atomicAdd(&counters[0], atomicLoad(&kept));
    }
    workgroupBarrier();
    for (var k = 0u; k < count; k++) {
        write(keptBase + local + k, placed[k], uvs[k], record);
    }
}
