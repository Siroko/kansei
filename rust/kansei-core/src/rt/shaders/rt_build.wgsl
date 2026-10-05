// The grid's build (rt/grid.rs), from the triangles gathered this rebuild. `prepare` sizes the
// indirect dispatches from the gathered count. `count` (a thread a triangle) puts a triangle
// whose footprint is too wide for the cells in the big list, while it has room, one too wide for
// a thread in the wide list, and counts the cells every other one overlaps; `count_wide` counts
// the wide ones' cells, a workgroup a triangle (`prepare_wide` sizes it). `scan_*` turns the
// counts into each cell's start (an exclusive prefix sum of at most 1024 x 1024 cells); `fill`
// and `fill_wide` list each triangle in its cells (each cell's word then holds its end) and mark
// their macro cells. Included after rt_types.wgsl.
//
// A triangle visits the cells near its plane: over its box's columns along the plane's dominant
// axis, only the cells the plane crosses in each column, each confirmed by the separating axes of
// a triangle and a box (Akenine-Moller 2001), every cell widened by `grid.epsilon` so a triangle
// on a cell's face is listed on both sides of it.

@group(0) @binding(0) var<uniform> grid : KanseiRtGrid;
@group(0) @binding(1) var<storage, read_write> triangles : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cells : array<atomic<u32>>;
// [0] triangles claimed, [1] references needed, [2] wide triangles, [16..1040) the scan's block
// sums, [1056..) the wide triangles
@group(0) @binding(3) var<storage, read_write> counters : array<atomic<u32>>;
// the triangles' dispatch, then the wide ones'
@group(0) @binding(4) var<storage, read_write> dispatch : array<u32, 8>;

const SUMS : u32 = 16u;
const WIDE : u32 = 1056u;

fn triangleCount() -> u32 {
    return min(atomicLoad(&counters[0]), grid.triangleCapacity);
}

// x and y of a dispatch of `groups` workgroups
fn spread(groups: u32) -> vec2u {
    let x = min(groups, 65535u);
    return vec2u(x, select(0u, (groups + x - 1u) / max(x, 1u), x > 0u));
}

@compute @workgroup_size(1)
fn prepare() {
    let g = spread((triangleCount() + 63u) / 64u);
    dispatch[0] = g.x;
    dispatch[1] = g.y;
    dispatch[2] = 1u;
}

@compute @workgroup_size(1)
fn prepare_wide() {
    let g = spread(atomicLoad(&counters[2]));
    dispatch[4] = g.x;
    dispatch[5] = g.y;
    dispatch[6] = 1u;
}

struct Tri {
    v0 : vec3f,
    v1 : vec3f,
    v2 : vec3f,
}

fn triangleAt(id: u32) -> Tri {
    let a = triangles[id * 4u].xyz;
    return Tri(a, a + triangles[id * 4u + 1u].xyz, a + triangles[id * 4u + 2u].xyz);
}

fn cellIndex(c: vec3i) -> u32 {
    let d = vec3i(grid.dims);
    return u32((c.z * d.y + c.y) * d.x + c.x);
}

fn macroIndex(c: vec3i) -> u32 {
    let m = vec3u(c) >> vec3u(KANSEI_RT_MACRO_SHIFT);
    return (m.z * grid.macroDims.y + m.y) * grid.macroDims.x + m.x;
}

// Whether the triangle's projections on `axis` miss a box of half-size h round the origin.
fn separated(axis: vec3f, v0: vec3f, v1: vec3f, v2: vec3f, h: f32) -> bool {
    let p = vec3f(dot(axis, v0), dot(axis, v1), dot(axis, v2));
    let r = h * (abs(axis.x) + abs(axis.y) + abs(axis.z));
    return min(min(p.x, p.y), p.z) > r || max(max(p.x, p.y), p.z) < -r;
}

// Whether the triangle (normal n) overlaps cell c widened by epsilon, c already within its box's
// cells: its plane passes through the cell and no edge-cross axis separates them.
fn overlaps(t: Tri, n: vec3f, c: vec3i) -> bool {
    let h = 0.5 * grid.cell + grid.epsilon;
    let centre = grid.origin + (vec3f(c) + 0.5) * grid.cell;
    if (abs(dot(n, centre - t.v0)) > h * (abs(n.x) + abs(n.y) + abs(n.z))) {
        return false;
    }
    let v0 = t.v0 - centre;
    let v1 = t.v1 - centre;
    let v2 = t.v2 - centre;
    let edges = array<vec3f, 3>(v1 - v0, v2 - v1, v0 - v2);
    for (var k = 0; k < 3; k++) {
        let e = edges[k];
        if (separated(vec3f(0.0, -e.z, e.y), v0, v1, v2, h)) { return false; }
        if (separated(vec3f(e.z, 0.0, -e.x), v0, v1, v2, h)) { return false; }
        if (separated(vec3f(-e.y, e.x, 0.0), v0, v1, v2, h)) { return false; }
    }
    return true;
}

struct CellRange {
    lo : vec3i,
    hi : vec3i,
}

// The cells the triangle's box covers (widened by epsilon), clamped to the grid; lo > hi on an
// axis when it misses the grid.
fn cellRange(t: Tri) -> CellRange {
    let lo = (min(min(t.v0, t.v1), t.v2) - grid.epsilon - grid.origin) / grid.cell;
    let hi = (max(max(t.v0, t.v1), t.v2) + grid.epsilon - grid.origin) / grid.cell;
    return CellRange(max(vec3i(floor(lo)), vec3i(0)), min(vec3i(floor(hi)), vec3i(grid.dims) - 1));
}

// The plane's dominant axis k (the columns run along it) and the other two.
fn axes(n: vec3f) -> vec3i {
    let a = abs(n);
    var k = 2;
    if (a.x >= a.y && a.x >= a.z) { k = 0; } else if (a.y >= a.z) { k = 1; }
    return vec3i((k + 1) % 3, (k + 2) % 3, k);
}

// Columns of the triangle's footprint across its dominant axis.
fn columns(r: CellRange, ax: vec3i) -> u32 {
    return u32(max(r.hi[ax.x] - r.lo[ax.x] + 1, 0)) * u32(max(r.hi[ax.y] - r.lo[ax.y] + 1, 0));
}

// Count (fill false) or list (fill true) triangle `id` in every cell it overlaps, in columns
// `first`, `first + step`, ... of its footprint.
fn scatter(id: u32, t: Tri, r: CellRange, ax: vec3i, fill: bool, first: u32, step: u32) {
    let n = cross(t.v1 - t.v0, t.v2 - t.v0);
    let i = ax.x;
    let j = ax.y;
    let k = ax.z;
    let d = dot(n, t.v0);
    let o = grid.origin;
    let e = grid.epsilon;
    let across = u32(r.hi[i] - r.lo[i] + 1);
    let total = columns(r, ax);
    for (var q = first; q < total; q += step) {
        let ci = r.lo[i] + i32(q % across);
        let cj = r.lo[j] + i32(q / across);
        // the plane's extent along k over this column's footprint, widened
        var lo = 1e30;
        var hi = -1e30;
        for (var corner = 0; corner < 4; corner++) {
            let si = corner & 1;
            let sj = corner >> 1u;
            let xi = o[i] + f32(ci + si) * grid.cell + select(-e, e, si == 1);
            let xj = o[j] + f32(cj + sj) * grid.cell + select(-e, e, sj == 1);
            let xk = (d - n[i] * xi - n[j] * xj) / n[k];
            lo = min(lo, xk);
            hi = max(hi, xk);
        }
        let k0 = max(i32(floor((lo - e - o[k]) / grid.cell)), r.lo[k]);
        let k1 = min(i32(floor((hi + e - o[k]) / grid.cell)), r.hi[k]);
        for (var ck = k0; ck <= k1; ck++) {
            var c = vec3i(0);
            c[i] = ci;
            c[j] = cj;
            c[k] = ck;
            if (!overlaps(t, n, c)) {
                continue;
            }
            let cell = cellIndex(c);
            if (fill) {
                let slot = atomicAdd(&cells[cell], 1u);
                if (slot < grid.refCapacity) {
                    atomicStore(&cells[grid.refsBase + slot], id);
                }
                atomicStore(&cells[grid.macroBase + macroIndex(c)], 1u);
            } else {
                atomicAdd(&cells[cell], 1u);
            }
        }
    }
}

fn threadTriangle(wid: vec3u, groups: vec3u, lane: u32) -> u32 {
    return (wid.y * groups.x + wid.x) * 64u + lane;
}

// A triangle spanning more columns than this is scattered by a workgroup of its own, a column a
// thread in turn, not by one thread: a wall at 5 cm cells would keep one walking thousands.
const WIDE_COLUMNS : u32 = 32u;

@compute @workgroup_size(64)
fn count(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lane: u32) {
    let id = threadTriangle(wid, groups, lane);
    if (id >= triangleCount()) {
        return;
    }
    let t = triangleAt(id);
    let r = cellRange(t);
    if (any(r.lo > r.hi)) {
        return;
    }
    let ax = axes(cross(t.v1 - t.v0, t.v2 - t.v0));
    let cols = columns(r, ax);
    if (cols > grid.bigCells) {
        let slot = atomicAdd(&cells[grid.bigBase], 1u);
        if (slot < grid.bigCapacity) {
            atomicStore(&cells[grid.bigBase + 1u + slot], id);
            triangles[id * 4u].w = bitcast<f32>(bitcast<u32>(triangles[id * 4u].w) | KANSEI_RT_BIG);
            return;
        }
    }
    if (cols > WIDE_COLUMNS) {
        atomicStore(&counters[WIDE + atomicAdd(&counters[2], 1u)], id);
        return;
    }
    scatter(id, t, r, ax, false, 0u, 1u);
}

@compute @workgroup_size(64)
fn fill(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lane: u32) {
    let id = threadTriangle(wid, groups, lane);
    if (id >= triangleCount() || (bitcast<u32>(triangles[id * 4u].w) & KANSEI_RT_BIG) != 0u) {
        return;
    }
    let t = triangleAt(id);
    let r = cellRange(t);
    if (any(r.lo > r.hi)) {
        return;
    }
    let ax = axes(cross(t.v1 - t.v0, t.v2 - t.v0));
    if (columns(r, ax) <= WIDE_COLUMNS) {
        scatter(id, t, r, ax, true, 0u, 1u);
    }
}

// Wide triangle number (y * x + x of the workgroup), its 64 threads a column each in turn.
fn wideScatter(wid: vec3u, groups: vec3u, lane: u32, fill: bool) {
    let w = wid.y * groups.x + wid.x;
    if (w >= atomicLoad(&counters[2])) {
        return;
    }
    let id = atomicLoad(&counters[WIDE + w]);
    let t = triangleAt(id);
    scatter(id, t, cellRange(t), axes(cross(t.v1 - t.v0, t.v2 - t.v0)), fill, lane, 64u);
}

@compute @workgroup_size(64)
fn count_wide(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lane: u32) {
    wideScatter(wid, groups, lane, false);
}

@compute @workgroup_size(64)
fn fill_wide(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lane: u32) {
    wideScatter(wid, groups, lane, true);
}

// ---- the exclusive prefix sum of the counts: blocks of 1024 (256 threads, 4 each), the block
// sums (one workgroup), the sums added back ----

var<workgroup> partial : array<u32, 256>;

// Scans `partial` in place (inclusive, Hillis-Steele) and returns this thread's inclusive sum.
fn scanWorkgroup(lane: u32, value: u32) -> u32 {
    partial[lane] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var add = 0u;
        if (lane >= offset) {
            add = partial[lane - offset];
        }
        workgroupBarrier();
        partial[lane] += add;
        workgroupBarrier();
    }
    return partial[lane];
}

@compute @workgroup_size(256)
fn scan_blocks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lane: u32) {
    let total = grid.cellCount;
    let base = wid.x * 1024u + lane * 4u;
    var v : array<u32, 4>;
    var sum = 0u;
    for (var k = 0u; k < 4u; k++) {
        let i = base + k;
        if (i < total) {
            v[k] = atomicLoad(&cells[i]);
        }
        sum += v[k];
    }
    let inclusive = scanWorkgroup(lane, sum);
    var running = inclusive - sum;
    for (var k = 0u; k < 4u; k++) {
        let i = base + k;
        if (i < total) {
            atomicStore(&cells[i], running);
        }
        running += v[k];
    }
    if (lane == 255u) {
        atomicStore(&counters[SUMS + wid.x], inclusive);
    }
}

@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_index) lane: u32) {
    let blocks = (grid.cellCount + 1023u) / 1024u;
    var v : array<u32, 4>;
    var sum = 0u;
    for (var k = 0u; k < 4u; k++) {
        let i = lane * 4u + k;
        if (i < blocks) {
            v[k] = atomicLoad(&counters[SUMS + i]);
        }
        sum += v[k];
    }
    let inclusive = scanWorkgroup(lane, sum);
    var running = inclusive - sum;
    for (var k = 0u; k < 4u; k++) {
        let i = lane * 4u + k;
        if (i < blocks) {
            atomicStore(&counters[SUMS + i], running);
        }
        running += v[k];
    }
    // the references every cell needs
    if (lane == 255u) {
        atomicStore(&counters[1], inclusive);
    }
}

@compute @workgroup_size(256)
fn scan_add(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= grid.cellCount) {
        return;
    }
    atomicAdd(&cells[i], atomicLoad(&counters[SUMS + i / 1024u]));
}
