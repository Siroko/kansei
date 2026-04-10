// ── GPU Radix Sort (4-bit, 8 passes for 32-bit keys) ─────────────────────────
//
// Three entry points: histogram, prefix_sum, scatter
// Used to sort TLAS instances by Morton code.

struct SortParams {
    count          : u32,
    bit_offset     : u32,   // 0, 4, 8, 12, 16, 20, 24, 28
    workgroup_count: u32,
    _pad           : u32,
}

@group(0) @binding(0) var<storage, read>       keys_in    : array<u32>;
@group(0) @binding(1) var<storage, read>       vals_in    : array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out   : array<u32>;
@group(0) @binding(3) var<storage, read_write> vals_out   : array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms : array<u32>;
@group(0) @binding(5) var<uniform>             params     : SortParams;

const WG_SIZE = 256u;
const RADIX   = 16u;

// ── Pass 1: Per-workgroup histogram ──────────────────────────────────────────

var<workgroup> local_hist: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn histogram(
    @builtin(global_invocation_id) gid  : vec3u,
    @builtin(workgroup_id)         wg_id: vec3u,
    @builtin(local_invocation_id)  lid  : vec3u,
) {
    if (lid.x < RADIX) {
        atomicStore(&local_hist[lid.x], 0u);
    }
    workgroupBarrier();

    let idx = gid.x;
    if (idx < params.count) {
        let key   = keys_in[idx];
        let digit = (key >> params.bit_offset) & 0xFu;
        atomicAdd(&local_hist[digit], 1u);
    }
    workgroupBarrier();

    if (lid.x < RADIX) {
        histograms[lid.x * params.workgroup_count + wg_id.x] = atomicLoad(&local_hist[lid.x]);
    }
}

// ── Pass 2: Blelloch exclusive prefix sum (single workgroup) ─────────────────

var<workgroup> prefix_temp: array<u32, 4096>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(global_invocation_id) gid: vec3u) {
    let total_bins = RADIX * params.workgroup_count;
    let idx = gid.x;

    // Load
    if (idx < total_bins) {
        prefix_temp[idx] = histograms[idx];
    } else {
        prefix_temp[idx] = 0u;
    }
    workgroupBarrier();

    // Up-sweep (reduce)
    var offset = 1u;
    var d = total_bins >> 1u;
    while (d > 0u) {
        if (idx < d) {
            let ai = offset * (2u * idx + 1u) - 1u;
            let bi = offset * (2u * idx + 2u) - 1u;
            if (bi < total_bins) {
                prefix_temp[bi] += prefix_temp[ai];
            }
        }
        offset <<= 1u;
        d >>= 1u;
        workgroupBarrier();
    }

    // Clear last element
    if (idx == 0u) {
        prefix_temp[total_bins - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep
    d = 1u;
    while (d < total_bins) {
        offset >>= 1u;
        if (idx < d) {
            let ai = offset * (2u * idx + 1u) - 1u;
            let bi = offset * (2u * idx + 2u) - 1u;
            if (bi < total_bins) {
                let temp = prefix_temp[ai];
                prefix_temp[ai] = prefix_temp[bi];
                prefix_temp[bi] += temp;
            }
        }
        d <<= 1u;
        workgroupBarrier();
    }

    // Store
    if (idx < total_bins) {
        histograms[idx] = prefix_temp[idx];
    }
}

// ── Pass 3: Scatter elements to sorted positions ─────────────────────────────

var<workgroup> scatter_hist: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) gid  : vec3u,
    @builtin(workgroup_id)         wg_id: vec3u,
    @builtin(local_invocation_id)  lid  : vec3u,
) {
    // Load prefix sums for this workgroup
    if (lid.x < RADIX) {
        atomicStore(&scatter_hist[lid.x], histograms[lid.x * params.workgroup_count + wg_id.x]);
    }
    workgroupBarrier();

    let idx = gid.x;
    if (idx < params.count) {
        let key   = keys_in[idx];
        let val   = vals_in[idx];
        let digit = (key >> params.bit_offset) & 0xFu;
        let dest  = atomicAdd(&scatter_hist[digit], 1u);
        keys_out[dest] = key;
        vals_out[dest] = val;
    }
}
