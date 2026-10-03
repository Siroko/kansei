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

// ── Pass 2: Exclusive prefix sum over all digit × workgroup bins ────────────
//
// The bins are digit-major, so the scan gives every (digit, workgroup) pair its
// first output slot. A serial scan is enough for the TLAS (16 bins per 256
// instances) and handles any bin count.

@compute @workgroup_size(1)
fn prefix_sum() {
    let total_bins = RADIX * params.workgroup_count;
    var sum = 0u;
    for (var i = 0u; i < total_bins; i++) {
        let c = histograms[i];
        histograms[i] = sum;
        sum += c;
    }
}

// ── Pass 3: Scatter elements to sorted positions ─────────────────────────────
//
// Stable: each element's slot within its digit is the number of earlier
// elements in its workgroup with the same digit, so every pass keeps the order
// the previous passes established (LSD radix sort depends on it).

var<workgroup> scatter_digits: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) gid  : vec3u,
    @builtin(workgroup_id)         wg_id: vec3u,
    @builtin(local_invocation_id)  lid  : vec3u,
) {
    let idx = gid.x;
    var digit = RADIX; // matches no real digit
    if (idx < params.count) {
        digit = (keys_in[idx] >> params.bit_offset) & 0xFu;
    }
    scatter_digits[lid.x] = digit;
    workgroupBarrier();

    if (idx < params.count) {
        var rank = 0u;
        for (var j = 0u; j < lid.x; j++) {
            rank += select(0u, 1u, scatter_digits[j] == digit);
        }
        let dest = histograms[digit * params.workgroup_count + wg_id.x] + rank;
        keys_out[dest] = keys_in[idx];
        vals_out[dest] = vals_in[idx];
    }
}
