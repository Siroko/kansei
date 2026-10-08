// GPU radix sort (4-bit digits, 8 passes for 32-bit keys), ported from
// rust/kansei-core/src/pathtracer/shaders/radix-sort.wgsl.
// Three entry points: histogram, prefix_sum, scatter.
export const radixSortShader = /* wgsl */`
struct SortParams {
    count     : u32,
    bitOffset : u32,
    workgroupCount : u32,
    _pad      : u32,
}

@group(0) @binding(0) var<storage, read>       keysIn     : array<u32>;
@group(0) @binding(1) var<storage, read>       valsIn     : array<u32>;
@group(0) @binding(2) var<storage, read_write> keysOut    : array<u32>;
@group(0) @binding(3) var<storage, read_write> valsOut    : array<u32>;
@group(0) @binding(4) var<storage, read_write> histograms : array<u32>;
@group(0) @binding(5) var<uniform>             params     : SortParams;

const WG_SIZE = 256u;
const RADIX = 16u;

var<workgroup> localHist: array<atomic<u32>, 16>;

// Pass 1: Per-workgroup histogram
@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wgId: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    if (lid.x < RADIX) { atomicStore(&localHist[lid.x], 0u); }
    workgroupBarrier();

    let idx = gid.x;
    if (idx < params.count) {
        let key = keysIn[idx];
        let digit = (key >> params.bitOffset) & 0xFu;
        atomicAdd(&localHist[digit], 1u);
    }
    workgroupBarrier();

    if (lid.x < RADIX) {
        histograms[lid.x * params.workgroupCount + wgId.x] = atomicLoad(&localHist[lid.x]);
    }
}

// Pass 2: Exclusive prefix sum over all digit x workgroup bins.
// The bins are digit-major, so the scan gives every (digit, workgroup) pair its
// first output slot. A serial scan is enough for the TLAS (16 bins per 256
// instances) and handles any bin count.
@compute @workgroup_size(1)
fn prefix_sum() {
    let totalBins = RADIX * params.workgroupCount;
    var sum = 0u;
    for (var i = 0u; i < totalBins; i++) {
        let c = histograms[i];
        histograms[i] = sum;
        sum += c;
    }
}

// Pass 3: Scatter elements to sorted positions.
// Stable: each element's slot within its digit is the number of earlier
// elements in its workgroup with the same digit, so every pass keeps the order
// the previous passes established (LSD radix sort depends on it).
var<workgroup> scatterDigits: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wgId: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let idx = gid.x;
    var digit = RADIX; // matches no real digit
    if (idx < params.count) {
        digit = (keysIn[idx] >> params.bitOffset) & 0xFu;
    }
    scatterDigits[lid.x] = digit;
    workgroupBarrier();

    if (idx < params.count) {
        var rank = 0u;
        for (var j = 0u; j < lid.x; j++) {
            rank += select(0u, 1u, scatterDigits[j] == digit);
        }
        let dest = histograms[digit * params.workgroupCount + wgId.x] + rank;
        keysOut[dest] = keysIn[idx];
        valsOut[dest] = valsIn[idx];
    }
}
`;
