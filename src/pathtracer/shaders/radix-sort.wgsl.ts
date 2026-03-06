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

// Pass 2: Prefix sum over histograms (single workgroup)
var<workgroup> prefixTemp: array<u32, 4096>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(global_invocation_id) gid: vec3u) {
    let totalBins = RADIX * params.workgroupCount;
    let idx = gid.x;

    // Load
    if (idx < totalBins) {
        prefixTemp[idx] = histograms[idx];
    } else {
        prefixTemp[idx] = 0u;
    }
    workgroupBarrier();

    // Blelloch scan (up-sweep)
    var offset = 1u;
    var d = totalBins >> 1u;
    while (d > 0u) {
        if (idx < d) {
            let ai = offset * (2u * idx + 1u) - 1u;
            let bi = offset * (2u * idx + 2u) - 1u;
            if (bi < totalBins) {
                prefixTemp[bi] += prefixTemp[ai];
            }
        }
        offset <<= 1u;
        d >>= 1u;
        workgroupBarrier();
    }

    // Clear last
    if (idx == 0u) { prefixTemp[totalBins - 1u] = 0u; }
    workgroupBarrier();

    // Down-sweep
    d = 1u;
    while (d < totalBins) {
        offset >>= 1u;
        if (idx < d) {
            let ai = offset * (2u * idx + 1u) - 1u;
            let bi = offset * (2u * idx + 2u) - 1u;
            if (bi < totalBins) {
                let temp = prefixTemp[ai];
                prefixTemp[ai] = prefixTemp[bi];
                prefixTemp[bi] += temp;
            }
        }
        d <<= 1u;
        workgroupBarrier();
    }

    // Store
    if (idx < totalBins) {
        histograms[idx] = prefixTemp[idx];
    }
}

// Pass 3: Scatter elements to sorted positions
var<workgroup> scatterHist: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wgId: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    // Load local histogram prefix for this workgroup
    if (lid.x < RADIX) {
        atomicStore(&scatterHist[lid.x], histograms[lid.x * params.workgroupCount + wgId.x]);
    }
    workgroupBarrier();

    let idx = gid.x;
    if (idx < params.count) {
        let key = keysIn[idx];
        let val = valsIn[idx];
        let digit = (key >> params.bitOffset) & 0xFu;
        let dest = atomicAdd(&scatterHist[digit], 1u);
        keysOut[dest] = key;
        valsOut[dest] = val;
    }
}
`;
