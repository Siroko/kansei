export const shaderCode = /* wgsl */`
// Exclusive scan of the per-block sums, run as ONE workgroup. Handles any number of
// blocks by scanning 512-entry chunks in sequence and carrying the running total, so
// the grid is no longer capped at 512 blocks × 512 cells.
const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> blockSums: array<u32>;

var<workgroup> sdata: array<u32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let len = arrayLength(&blockSums);
    let chunks = (len + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    var running = 0u; // identical in every thread (derived from uniform values only)

    for (var c = 0u; c < chunks; c++) {
        let base = c * BLOCK_SIZE;
        let ai = tid;
        let bi = tid + 256u;
        let ga = base + ai;
        let gb = base + bi;
        sdata[ai] = select(0u, blockSums[ga], ga < len);
        sdata[bi] = select(0u, blockSums[gb], gb < len);

        var offset = 1u;
        for (var d = BLOCK_SIZE >> 1u; d > 0u; d >>= 1u) {
            workgroupBarrier();
            if (tid < d) {
                let ai2 = offset * (2u * tid + 1u) - 1u;
                let bi2 = offset * (2u * tid + 2u) - 1u;
                sdata[bi2] += sdata[ai2];
            }
            offset <<= 1u;
        }

        workgroupBarrier();
        let chunkTotal = sdata[BLOCK_SIZE - 1u];
        workgroupBarrier();
        if (tid == 0u) {
            sdata[BLOCK_SIZE - 1u] = 0u;
        }

        for (var d2 = 1u; d2 < BLOCK_SIZE; d2 <<= 1u) {
            offset >>= 1u;
            workgroupBarrier();
            if (tid < d2) {
                let ai3 = offset * (2u * tid + 1u) - 1u;
                let bi3 = offset * (2u * tid + 2u) - 1u;
                let temp = sdata[ai3];
                sdata[ai3] = sdata[bi3];
                sdata[bi3] += temp;
            }
        }

        workgroupBarrier();
        if (ga < len) { blockSums[ga] = sdata[ai] + running; }
        if (gb < len) { blockSums[gb] = sdata[bi] + running; }
        running += chunkTotal;
        workgroupBarrier();
    }
}
`;
