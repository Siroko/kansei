// Counting-sort scatter. Besides the index permutation, it writes cell-ordered copies of the
// grid's source buffers (bindings from 5 on, substituted for the markers), so passes read
// neighbours contiguously.
@group(0) @binding(0) var<storage, read> cellIndices: array<u32>;
@group(0) @binding(1) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(4) var<uniform> grid: NeighbourGrid;
//__COPY_BINDINGS__

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= grid.count) { return; }

    let cell = cellIndices[idx];
    let slot = cellOffsets[cell] + atomicAdd(&scatterCounters[cell], 1u);
    sortedIndices[slot] = idx;
    //__COPIES__
}
