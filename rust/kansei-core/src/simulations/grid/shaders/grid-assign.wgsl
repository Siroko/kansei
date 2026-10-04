// Each point's cell, and how many points each cell holds.
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> grid: NeighbourGrid;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= grid.count) { return; }

    let cell = neighbourCellIndex(neighbourCell(positions[idx].xyz, grid), grid);
    cellIndices[idx] = cell;
    atomicAdd(&cellCounts[cell], 1u);
}
