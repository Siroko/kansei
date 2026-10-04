// A neighbour grid (`simulations::grid::NeighbourGrid`): `dims` cells `cellSize` wide from
// `origin`, holding the first `count` points. Cell (x, y, z) has index x + dims.x * (y + dims.y * z).
// After the grid's passes, the points in cell c are the sorted slots cellOffsets[c] up to
// cellOffsets[c] + cellCounts[c] (or up to the next cell's offset, the last cell's up to
// `count`), and sortedIndices maps a slot to its point.
struct NeighbourGrid {
    origin: vec3<f32>,
    cellSize: f32,
    dims: vec3<u32>,
    count: u32,
};

// The cell holding `p`, which may lie outside the grid.
fn neighbourCell(p: vec3<f32>, grid: NeighbourGrid) -> vec3<i32> {
    return vec3<i32>(floor((p - grid.origin) / grid.cellSize));
}

fn neighbourCellInside(cell: vec3<i32>, grid: NeighbourGrid) -> bool {
    return all(cell >= vec3<i32>(0)) && all(cell < vec3<i32>(grid.dims));
}

// The index of `cell` clamped into the grid: points outside it go into its edge cells.
fn neighbourCellIndex(cell: vec3<i32>, grid: NeighbourGrid) -> u32 {
    let c = vec3<u32>(clamp(cell, vec3<i32>(0), vec3<i32>(grid.dims) - vec3<i32>(1)));
    return c.x + grid.dims.x * (c.y + grid.dims.y * c.z);
}
