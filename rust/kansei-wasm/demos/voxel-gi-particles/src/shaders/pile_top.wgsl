// The highest particle centre, as the bits of a non-negative f32 (which order as u32 do), from the
// positions the neighbour grid sorted: rays above it plus the largest radius meet no particle
// (room_spheres.wgsl). `top` is cleared to 0 before.
@group(0) @binding(0) var<uniform> grid: NeighbourGrid;
@group(0) @binding(1) var<storage, read> sortedPositions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> top: atomic<u32>;

var<workgroup> highest: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) local: u32) {
    if (gid.x < grid.count) {
        atomicMax(&highest, bitcast<u32>(max(sortedPositions[gid.x].y, 0.0)));
    }
    workgroupBarrier();
    if (local == 0u) {
        atomicMax(&top, atomicLoad(&highest));
    }
}
