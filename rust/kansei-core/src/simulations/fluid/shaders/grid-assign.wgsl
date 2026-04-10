@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    let coord = getCellCoord(pos, params);
    let cell = cellHash(coord, params);

    cellIndices[idx] = cell;
    atomicAdd(&cellCounts[cell], 1u);
}
