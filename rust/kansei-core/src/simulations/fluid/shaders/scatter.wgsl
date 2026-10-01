// Counting-sort scatter. Besides the index permutation, writes cell-ordered copies of
// positions and velocities so the density/forces passes read neighbors contiguously.
@group(0) @binding(0) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;
@group(0) @binding(5) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> sortedVelocities: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let cell = cellIndices[idx];
    let offset = cellOffsets[cell];
    let slot = offset + atomicAdd(&scatterCounters[cell], 1u);
    sortedIndices[slot] = idx;
    sortedPositions[slot] = positions[idx];
    sortedVelocities[slot] = velocities[idx];
}
