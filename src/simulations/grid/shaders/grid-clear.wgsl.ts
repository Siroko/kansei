export const shaderCode = /* wgsl */`
// Clears both per-cell counters in one dispatch.
@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> scatterCounters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&cellCounts)) { return; }
    atomicStore(&cellCounts[idx], 0u);
    atomicStore(&scatterCounters[idx], 0u);
}
`;
