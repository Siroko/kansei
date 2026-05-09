@group(0) @binding(0) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> scatterCounters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let len0 = arrayLength(&cellCounts);
    let len1 = arrayLength(&scatterCounters);
    if (idx < len0) {
        atomicStore(&cellCounts[idx], 0u);
    }
    if (idx < len1) {
        atomicStore(&scatterCounters[idx], 0u);
    }
}
