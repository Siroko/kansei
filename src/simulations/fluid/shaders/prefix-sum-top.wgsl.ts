export const shaderCode = /* wgsl */`

const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> blockSums: array<u32>;

var<workgroup> shared: array<u32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let len = arrayLength(&blockSums);

    let ai = tid;
    let bi = tid + 256u;
    shared[ai] = select(0u, blockSums[ai], ai < len);
    shared[bi] = select(0u, blockSums[bi], bi < len);

    var offset = 1u;
    for (var d = BLOCK_SIZE >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai2 = offset * (2u * tid + 1u) - 1u;
            let bi2 = offset * (2u * tid + 2u) - 1u;
            shared[bi2] += shared[ai2];
        }
        offset <<= 1u;
    }

    if (tid == 0u) {
        shared[BLOCK_SIZE - 1u] = 0u;
    }

    for (var d2 = 1u; d2 < BLOCK_SIZE; d2 <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d2) {
            let ai3 = offset * (2u * tid + 1u) - 1u;
            let bi3 = offset * (2u * tid + 2u) - 1u;
            let temp = shared[ai3];
            shared[ai3] = shared[bi3];
            shared[bi3] += temp;
        }
    }

    workgroupBarrier();

    if (ai < len) { blockSums[ai] = shared[ai]; }
    if (bi < len) { blockSums[bi] = shared[bi]; }
}
`;
