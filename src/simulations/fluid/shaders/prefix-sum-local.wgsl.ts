export const shaderCode = /* wgsl */`

const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

var<workgroup> sdata: array<u32, 512>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let blockOffset = wid.x * BLOCK_SIZE;
    let inputLen = arrayLength(&input);

    // Load two elements per thread into sdata memory
    let ai = tid;
    let bi = tid + 256u;
    let globalAi = blockOffset + ai;
    let globalBi = blockOffset + bi;
    sdata[ai] = select(0u, input[globalAi], globalAi < inputLen);
    sdata[bi] = select(0u, input[globalBi], globalBi < inputLen);

    // Up-sweep (reduce) phase
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

    // Save block total and clear last element
    if (tid == 0u) {
        blockSums[wid.x] = sdata[BLOCK_SIZE - 1u];
        sdata[BLOCK_SIZE - 1u] = 0u;
    }

    // Down-sweep phase
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

    // Write results
    if (globalAi < inputLen) { output[globalAi] = sdata[ai]; }
    if (globalBi < inputLen) { output[globalBi] = sdata[bi]; }
}
`;
