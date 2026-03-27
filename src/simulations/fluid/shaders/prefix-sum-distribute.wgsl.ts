export const shaderCode = /* wgsl */`

const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let blockOffset = wid.x * BLOCK_SIZE;
    let blockSum = blockSums[wid.x];
    let outputLen = arrayLength(&output);

    let globalAi = blockOffset + lid.x;
    let globalBi = blockOffset + lid.x + 256u;

    if (globalAi < outputLen) { output[globalAi] += blockSum; }
    if (globalBi < outputLen) { output[globalBi] += blockSum; }
}
`;
