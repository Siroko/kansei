import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let cell = cellIndices[idx];
    let offset = cellOffsets[cell];
    let slot = atomicAdd(&scatterCounters[cell], 1u);
    sortedIndices[offset + slot] = idx;
}
`;
