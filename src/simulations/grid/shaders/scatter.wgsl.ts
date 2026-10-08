import { neighbourGridWgsl } from './neighbour-grid.wgsl';

/**
 * Counting-sort scatter. Besides the index permutation, it writes cell-ordered copies of
 * `copies` source buffers (bindings from 5 on, source then copy), so passes read neighbours
 * contiguously.
 */
export function scatterShader(copies: number): string {
    let bindings = '';
    let writes = '';
    for (let k = 0; k < copies; k++) {
        bindings += `@group(0) @binding(${5 + 2 * k}) var<storage, read_write> source${k}: array<vec4<f32>>;\n`;
        bindings += `@group(0) @binding(${6 + 2 * k}) var<storage, read_write> sorted${k}: array<vec4<f32>>;\n`;
        writes += `sorted${k}[slot] = source${k}[idx];\n    `;
    }
    return /* wgsl */`
${neighbourGridWgsl}
@group(0) @binding(0) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(4) var<uniform> grid: NeighbourGrid;
${bindings}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= grid.count) { return; }

    let cell = cellIndices[idx];
    let slot = cellOffsets[cell] + atomicAdd(&scatterCounters[cell], 1u);
    sortedIndices[slot] = idx;
    ${writes}
}
`;
}
