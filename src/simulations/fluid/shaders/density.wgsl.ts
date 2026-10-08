import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

// One thread per *sorted* slot: threads in a workgroup are spatial neighbors, so they
// walk the same cells and hit the same cache lines. Output is in sorted order too.
@group(0) @binding(0) var<storage, read_write> sortedPositions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> densities: array<vec2<f32>>; // sorted order
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.particleCount) { return; }

    let pos = sortedPositions[slot].xyz;
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;
    let h2 = h * h;

    var density = 0.0;
    var nearDensity = 0.0;

    let zStart = select(-1, 0, params.dimensions == 2u);
    let zEnd = select(1, 0, params.dimensions == 2u);

    for (var dz = zStart; dz <= zEnd; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = coord + vec3<i32>(dx, dy, dz);
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= vec3<i32>(params.gridDims))) {
                    continue;
                }

                let neighborCell = cellHash(neighborCoord, params);
                let cellStart = cellOffsets[neighborCell];
                let cellEnd = select(cellOffsets[neighborCell + 1u], params.particleCount, neighborCell + 1u >= params.totalCells);

                for (var j = cellStart; j < cellEnd; j++) {
                    if (j == slot) { continue; }
                    let diff = pos - sortedPositions[j].xyz;
                    let d2 = dot(diff, diff);
                    if (d2 >= h2) { continue; }
                    // spiky kernels: (h-r)^2 and (h-r)^3, normalisation applied once below
                    let v = h - sqrt(d2);
                    let v2 = v * v;
                    density += v2;
                    nearDensity += v2 * v;
                }
            }
        }
    }

    densities[slot] = vec2<f32>(density * params.spikyPow2Factor, nearDensity * params.spikyPow3Factor);
}
`;
