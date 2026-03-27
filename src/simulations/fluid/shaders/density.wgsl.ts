import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

fn densityKernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return v * v * params.spikyPow2Factor;
}

fn nearDensityKernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return v * v * v * params.spikyPow3Factor;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;

    var density = 0.0;
    var nearDensity = 0.0;

    let zStart = select(-1, 0, params.dimensions == 2u);
    let zEnd = select(1, 0, params.dimensions == 2u);

    for (var dz = zStart; dz <= zEnd; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = coord + vec3<i32>(dx, dy, dz);

                // Bounds check
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= vec3<i32>(params.gridDims))) {
                    continue;
                }

                let neighborCell = cellHash(neighborCoord, params);
                let cellStart = cellOffsets[neighborCell];
                let cellEnd = select(cellOffsets[neighborCell + 1u], params.particleCount, neighborCell + 1u >= params.totalCells);

                for (var j = cellStart; j < cellEnd; j++) {
                    let neighborIdx = sortedIndices[j];
                    if (neighborIdx == idx) { continue; }

                    let neighborPos = positions[neighborIdx].xyz;
                    let diff = pos - neighborPos;
                    let dist = length(diff);

                    density += densityKernel(dist, h);
                    nearDensity += nearDensityKernel(dist, h);
                }
            }
        }
    }

    densities[idx] = vec2<f32>(density, nearDensity);
}
`;
