import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

struct DensityFieldParams {
    texDims       : vec3<u32>,
    particleCount : u32,
    boundsMin     : vec3<f32>,
    smoothingRadius: f32,
    boundsMax     : vec3<f32>,
    kernelScale   : f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accumBuffer: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> fieldParams: DensityFieldParams;

fn worldToVoxel(worldPos: vec3<f32>) -> vec3<f32> {
    let norm = (worldPos - fieldParams.boundsMin) / (fieldParams.boundsMax - fieldParams.boundsMin);
    return norm * vec3<f32>(fieldParams.texDims);
}

fn voxelIndex(coord: vec3<u32>) -> u32 {
    return coord.z * fieldParams.texDims.x * fieldParams.texDims.y
         + coord.y * fieldParams.texDims.x
         + coord.x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fieldParams.particleCount) { return; }

    let pos = positions[idx].xyz;
    let voxelPos = worldToVoxel(pos);
    let h = fieldParams.smoothingRadius;
    let dims = fieldParams.texDims;

    // Compute kernel radius in voxel units
    let voxelSize = (fieldParams.boundsMax - fieldParams.boundsMin) / vec3<f32>(dims);
    let kernelVoxels = vec3<i32>(ceil(vec3<f32>(h) / voxelSize));

    let centerVoxel = vec3<i32>(floor(voxelPos));

    for (var dz = -kernelVoxels.z; dz <= kernelVoxels.z; dz++) {
        for (var dy = -kernelVoxels.y; dy <= kernelVoxels.y; dy++) {
            for (var dx = -kernelVoxels.x; dx <= kernelVoxels.x; dx++) {
                let voxel = centerVoxel + vec3<i32>(dx, dy, dz);

                // Bounds check
                if (any(voxel < vec3<i32>(0)) || any(voxel >= vec3<i32>(dims))) {
                    continue;
                }

                // World-space distance from particle to voxel center
                let voxelCenter = (vec3<f32>(voxel) + 0.5) * voxelSize + fieldParams.boundsMin;
                let dist = length(pos - voxelCenter);

                if (dist >= h) { continue; }

                // Poly6-like kernel (smooth, non-negative)
                let t = 1.0 - dist / h;
                let weight = t * t * t * fieldParams.kernelScale;

                // Atomic accumulate (fixed-point: multiply by 1024, store as u32)
                let fixedPoint = u32(weight * 1024.0);
                if (fixedPoint > 0u) {
                    let vi = voxelIndex(vec3<u32>(voxel));
                    atomicAdd(&accumBuffer[vi], fixedPoint);
                }
            }
        }
    }
}
`;
