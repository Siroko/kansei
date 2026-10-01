// Surface field (Zhu & Bridson 2005, "Animating sand as a fluid"), splat: per voxel, the
// kernel-weighted sum of the weights and of the particles' offsets from the voxel's centre.
// Fixed point in u32 atomics (offsets shifted by the kernel radius to stay positive).

struct SurfaceFieldParams {
    texDims       : vec3<u32>,
    particleCount : u32,
    boundsMin     : vec3<f32>,
    kernelRadius  : f32,
    boundsMax     : vec3<f32>,
    kernelScale   : f32,
    particleRadius: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};

const FIXED: f32 = 4096.0;

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: SurfaceFieldParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }
    let pos = positions[idx].xyz;
    let dims = params.texDims;
    let voxel_size = (params.boundsMax - params.boundsMin) / vec3<f32>(dims);
    let r = params.kernelRadius;
    let reach = vec3<i32>(ceil(vec3<f32>(r) / voxel_size));
    let center = vec3<i32>(floor((pos - params.boundsMin) / voxel_size));
    for (var dz = -reach.z; dz <= reach.z; dz++) {
        for (var dy = -reach.y; dy <= reach.y; dy++) {
            for (var dx = -reach.x; dx <= reach.x; dx++) {
                let v = center + vec3<i32>(dx, dy, dz);
                if (any(v < vec3<i32>(0)) || any(v >= vec3<i32>(dims))) { continue; }
                let c = (vec3<f32>(v) + 0.5) * voxel_size + params.boundsMin;
                let o = pos - c;
                let q = dot(o, o) / (r * r);
                if (q >= 1.0) { continue; }
                let w = (1.0 - q) * (1.0 - q) * (1.0 - q);
                let i = 4u * (u32(v.z) * dims.x * dims.y + u32(v.y) * dims.x + u32(v.x));
                atomicAdd(&accum[i], u32(w * FIXED));
                let s = w * (o + vec3<f32>(r)) * FIXED;
                atomicAdd(&accum[i + 1u], u32(s.x));
                atomicAdd(&accum[i + 2u], u32(s.y));
                atomicAdd(&accum[i + 3u], u32(s.z));
            }
        }
    }
}
