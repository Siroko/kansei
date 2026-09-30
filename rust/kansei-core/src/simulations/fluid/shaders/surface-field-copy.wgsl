// Surface field, resolve: 1 + (particle radius - distance to the weighted mean of the particles
// around) / kernel radius. 1 is the surface, more is inside; a voxel no particle reaches is at
// least a kernel radius away. A lone particle is a sphere of the particle radius, and a flat
// layer of particles a flat surface. Where the particles around weigh past a third of the
// bulk's (kernelScale normalises the weight to 1 in the bulk) the voxel is inside whatever the
// mean: against a wall or floor the mean leans away from it, which would otherwise put a second
// surface along every wall.

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

@group(0) @binding(0) var<storage, read_write> accum: array<u32>;
@group(0) @binding(1) var densityTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: SurfaceFieldParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = params.texDims;
    if (any(gid >= dims)) { return; }
    let i = 4u * (gid.z * dims.x * dims.y + gid.y * dims.x + gid.x);
    let w = f32(accum[i]);
    let r = params.kernelRadius;
    var distance = r;
    if (w > 0.0) {
        let mean = vec3<f32>(f32(accum[i + 1u]), f32(accum[i + 2u]), f32(accum[i + 3u])) / w - vec3<f32>(r);
        distance = min(length(mean), r);
    }
    let full = w / FIXED * params.kernelScale;
    let value = max(1.0 + (params.particleRadius - distance) / r, 1.0 + (full - 0.35) * 2.0);
    textureStore(densityTex, gid, vec4<f32>(value, 0.0, 0.0, 1.0));
    accum[i] = 0u;
    accum[i + 1u] = 0u;
    accum[i + 2u] = 0u;
    accum[i + 3u] = 0u;
}
