export const shaderCode = /* wgsl */`

struct DensityFieldParams {
    texDims       : vec3<u32>,
    particleCount : u32,
    boundsMin     : vec3<f32>,
    smoothingRadius: f32,
    boundsMax     : vec3<f32>,
    kernelScale   : f32,
};

@group(0) @binding(0) var<storage, read_write> accumBuffer: array<u32>;
@group(0) @binding(1) var densityTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> fieldParams: DensityFieldParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = fieldParams.texDims;
    if (any(gid >= dims)) { return; }

    let vi = gid.z * dims.x * dims.y + gid.y * dims.x + gid.x;
    let raw = f32(accumBuffer[vi]) / 1024.0;

    // Store density in R channel. G,B,A reserved for future (normal, material ID).
    textureStore(densityTex, gid, vec4<f32>(raw, 0.0, 0.0, 1.0));

    // Clear accumulator for next frame
    accumBuffer[vi] = 0u;
}
`;
