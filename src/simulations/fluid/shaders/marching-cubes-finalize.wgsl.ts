export const shaderCode = /* wgsl */`
struct Counter {
    value: atomic<u32>,
};

struct Params {
    dims_and_max_tris: vec4<u32>,
    bounds_min_and_iso: vec4<f32>,
    bounds_max_pad: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> counter: Counter;
@group(0) @binding(1) var<storage, read_write> indirectArgs: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let triCount = min(atomicLoad(&counter.value), params.dims_and_max_tris.w);
    indirectArgs[0] = triCount * 3u;
    indirectArgs[1] = 1u;
    indirectArgs[2] = 0u;
    indirectArgs[3] = 0u;
    indirectArgs[4] = 0u;
}
`;
