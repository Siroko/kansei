export const shaderCode = /* wgsl */`
struct Counter {
    value: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> counter: Counter;
@group(0) @binding(1) var<storage, read_write> indirectArgs: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    atomicStore(&counter.value, 0u);
    indirectArgs[0] = 0u;
    indirectArgs[1] = 1u;
    indirectArgs[2] = 0u;
    indirectArgs[3] = 0u;
    indirectArgs[4] = 0u;
}
`;
