struct Params { count: u32, threshold: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>, 2>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> group_max: atomic<u32>;
var<workgroup> group_above: atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        atomicStore(&group_max, 0u);
        atomicStore(&group_above, 0u);
    }
    workgroupBarrier();
    if (gid.x < params.count) {
        let s = length(velocities[gid.x].xyz);
        // non-negative floats order as their bits do
        atomicMax(&group_max, bitcast<u32>(s));
        if (s > params.threshold) {
            atomicAdd(&group_above, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        atomicMax(&result[0], atomicLoad(&group_max));
        atomicAdd(&result[1], atomicLoad(&group_above));
    }
}
