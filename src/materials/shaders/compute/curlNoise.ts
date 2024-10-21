export const shaderCode = /* wgsl */`

@group(0) @binding(0)
var<storage, read_write> positions: array<vec3<f32>>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3<u32>,
) {
    var value: vec3<f32> = positions[global_id.x];
    value.x += 0.1;

    output[global_id.x] = value;
}
`;