export const shaderCode = /* wgsl */`

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3<u32>,
) {
    var value:vec4<f32> = output[global_id.x];
    value += vec4<f32>(10.0, 1.0, 10.0, 1.0);  // Increase by 10 instead of 1

    output[global_id.x] = value;
}
`;
