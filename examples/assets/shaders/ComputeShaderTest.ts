export const shaderCode = /* wgsl */`
@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(1) 
var<uniform> uTime: vec4<f32>;

fn hash(p: vec3<f32>) -> vec3<f32> {
    let q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yxz + 33.33);
    return fract((r.xxy + r.yxx) * r.zyx);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(dot(hash(i + vec3<f32>(0.0, 0.0, 0.0)), f - vec3<f32>(0.0, 0.0, 0.0)),
                dot(hash(i + vec3<f32>(1.0, 0.0, 0.0)), f - vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(dot(hash(i + vec3<f32>(0.0, 1.0, 0.0)), f - vec3<f32>(0.0, 1.0, 0.0)),
                dot(hash(i + vec3<f32>(1.0, 1.0, 0.0)), f - vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(dot(hash(i + vec3<f32>(0.0, 0.0, 1.0)), f - vec3<f32>(0.0, 0.0, 1.0)),
                dot(hash(i + vec3<f32>(1.0, 0.0, 1.0)), f - vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(dot(hash(i + vec3<f32>(0.0, 1.0, 1.0)), f - vec3<f32>(0.0, 1.0, 1.0)),
                dot(hash(i + vec3<f32>(1.0, 1.0, 1.0)), f - vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

fn curl(p: vec3<f32>) -> vec3<f32> {
    let e = vec3<f32>(0.01, 0.0, 0.0);
    let dx = vec3<f32>(
        noise(p + e.xyy) - noise(p - e.xyy),
        noise(p + e.yxy) - noise(p - e.yxy),
        noise(p + e.yyx) - noise(p - e.yyx)
    );
    return normalize(dx);
}

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3<u32>,
) {
    let id = f32(global_id.x);
    let time = uTime.x;
    let scale = 0.1;
    let speed = 0.5;

    var value = output[global_id.x];
    let curlNoise = curl(vec3<f32>(sin(time) * 0.2, cos(time) * 0.2, 0.0) + value.xyz * 0.003);
    // value += vec4<f32>(curlNoise * 0.3, 0.0);

    output[global_id.x] = value;
}
`;
