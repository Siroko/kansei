export const shaderCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>
};

@group(0) @binding(0) var map : texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@group(1) @binding(0) var<uniform> modelMatrix:mat4x4<f32>;
@group(1) @binding(1) var<uniform> worldMatrix:mat4x4<f32>;

@group(2) @binding(0) var<uniform> viewMatrix:mat4x4<f32>;
@group(2) @binding(1) var<uniform> projectionMatrix:mat4x4<f32>;

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>
) -> VertexOut
{
    var output : VertexOut;
    output.position = projectionMatrix * viewMatrix * worldMatrix * position;
    output.normal = (worldMatrix * vec4<f32>(normal, 1.0)).xyz;
    output.uv = uv;
    return output;
} 

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    var mapColor:vec4<f32> = textureSample(map, samp, fragData.uv);
    
    return mapColor;
} 
`;