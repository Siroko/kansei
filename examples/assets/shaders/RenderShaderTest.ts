export const renderCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>
};

@group(1) @binding(0) var<uniform> modelMatrix:mat4x4<f32>;
@group(1) @binding(1) var<uniform> worldMatrix:mat4x4<f32>;

@group(2) @binding(0) var<uniform> viewMatrix:mat4x4<f32>;
@group(2) @binding(1) var<uniform> projectionMatrix:mat4x4<f32>;

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) a_particlePos : vec4<f32>
) -> VertexOut
{
    var output : VertexOut;
    var offsetVertex: vec4<f32> = vec4<f32>(position.xyz + a_particlePos.xyz, 1.0);
    output.position = projectionMatrix * viewMatrix * modelMatrix * offsetVertex;
    output.normal = (worldMatrix * vec4<f32>(normal, 1.0)).xyz;
    output.uv = uv;
    return output;
} 

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    return vec4<f32>(fragData.normal, 1.0);
} 
`;