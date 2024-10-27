export const renderCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) worldPosition : vec4<f32>,
};

@group(1) @binding(0) var<uniform> normalMatrix:mat4x4<f32>;
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
    output.position = projectionMatrix * viewMatrix * worldMatrix * offsetVertex;
    output.normal = (normalMatrix * vec4<f32>(normal, 1.0)).xyz;
    output.uv = uv;
    output.worldPosition = worldMatrix * offsetVertex;
    return output;
} 

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    var pointLight = vec3<f32>(0.0, 1000.0, 1.0);
    var lightVector = pointLight - fragData.worldPosition.xyz;
    var lightDistance = length(lightVector);
    var lightDirection = lightVector / lightDistance;
    var diffuse = max(dot(fragData.normal, lightDirection), 0.0);
    var color = vec3<f32>(diffuse, diffuse, diffuse);
    return vec4<f32>(color, 1.0);
} 
`;