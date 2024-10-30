export const renderCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) viewPosition : vec4<f32>,
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
    output.normal = (worldMatrix * vec4<f32>(normal, 0.0)).xyz;
    output.uv = uv;
    output.viewPosition = viewMatrix * worldMatrix * offsetVertex;
    return output;
} 

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    var pointLight = vec3<f32>(100.0, 50.0, 0.0);
    var lightVector = pointLight;
    var lightDirection = normalize(lightVector);
    var surfaceColor = vec3<f32>(0.967, 0.8321, 0.837);
    var fogColor = vec3<f32>(0.1);
    var diffuse = (dot(fragData.normal, lightDirection) + 1.0) / 2.0;
    var halfLambertDiffuse = (diffuse * 0.5 + 0.5) * surfaceColor;
    halfLambertDiffuse = mix(halfLambertDiffuse, fogColor,  smoothstep(0.0, 1.0, abs(fragData.viewPosition.z / 1000.0)));
    // halfLambertDiffuse = vec3<f32>(fragData.viewPosition.x, fragData.viewPosition.y, abs(fragData.viewPosition.z / 1000.0));
    return vec4<f32>(halfLambertDiffuse, 1.0);
} 
`;