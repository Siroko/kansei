export const shaderCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) viewPosition : vec4<f32>
};

@group(0) @binding(0) var map : texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
    output.normal = (worldMatrix * vec4<f32>(normal, 1.0)).xyz;
    output.uv = uv;
    output.viewPosition = worldMatrix * position;
    return output;
} 

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    var uv: vec2<f32> = fragData.uv;
    var mapColor: vec4<f32> = textureSample(map, samp, uv);
    var sd: f32 = median(mapColor.r, mapColor.g, mapColor.b);
    let screenPxDistance: f32 = screenPxRange(uv) * (sd - 0.5);
    let opacity: f32 = clamp(screenPxDistance + 0.5, 0.0, 1.0);

    var bgColor: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    var fgColor: vec4<f32> = vec4<f32>(1.0 - fragData.viewPosition.z, 1.0, 1.0, opacity);
    var color: vec4<f32> = mix(bgColor, fgColor, opacity);
    
    return color;
} 

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

fn screenPxRange(uv: vec2f) -> f32 {
    let textureDimensions = vec2<f32>(textureDimensions(map));
    let pxRange = vec2<f32>(2.0, 2.0);
    let unitRange = vec2<f32>(pxRange) / textureDimensions;
    let screenTexSize = vec2<f32>(1.0) / fwidth(uv);
    return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

`;