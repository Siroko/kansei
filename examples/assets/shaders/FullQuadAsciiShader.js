export const shaderCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>

};

@group(0) @binding(0) var externalTexture : texture_external;
@group(0) @binding(1) var asciiMap : texture_2d<f32>;
@group(0) @binding(2) var mySampler: sampler;
@group(0) @binding(3) var<uniform> res:vec4<f32>;

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
    output.position = projectionMatrix * viewMatrix * modelMatrix * position;
    output.normal = (worldMatrix * vec4<f32>(normal, 1.0)).xyz;
    output.uv = uv;
    return output;
} 

fn rand(n: vec2<f32>) -> f32 { 
    return fract(sin(dot(n, vec2<f32>(12.9898, 4.1414))) * 43758.5453);
}
@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    var cols: f32 = 120.0;
    var rows: f32 = 60.0;

    var uv:vec2<f32> = vec2<f32>(fragData.position.xy / vec2<f32>(res.z, res.w));

    var discreteUv: vec2<f32> = vec2<f32>(fragData.position.xy / vec2<f32>(res.z, res.w));
    discreteUv.x *= cols;
    discreteUv.x = floor(discreteUv.x) / cols;
    discreteUv.y *= rows;
    discreteUv.y = floor(discreteUv.y) / rows;

    var gridUv: vec2<f32> = vec2<f32>(uv.x * cols % 1.0, uv.y * rows % 1.0);

    var discreteCoords:vec2<u32> = vec2<u32>(u32(discreteUv.x * res.x), u32(discreteUv.y * res.y));
    var videoColor:vec4<f32> = textureLoad(externalTexture, discreteCoords);
    var luminance:f32 = (videoColor.x + videoColor.g + videoColor.b) / 3.0;
    
    var colsInMap: f32 = 10.0;
    var rowsInMap: f32 = 10.0;
    var discreteLuminance:f32 = floor(luminance * colsInMap) / colsInMap;
    discreteLuminance = clamp(0.0, 1.0, ((discreteLuminance - 0.5)) + 0.5);

    var jitter: f32 = floor(rand(discreteUv) * rowsInMap) / rowsInMap;
    var gridCoords: vec2<f32> = vec2<f32>(gridUv.x / colsInMap + discreteLuminance, gridUv.y / rowsInMap + jitter);
    var asciiTable:vec4<f32> = textureSample(asciiMap, mySampler, gridCoords);
    // asciiTable *= videoColor;
    // return asciiTable;
    return vec4<f32>(fragData.uv, 0.0, 1.0);
    // return vec4<f32>(discreteLuminance * gridUv.x, discreteLuminance * gridUv.y, discreteLuminance, 1.0);
} 
`;