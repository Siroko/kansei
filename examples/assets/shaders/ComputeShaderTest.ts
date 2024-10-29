export const shaderCode = /* wgsl */`
#include <curl>

// Main compute shader function
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assuming you have a buffer for positions
    var value = positions[global_id.x];
    
    var projected = projectionMatrix * viewMatrix * worldMatrix * vec4<f32>(value, 1.0);
    var ndc = projected.xyz / projected.w; 
    var ndcMouse = mousePosition;
    ndcMouse.y *= -1.0;
    var distanceToMouse = distance(ndcMouse, ndc.xy);
    if(distanceToMouse < 0.5) {
        value.x += mouseDirection.x * 10.0;
        value.y += mouseDirection.y * 10.0;
    }

    // Apply curl noise
    let curl_velocity = getCurlVelocity(vec4<f32>(value, 0.0));
    value += curl_velocity * 0.01;

    // Store the result back
    positions[global_id.x] = value;
}

// Declare the storage buffer for positions
@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<uniform> time: vec4<f32>;
@group(0) @binding(2) var<uniform> mouseDirection: vec2<f32>;
@group(0) @binding(3) var<uniform> mousePosition: vec2<f32>;
@group(0) @binding(4) var<uniform> worldMatrix: mat4x4<f32>;
@group(0) @binding(5) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(6) var<uniform> projectionMatrix: mat4x4<f32>;
// @group(0) @binding(3) var<uniform> mouseStrength: f32;

`;
