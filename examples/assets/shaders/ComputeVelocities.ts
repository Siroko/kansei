export const shaderCode = /* wgsl */`
#include <curl>

// Main compute shader function
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assuming you have a buffer for positions
    var velocity = velocities[global_id.x];
    var position = positions[global_id.x];
    
    // damping
    velocity *= 0.99;

    var projected = projectionMatrix * viewMatrix * worldMatrix * vec4<f32>(position, 1.0);
    var ndc = projected.xyz / projected.w; 
    var ndcMouse = mousePosition;
    ndcMouse.y *= -1.0;
    var mouseStrength = time.w;
    var distanceToMouse = distance(ndcMouse, ndc.xy);

    if(distanceToMouse < mouseStrength * 2.0 && mouseStrength > 0.01) {
        var nDistance = distanceToMouse / (mouseStrength * 2.0);
        velocity.x += mouseDirection.x * mouseStrength * 300.0 * (1.0 - nDistance);
        velocity.y += mouseDirection.y * mouseStrength * 300.0 * (1.0 - nDistance);
    }

    // Apply curl noise
    let curl_velocity = getCurlVelocity(vec4<f32>(position, 0.0));
    velocity += curl_velocity * 0.001;

    // Store the result back
    velocities[global_id.x] = velocity;
}

// Declare the storage buffer for positions
@group(0) @binding(0) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> time: vec4<f32>;
@group(0) @binding(3) var<uniform> mouseDirection: vec2<f32>;
@group(0) @binding(4) var<uniform> mousePosition: vec2<f32>;
@group(0) @binding(5) var<uniform> worldMatrix: mat4x4<f32>;
@group(0) @binding(6) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(7) var<uniform> projectionMatrix: mat4x4<f32>;

`;
