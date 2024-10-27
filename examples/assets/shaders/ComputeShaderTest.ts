export const shaderCode = /* wgsl */`
#include <curl>

// Main compute shader function
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assuming you have a buffer for positions
    var value = positions[global_id.x];
    
    // Apply curl noise
    let curl_velocity = getCurlVelocity(vec4<f32>(value, 0.0));
    value += curl_velocity * 0.1;

    // Store the result back
    positions[global_id.x] = value;
}

// Declare the storage buffer for positions
@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;

`;
