export const shaderCode = /* wgsl */`
// Main compute shader function
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assuming you have a buffer for positions
    var position = positions[global_id.x];
    var velocity = velocities[global_id.x];
    
    position += velocity;

    // position.y *= 1.5;
    position.z = 10.0 * pow(sin(time.x + position.x*0.1), 3.0) + cos(time.x + position.y*0.1);


    // Store the result back
    positions[global_id.x] = position;
}

// Declare the storage buffer for positions
@group(0) @binding(0) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> origPositions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(3) var<uniform> time: vec4<f32>;
@group(0) @binding(4) var<uniform> width: f32;
`;
