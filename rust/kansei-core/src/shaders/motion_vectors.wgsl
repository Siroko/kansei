// Motion vectors for materials that write the GBuffer's velocity target (TAA):
// `MaterialOptions::outputs_velocity`, fragment output @location(4).
//
// Declare the mesh transforms with KanseiMeshTransforms instead of a bare mat4x4 world matrix,
//     @group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;
// compute this frame's and last frame's clip positions in the vertex shader (last frame's with
// mesh.prevWorld and last frame's animation: wind time, bones, instance transforms) and pass
// them on,
//     out.curr_clip = kansei_camera_temporal.viewProj * world_pos;
//     out.prev_clip = kansei_camera_temporal.prevViewProj * prev_world_pos;
// then write @location(4) kansei_motion_vector(in.curr_clip, in.prev_clip). Mark the vertex
// shader's position output `@builtin(position) @invariant`: the velocity pass depth-tests against
// the GBuffer pass, which runs the same shader in another pipeline.
// Pixels no material writes are reprojected from depth by the TAA (camera motion only).

struct KanseiCameraTemporal {
    viewProj     : mat4x4f,   // this frame, unjittered
    prevViewProj : mat4x4f,   // last frame, unjittered
    jitter       : vec2f,     // NDC offset of this frame's projection
    prevJitter   : vec2f,
    frame        : u32,
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,
}

struct KanseiMeshTransforms {
    world     : mat4x4f,
    prevWorld : mat4x4f,
}

@group(1) @binding(3) var<uniform> kansei_camera_temporal : KanseiCameraTemporal;

// Screen-space motion in uv units (current minus previous) from unjittered clip positions.
fn kansei_motion_vector(currClip: vec4f, prevClip: vec4f) -> vec2f {
    let c = currClip.xy / currClip.w;
    let p = prevClip.xy / prevClip.w;
    return (c - p) * vec2f(0.5, -0.5);
}
