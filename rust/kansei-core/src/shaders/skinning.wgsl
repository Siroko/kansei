// Vertex skinning by vertex pulling (`animation::SkinnedMesh`, `animation::BonePalette`); the
// geometry keeps the standard vertex layout and its bind pose.
//
// Group 0, binding 1: the bone palette, one matrix per skin joint (its model transform times its
// inverse bind matrix): this frame's matrices, then last frame's for motion vectors.
// Group 0, binding 2: per vertex, four skin joints (u16 pairs) and four weights (unorm16 pairs).
//
// In vertex_main, take `@builtin(vertex_index)` (an indexed draw gives the index buffer's value)
// and call `kansei_skin(vertex_index, position, normal)`: the result is in the mesh's model space,
// ready for the world matrix. Its `prev_position` is last frame's, for
// `kansei_camera_temporal.prevViewProj * mesh.prevWorld` (`cameras::MOTION_VECTORS_WGSL`).

@group(0) @binding(1) var<storage, read> kansei_bones: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> kansei_skin_vertices: array<vec4u>;

struct KanseiSkinned {
    position: vec3f,
    normal: vec3f,
    prev_position: vec3f,
}

// The vertex's blended joint matrix from the palette half starting at `base`.
fn kansei_skin_matrix(vertex: u32, base: u32) -> mat4x4f {
    let s = kansei_skin_vertices[vertex];
    let w01 = unpack2x16unorm(s.z);
    let w23 = unpack2x16unorm(s.w);
    return kansei_bones[base + (s.x & 0xffffu)] * w01.x
         + kansei_bones[base + (s.x >> 16u)] * w01.y
         + kansei_bones[base + (s.y & 0xffffu)] * w23.x
         + kansei_bones[base + (s.y >> 16u)] * w23.y;
}

fn kansei_skin(vertex: u32, position: vec3f, normal: vec3f) -> KanseiSkinned {
    let joints = arrayLength(&kansei_bones) / 2u;
    let m = kansei_skin_matrix(vertex, 0u);
    var out: KanseiSkinned;
    out.position = (m * vec4f(position, 1.0)).xyz;
    out.normal = (m * vec4f(normal, 0.0)).xyz;
    out.prev_position = (kansei_skin_matrix(vertex, joints) * vec4f(position, 1.0)).xyz;
    return out;
}
