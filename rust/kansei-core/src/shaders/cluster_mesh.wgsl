// A cluster mesh's words (clusters/gpu.rs, ClusterMesh::gpu_words) in `kansei_cluster_mesh`,
// which the including module binds. Header: where the vertices, the clusters' vertices, their
// triangles, the cluster records and the level records start (words), the cluster and level
// counts, and the triangles every cluster is drawn as.

const KANSEI_CLUSTER_WORDS: u32 = 28u;
const KANSEI_LEVEL_WORDS: u32 = 8u;
const KANSEI_VERTEX_WORDS: u32 = 9u;

fn kansei_cluster_word(cluster: u32, word: u32) -> u32 {
    return kansei_cluster_mesh[kansei_cluster_mesh[3] + cluster * KANSEI_CLUSTER_WORDS + word];
}

fn kansei_cluster_f32(cluster: u32, word: u32) -> f32 {
    return bitcast<f32>(kansei_cluster_word(cluster, word));
}

fn kansei_cluster_vec4(cluster: u32, word: u32) -> vec4<f32> {
    return vec4<f32>(kansei_cluster_f32(cluster, word), kansei_cluster_f32(cluster, word + 1u), kansei_cluster_f32(cluster, word + 2u), kansei_cluster_f32(cluster, word + 3u));
}

// The mesh vertex that is vertex `local` of `cluster` (an index of its triangles).
fn kansei_cluster_local_vertex(cluster: u32, local: u32) -> u32 {
    return kansei_cluster_mesh[kansei_cluster_mesh[1] + kansei_cluster_word(cluster, 0u) + local];
}

// Word `word` of a mesh vertex: position 0-3, normal 4-6, uv 7-8.
fn kansei_vertex_f32(vertex: u32, word: u32) -> f32 {
    return bitcast<f32>(kansei_cluster_mesh[kansei_cluster_mesh[0] + vertex * KANSEI_VERTEX_WORDS + word]);
}

fn kansei_level_word(level: u32, word: u32) -> u32 {
    return kansei_cluster_mesh[kansei_cluster_mesh[4] + level * KANSEI_LEVEL_WORDS + word];
}
