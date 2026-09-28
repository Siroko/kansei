//! A `ClusterMesh` on the GPU, and the camera's per-frame cut of it.

use super::{ClusterMesh, Sphere};
use crate::geometries::Vertex;

/// The packed mesh's layout (`ClusterMesh::gpu_words`, read by cluster_mesh.wgsl).
pub(crate) const HEADER_WORDS: usize = 8;
pub(crate) const VERTEX_WORDS: usize = 9;
pub(crate) const CLUSTER_WORDS: usize = 28;
pub(crate) const LEVEL_WORDS: usize = 8;
/// The error of a missing parent (∞): shaders never see infinities.
pub(crate) const NO_PARENT: f32 = -1.0;

pub(crate) const CLUSTER_MESH_WGSL: &str = include_str!("../shaders/cluster_mesh.wgsl");

const _: () = assert!(std::mem::size_of::<Vertex>() == VERTEX_WORDS * 4);

impl ClusterMesh {
    /// The most triangles in one cluster: what every cluster is drawn as.
    pub fn max_triangles(&self) -> u32 {
        self.clusters.iter().map(|c| c.triangle_count).max().unwrap_or(0)
    }

    /// The mesh as the GPU reads it, in one buffer. A header of where each section starts (in
    /// words), the cluster and level counts, and `max_triangles`; then the vertices (`Vertex` as
    /// is), the clusters' vertices, their triangles (3 local indices in a word's low 3 bytes),
    /// the cluster records and the level records (see cluster_mesh.wgsl).
    pub fn gpu_words(&self) -> Vec<u32> {
        let levels = self.levels();
        let vertices = HEADER_WORDS;
        let cluster_vertices = vertices + self.vertices.len() * VERTEX_WORDS;
        let triangles = cluster_vertices + self.cluster_vertices.len();
        let clusters = triangles + self.cluster_triangles.len() / 3;
        let level_records = clusters + self.clusters.len() * CLUSTER_WORDS;
        let mut words = Vec::with_capacity(level_records + levels.len() * LEVEL_WORDS);
        words.extend([vertices, cluster_vertices, triangles, clusters, level_records, self.clusters.len(), levels.len(), self.max_triangles() as usize].map(|w| w as u32));
        words.extend_from_slice(bytemuck::cast_slice(&self.vertices));
        words.extend_from_slice(&self.cluster_vertices);
        words.extend(self.cluster_triangles.chunks(3).map(|t| t[0] as u32 | (t[1] as u32) << 8 | (t[2] as u32) << 16));
        let parent = |e: f32| if e.is_finite() { e } else { NO_PARENT };
        let sphere = |s: Sphere| [s.center.x, s.center.y, s.center.z, s.radius].map(f32::to_bits);
        for c in &self.clusters {
            words.extend([c.vertex_offset, c.triangle_offset, c.triangle_count, c.level]);
            words.extend(sphere(c.bounds));
            words.extend([c.cone_apex.x, c.cone_apex.y, c.cone_apex.z, c.cone_cutoff].map(f32::to_bits));
            words.extend([c.cone_axis.x, c.cone_axis.y, c.cone_axis.z, c.error].map(f32::to_bits));
            words.extend(sphere(c.lod_bounds));
            words.extend(sphere(c.parent_bounds));
            words.extend([parent(c.parent_error).to_bits(), 0, 0, 0]);
        }
        let finite = |x: f32| if x.is_finite() { x } else { 0.0 };
        for l in &levels {
            words.extend([l.first, l.count, l.min_error.to_bits(), parent(l.max_parent_error).to_bits(), finite(l.near_reach).to_bits(), finite(l.far_reach).to_bits(), 0, 0]);
        }
        words
    }
}
