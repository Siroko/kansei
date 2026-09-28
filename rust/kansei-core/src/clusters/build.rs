use glam::Vec3;
use optimesh::clusterizer::{build_meshlets, build_meshlets_bound, Meshlet, MeshletBuffers, Positions};
use optimesh::meshletutils::compute_cluster_bounds;

use super::{Cluster, ClusterMesh, ClusterOptions, Sphere};
use crate::geometries::Geometry;

impl ClusterMesh {
    /// Split `geometry` into clusters and build the graph of coarser versions over them.
    pub fn build(geometry: &Geometry, options: &ClusterOptions) -> ClusterMesh {
        let positions: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
        let mut mesh = ClusterMesh { vertices: geometry.vertices.clone(), clusters: Vec::new(), cluster_vertices: Vec::new(), cluster_triangles: Vec::new() };
        mesh.split(&geometry.indices, &positions, 0.0, None, 0, options);
        mesh
    }

    /// Clusters of `indices` (into `vertices`), carrying `error` and `lod_bounds` (their own
    /// bounds when `None`); per new cluster, its index and its triangles as indices into
    /// `vertices`.
    pub(super) fn split(&mut self, indices: &[u32], positions: &[f32], error: f32, lod_bounds: Option<Sphere>, level: u32, options: &ClusterOptions) -> Vec<(usize, Vec<u32>)> {
        if indices.is_empty() {
            return Vec::new();
        }
        let bound = build_meshlets_bound(indices.len(), options.max_vertices, options.max_triangles);
        let mut meshlets = vec![Meshlet::default(); bound];
        let mut vertices = vec![0u32; bound * options.max_vertices];
        let mut triangles = vec![0u8; bound * options.max_triangles * 3];
        let count = build_meshlets(
            &mut MeshletBuffers { meshlets: &mut meshlets, vertices: &mut vertices, triangles: &mut triangles },
            indices,
            &Positions { data: positions, count: positions.len() / 3, stride: 12 },
            options.max_vertices,
            options.max_triangles,
            options.cone_weight,
        );
        let mut out = Vec::with_capacity(count);
        for m in &meshlets[..count] {
            let local_vertices = &vertices[m.vertex_offset as usize..(m.vertex_offset + m.vertex_count) as usize];
            // (meshlet triangle offsets count indices, 3 per triangle)
            let local_triangles = &triangles[m.triangle_offset as usize..(m.triangle_offset + m.triangle_count * 3) as usize];
            let global: Vec<u32> = local_triangles.iter().map(|&t| local_vertices[t as usize]).collect();
            let b = compute_cluster_bounds(&global, positions, positions.len() / 3, 12);
            let bounds = Sphere { center: Vec3::from(b.center), radius: b.radius };
            let lod_bounds = lod_bounds.unwrap_or(bounds);
            out.push((self.clusters.len(), global));
            self.clusters.push(Cluster {
                vertex_offset: self.cluster_vertices.len() as u32,
                vertex_count: m.vertex_count,
                triangle_offset: (self.cluster_triangles.len() / 3) as u32,
                triangle_count: m.triangle_count,
                bounds,
                cone_axis: Vec3::from(b.cone_axis),
                cone_cutoff: b.cone_cutoff,
                error,
                lod_bounds,
                parent_error: f32::INFINITY,
                parent_bounds: lod_bounds,
                level,
            });
            self.cluster_vertices.extend_from_slice(local_vertices);
            self.cluster_triangles.extend_from_slice(local_triangles);
        }
        out
    }
}
