//! Cluster LOD (meshlets, as Unreal's Nanite): a mesh split into clusters of about 124
//! triangles, with a graph of coarser versions over them, so a view can draw each part of the
//! mesh at the coarsest version whose error it doesn't see. See
//! `docs/plans/2026-09-28-cluster-lod-design.md`.

use glam::Vec3;

/// A sphere; clusters' bounds and the spheres their errors are measured from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    /// A sphere enclosing all of `spheres` (grown from the first, not the smallest).
    pub fn enclosing(spheres: impl IntoIterator<Item = Sphere>) -> Sphere {
        let mut spheres = spheres.into_iter();
        let mut s = spheres.next().expect("at least one sphere");
        for o in spheres {
            let d = o.center.distance(s.center);
            if d + o.radius <= s.radius {
                continue;
            }
            if d + s.radius <= o.radius {
                s = o;
                continue;
            }
            let radius = (d + s.radius + o.radius) * 0.5;
            let center = s.center + (o.center - s.center) * ((radius - s.radius) / d.max(1e-12));
            s = Sphere { center, radius };
        }
        s
    }

    /// Whether `other` lies inside (to float precision).
    pub fn contains(&self, other: &Sphere) -> bool {
        self.center.distance(other.center) + other.radius <= self.radius * (1.0 + 1e-5) + 1e-6
    }
}

use crate::geometries::Vertex;

mod build;

/// How `ClusterMesh::build` splits and simplifies.
#[derive(Clone, Copy, Debug)]
pub struct ClusterOptions {
    pub max_vertices: usize,
    pub max_triangles: usize,
    /// Clusters per group simplified together (about).
    pub group_size: usize,
    /// Triangles a group keeps when simplified.
    pub simplify_ratio: f32,
    /// A group keeping more than this share of its triangles has stalled: it isn't simplified.
    pub stall_ratio: f32,
    /// Weight of normal-cone tightness against compactness when splitting (backface culling).
    pub cone_weight: f32,
    /// Weights of the normals and uvs in the simplifier's error.
    pub normal_weight: f32,
    pub uv_weight: f32,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self { max_vertices: 64, max_triangles: 124, group_size: 8, simplify_ratio: 0.5, stall_ratio: 0.85, cone_weight: 0.25, normal_weight: 0.5, uv_weight: 0.1 }
    }
}

/// One cluster: a run of `cluster_vertices` and `cluster_triangles`, its bounds, and the errors
/// the cut rule compares (see `ClusterMesh::select`).
#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    /// In triangles (3 local indices each).
    pub triangle_offset: u32,
    pub triangle_count: u32,
    /// Of its triangles, for culling.
    pub bounds: Sphere,
    /// Its triangles' normal cone (backface culling): axis, and cos of the half-angle (1 = none).
    pub cone_axis: Vec3,
    pub cone_cutoff: f32,
    /// The error of the simplification that made it (0 at level 0), measured from `lod_bounds`.
    pub error: f32,
    pub lod_bounds: Sphere,
    /// The error of the group simplified from it and its siblings (∞ when none was), from
    /// `parent_bounds`.
    pub parent_error: f32,
    pub parent_bounds: Sphere,
    pub level: u32,
}

/// A mesh as a graph of clusters over its own vertices.
pub struct ClusterMesh {
    /// The geometry's vertices, shared by every level.
    pub vertices: Vec<Vertex>,
    pub clusters: Vec<Cluster>,
    /// Per cluster, its vertices as indices into `vertices`.
    pub cluster_vertices: Vec<u32>,
    /// Per cluster, 3 indices into its own vertices per triangle.
    pub cluster_triangles: Vec<u8>,
}

impl ClusterMesh {
    /// Cluster `cluster`'s triangles, as indices into `vertices`.
    pub fn triangles(&self, cluster: usize) -> impl Iterator<Item = [u32; 3]> + '_ {
        let c = &self.clusters[cluster];
        let vertices = &self.cluster_vertices[c.vertex_offset as usize..(c.vertex_offset + c.vertex_count) as usize];
        self.cluster_triangles[c.triangle_offset as usize * 3..(c.triangle_offset + c.triangle_count) as usize * 3]
            .chunks(3)
            .map(move |t| [vertices[t[0] as usize], vertices[t[1] as usize], vertices[t[2] as usize]])
    }
}

#[cfg(test)]
mod tests;
