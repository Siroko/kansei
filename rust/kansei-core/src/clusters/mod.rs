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
mod cards;
mod gpu;
mod vertex_stage;
pub(crate) use vertex_stage::{cluster_vertex_stage, CLUSTER_VERTEX_ENTRY};

pub use gpu::{ClusterLod, InstanceTransform};
pub(crate) use gpu::{ClusterCulling, ClusterGpu, ClusterViewGpu, InstanceSource};

/// How `ClusterMesh::build` splits and simplifies.
#[derive(Clone, Copy, Debug)]
pub struct ClusterOptions {
    /// Vertices a cluster may reference (at most 256: its triangles' indices are bytes). 128
    /// lets clusters fill up to `max_triangles` (a cluster is drawn as `max_triangles`).
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
    /// Treat small, open, flat-ish components as cards (foliage: sprays, leaves, ribbons), whose
    /// coarser levels are pruned rather than simplified (off by default: a solid mesh built of
    /// separate flat panels would lose them at a distance).
    pub cards: bool,
    /// The most triangles a card has.
    pub card_max_triangles: usize,
    /// The share of a card's area its area-weighted normal keeps (1: flat; a tube or a closed
    /// shape: about 0).
    pub card_flatness: f32,
    /// Scales pruned levels' error: below 1, crowns thin out nearer (see `ClusterMesh::build`).
    pub card_error_scale: f32,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            max_vertices: 128,
            max_triangles: 124,
            group_size: 8,
            simplify_ratio: 0.5,
            stall_ratio: 0.85,
            cone_weight: 0.25,
            normal_weight: 0.5,
            uv_weight: 0.1,
            cards: false,
            card_max_triangles: 64,
            card_flatness: 0.5,
            card_error_scale: 1.0,
        }
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
    /// Its triangles' normal cone, for backface culling (`backfacing`), as meshoptimizer gives
    /// it: the apex, the axis, and the sine of the cone's half-angle (its cutoff widened by 90°).
    /// A cone too wide to cull (over ~168°) has a zero axis and a cutoff of 1, and culls nothing.
    pub cone_apex: Vec3,
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

impl Cluster {
    /// Whether every one of its triangles faces away from `eye` (the mesh's own space), so a
    /// view there can skip it: `dot(normalize(cone_apex - eye), cone_axis) >= cone_cutoff`.
    pub fn backfacing(&self, eye: Vec3) -> bool {
        (self.cone_apex - eye).normalize_or_zero().dot(self.cone_axis) >= self.cone_cutoff
    }
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

/// Where a view sees a mesh from, in the mesh's own space, and how much error it tolerates.
#[derive(Clone, Copy, Debug)]
pub struct LodView {
    pub eye: Vec3,
    /// Pixels per radian at the view's centre: the viewport's height / (2 tan(fov_y / 2)).
    pub pixels_per_radian: f32,
    /// Distances are clamped to this (an eye inside a sphere).
    pub near: f32,
    /// The error budget, pixels.
    pub threshold: f32,
}

/// `error` (metres) seen from the view as pixels: over the distance to the nearest point of
/// `sphere`. A parent's sphere contains its children's and its error is at least theirs, so its
/// projected error is at least theirs from any eye.
pub fn projected_error(error: f32, sphere: Sphere, view: &LodView) -> f32 {
    projected_error_at(error, sphere.center.distance(view.eye) - sphere.radius, view)
}

/// `error` seen from `distance` away (clamped to the view's `near`), in pixels.
pub fn projected_error_at(error: f32, distance: f32, view: &LodView) -> f32 {
    if error == 0.0 {
        return 0.0;
    }
    if !error.is_finite() {
        return f32::INFINITY;
    }
    error / distance.max(view.near) * view.pixels_per_radian
}

/// A relative margin `LevelBounds::may_draw` gives the budget, so rounding (on the GPU, with
/// transformed spheres) never skips a level the cut rule would draw from.
pub const WINDOW_SLACK: f32 = 1e-3;

/// What one build round's clusters (`Cluster::level`) span, to skip a whole level: `may_draw`
/// is false only when none of them can pass the cut rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelBounds {
    /// Its clusters: `first..first + count` of `ClusterMesh::clusters`.
    pub first: u32,
    pub count: u32,
    /// The smallest of its clusters' errors.
    pub min_error: f32,
    /// The largest of their parents' errors (∞ when one has none).
    pub max_parent_error: f32,
    /// The farthest a cluster's LOD sphere's nearest point lies from the mesh's origin, beyond
    /// the eye's own distance: the largest `|lod_bounds.center| - lod_bounds.radius`.
    pub near_reach: f32,
    /// The farthest a parent's sphere reaches from the origin: the largest
    /// `|parent_bounds.center| + parent_bounds.radius` (of the finite parents).
    pub far_reach: f32,
}

impl LevelBounds {
    /// Whether some cluster of the level may pass the cut rule for `view`. With `d` the eye's
    /// distance to the mesh's origin, each cluster's own sphere is at most `d + near_reach` away
    /// (so its error projects to at least `min_error` from there) and each parent's at least
    /// `d - far_reach` (so its error projects to at most `max_parent_error` from there).
    pub fn may_draw(&self, view: &LodView) -> bool {
        let d = view.eye.length();
        let fine_enough = projected_error_at(self.min_error, d + self.near_reach, view) <= view.threshold * (1.0 + WINDOW_SLACK);
        let parent_over = !self.max_parent_error.is_finite() || projected_error_at(self.max_parent_error, d - self.far_reach, view) > view.threshold * (1.0 - WINDOW_SLACK);
        fine_enough && parent_over
    }
}

impl ClusterMesh {
    /// Each build round's clusters and what they span (clusters are stored by round).
    pub fn levels(&self) -> Vec<LevelBounds> {
        let mut levels: Vec<LevelBounds> = Vec::new();
        for (i, c) in self.clusters.iter().enumerate() {
            while levels.len() <= c.level as usize {
                levels.push(LevelBounds { first: i as u32, count: 0, min_error: f32::INFINITY, max_parent_error: 0.0, near_reach: f32::NEG_INFINITY, far_reach: f32::NEG_INFINITY });
            }
            let level = &mut levels[c.level as usize];
            debug_assert_eq!(level.first + level.count, i as u32, "clusters are stored by level");
            level.count += 1;
            level.min_error = level.min_error.min(c.error);
            level.max_parent_error = level.max_parent_error.max(c.parent_error);
            level.near_reach = level.near_reach.max(c.lod_bounds.center.length() - c.lod_bounds.radius);
            if c.parent_error.is_finite() {
                level.far_reach = level.far_reach.max(c.parent_bounds.center.length() + c.parent_bounds.radius);
            }
        }
        levels
    }
}

impl ClusterMesh {
    /// The clusters `view` draws: each whose error is within the budget and whose parent's is
    /// over it. Every point of the mesh is in exactly one (no holes, no overlaps).
    pub fn select(&self, view: &LodView) -> Vec<usize> {
        (0..self.clusters.len())
            .filter(|&i| {
                let c = &self.clusters[i];
                projected_error(c.error, c.lod_bounds, view) <= view.threshold && projected_error(c.parent_error, c.parent_bounds, view) > view.threshold
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod card_tests;
