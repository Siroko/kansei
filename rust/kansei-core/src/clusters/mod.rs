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

#[cfg(test)]
mod tests;
