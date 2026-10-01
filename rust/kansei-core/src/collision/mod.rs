//! A minimal CPU collision world: static boxes and triangle meshes, with ray and sphere casts,
//! overlap tests and capsule push-out.
//!
//! Enough for a character on a course of obstacles: ledge detection casts rays and spheres at
//! them, a character capsule is kept out of them, and ground height is a ray down. The shapes and
//! queries are the textbook ones (Ericson, "Real-Time Collision Detection", 2005: slab tests,
//! Möller–Trumbore, closest points on boxes and triangles, rays against capsules for swept
//! spheres). Meshes are tested triangle by triangle behind their bounds; a BVH can come when
//! scenes need it.

use glam::{Quat, Vec3};

#[cfg(test)]
mod tests;

/// An oriented box: centre, half extents along its axes, and the rotation of its axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obb {
    pub center: Vec3,
    pub half_extents: Vec3,
    pub rotation: Quat,
}

impl Obb {
    pub fn new(center: Vec3, half_extents: Vec3, rotation: Quat) -> Self {
        Self { center, half_extents, rotation }
    }

    /// An axis-aligned box from its corners.
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self { center: (min + max) * 0.5, half_extents: (max - min) * 0.5, rotation: Quat::IDENTITY }
    }

    fn local_point(&self, p: Vec3) -> Vec3 {
        self.rotation.conjugate() * (p - self.center)
    }

    fn world_point(&self, p: Vec3) -> Vec3 {
        self.center + self.rotation * p
    }

    /// The point of the box nearest `p` (`p` itself inside).
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        self.world_point(self.local_point(p).clamp(-self.half_extents, self.half_extents))
    }

    pub fn contains(&self, p: Vec3) -> bool {
        self.local_point(p).abs().cmple(self.half_extents).all()
    }

    /// World-space bounds.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let axes = [self.rotation * Vec3::X, self.rotation * Vec3::Y, self.rotation * Vec3::Z];
        let reach = axes[0].abs() * self.half_extents.x + axes[1].abs() * self.half_extents.y + axes[2].abs() * self.half_extents.z;
        (self.center - reach, self.center + reach)
    }

    /// First hit of a ray (unit `direction`) within `max`: distance and outward normal.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec3)> {
        let (o, d) = (self.local_point(origin), self.rotation.conjugate() * direction);
        let (t, axis) = slab(o, d, self.half_extents, max)?;
        Some((t, self.rotation * axis))
    }

    /// First contact of a sphere of `radius` moving from `origin` along unit `direction` within
    /// `max`: distance and the normal at the contact (from the box toward the sphere). A sphere
    /// that starts overlapping hits at 0.
    pub fn sphere_cast(&self, origin: Vec3, radius: f32, direction: Vec3, max: f32) -> Option<(f32, Vec3)> {
        let (o, d) = (self.local_point(origin), self.rotation.conjugate() * direction);
        let e = self.half_extents;
        let inside = |p: Vec3| p.clamp(-e, e);
        if (o - inside(o)).length_squared() <= radius * radius {
            return Some((0.0, self.rotation * push_normal(o, e)));
        }
        // the box grown by the radius: its faces are exact, its edges and corners are rounded
        let (t, axis) = slab(o, d, e + Vec3::splat(radius), max)?;
        let p = o + d * t;
        let outside = (p.abs() - e).cmpgt(Vec3::splat(1e-5));
        let t = if (outside.bitmask().count_ones()) <= 1 {
            Some((t, axis))
        } else {
            // an edge or corner region: the ray against the capsules round the 12 edges
            let corner = |i: u32| Vec3::new(if i & 1 == 0 { -e.x } else { e.x }, if i & 2 == 0 { -e.y } else { e.y }, if i & 4 == 0 { -e.z } else { e.z });
            let mut best: Option<f32> = None;
            for a in 0..8u32 {
                for bit in [1u32, 2, 4] {
                    if a & bit == 0 {
                        if let Some(t) = ray_capsule(o, d, corner(a), corner(a | bit), radius) {
                            if t <= max && best.is_none_or(|b| t < b) {
                                best = Some(t);
                            }
                        }
                    }
                }
            }
            best.map(|t| {
                let c = o + d * t;
                (t, (c - inside(c)).normalize_or_zero())
            })
        }?;
        Some((t.0, self.rotation * t.1))
    }

    /// Push that moves a sphere at `center` out of the box (zero when apart).
    pub fn sphere_penetration(&self, center: Vec3, radius: f32) -> Vec3 {
        let o = self.local_point(center);
        let e = self.half_extents;
        let q = o.clamp(-e, e);
        let gap = o - q;
        let distance = gap.length();
        if distance > radius {
            return Vec3::ZERO;
        }
        if distance > 1e-6 {
            return self.rotation * (gap / distance * (radius - distance));
        }
        // the centre is inside: out through the nearest face
        let depth = e - o.abs();
        let axis = min_axis(depth);
        let mut push = Vec3::ZERO;
        push[axis] = if o[axis] < 0.0 { -(depth[axis] + radius) } else { depth[axis] + radius };
        self.rotation * push
    }
}

/// Index of the smallest component.
fn min_axis(v: Vec3) -> usize {
    if v.x <= v.y && v.x <= v.z { 0 } else if v.y <= v.z { 1 } else { 2 }
}

/// The outward face normal (box space) of the face nearest a point inside or on the box.
fn push_normal(o: Vec3, e: Vec3) -> Vec3 {
    let q = o.clamp(-e, e);
    let gap = o - q;
    if gap.length_squared() > 1e-12 {
        return gap.normalize();
    }
    let depth = e - o.abs();
    let axis = min_axis(depth);
    let mut n = Vec3::ZERO;
    n[axis] = if o[axis] < 0.0 { -1.0 } else { 1.0 };
    n
}

/// A ray against the axis-aligned box [-e, e]: entry distance within `max` and the entry face's
/// normal; a ray starting inside enters at 0 (normal zero).
fn slab(o: Vec3, d: Vec3, e: Vec3, max: f32) -> Option<(f32, Vec3)> {
    let (mut t0, mut t1) = (f32::NEG_INFINITY, f32::INFINITY);
    let mut axis = Vec3::ZERO;
    for i in 0..3 {
        if d[i].abs() < 1e-12 {
            if o[i].abs() > e[i] {
                return None;
            }
            continue;
        }
        let (a, b) = ((-e[i] - o[i]) / d[i], (e[i] - o[i]) / d[i]);
        let (near, far) = if a < b { (a, b) } else { (b, a) };
        if near > t0 {
            t0 = near;
            axis = Vec3::ZERO;
            axis[i] = -d[i].signum();
        }
        t1 = t1.min(far);
        if t0 > t1 {
            return None;
        }
    }
    if t1 < 0.0 || t0 > max {
        return None;
    }
    Some(if t0 < 0.0 { (0.0, Vec3::ZERO) } else { (t0, axis) })
}

/// First distance along a ray (unit `d`) at which it is within `r` of the segment `a`-`b`.
pub fn ray_capsule(o: Vec3, d: Vec3, a: Vec3, b: Vec3, r: f32) -> Option<f32> {
    let ab = b - a;
    let ao = o - a;
    let (ab_ab, ab_d, ab_ao) = (ab.dot(ab), ab.dot(d), ab.dot(ao));
    let mut best: Option<f32> = None;
    // the cylinder's side
    let qa = ab_ab - ab_d * ab_d;
    let qb = ab_ab * ao.dot(d) - ab_ao * ab_d;
    let qc = ab_ab * ao.dot(ao) - ab_ao * ab_ao - r * r * ab_ab;
    if qa.abs() > 1e-12 {
        let disc = qb * qb - qa * qc;
        if disc >= 0.0 {
            let t = (-qb - disc.sqrt()) / qa;
            let s = ab_ao + t * ab_d;
            if t >= 0.0 && s >= 0.0 && s <= ab_ab {
                best = Some(t);
            }
        }
    }
    // the end caps
    for c in [a, b] {
        if let Some(t) = ray_sphere(o, d, c, r) {
            if best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        }
    }
    best
}

/// First distance along a ray (unit `d`) at which it enters the sphere (`None` behind or missed).
pub fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let m = o - c;
    let b = m.dot(d);
    let cc = m.dot(m) - r * r;
    if cc > 0.0 && b > 0.0 {
        return None;
    }
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    Some((-b - disc.sqrt()).max(0.0))
}

/// A triangle soup with its bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleMesh {
    pub triangles: Vec<[Vec3; 3]>,
    min: Vec3,
    max: Vec3,
}

impl TriangleMesh {
    pub fn new(triangles: Vec<[Vec3; 3]>) -> Self {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in &triangles {
            for p in t {
                min = min.min(*p);
                max = max.max(*p);
            }
        }
        Self { triangles, min, max }
    }

    /// From indexed positions, transformed by `transform`.
    pub fn from_indexed(positions: &[Vec3], indices: &[u32], transform: glam::Mat4) -> Self {
        Self::new(indices.chunks_exact(3).map(|t| [0, 1, 2].map(|k| transform.transform_point3(positions[t[k] as usize]))).collect())
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.min, self.max)
    }

    /// Whether a ray (or a sphere of `pad` radius along it) could touch the bounds within `max`.
    fn may_hit(&self, o: Vec3, d: Vec3, max: f32, pad: f32) -> bool {
        let e = (self.max - self.min) * 0.5 + Vec3::splat(pad);
        slab(o - (self.min + self.max) * 0.5, d, e, max).is_some()
    }

    pub fn raycast(&self, o: Vec3, d: Vec3, max: f32) -> Option<(f32, Vec3)> {
        if !self.may_hit(o, d, max, 0.0) {
            return None;
        }
        let mut best: Option<(f32, Vec3)> = None;
        for t in &self.triangles {
            if let Some(dist) = ray_triangle(o, d, t) {
                if dist <= max && best.is_none_or(|b| dist < b.0) {
                    let mut n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
                    if n.dot(d) > 0.0 {
                        n = -n;
                    }
                    best = Some((dist, n));
                }
            }
        }
        best
    }

    pub fn sphere_cast(&self, o: Vec3, r: f32, d: Vec3, max: f32) -> Option<(f32, Vec3)> {
        if !self.may_hit(o, d, max, r) {
            return None;
        }
        let mut best: Option<f32> = None;
        for t in &self.triangles {
            let (closest, _) = closest_point_triangle(o, t);
            if closest.distance_squared(o) <= r * r {
                best = Some(0.0);
                break;
            }
            let mut hit = None;
            // the face, offset toward the sphere
            let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            let n = if n.dot(o - t[0]) < 0.0 { -n } else { n };
            let denom = n.dot(d);
            if denom < -1e-9 {
                let dist = (r - n.dot(o - t[0])) / denom;
                if dist >= 0.0 {
                    let p = o + d * dist - n * r;
                    if closest_point_triangle(p, t).0.distance_squared(p) < 1e-10 {
                        hit = Some(dist);
                    }
                }
            }
            // the edges (and so the corners)
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if let Some(dist) = ray_capsule(o, d, a, b, r) {
                    if hit.is_none_or(|h| dist < h) {
                        hit = Some(dist);
                    }
                }
            }
            if let Some(dist) = hit {
                if dist <= max && best.is_none_or(|b| dist < b) {
                    best = Some(dist);
                }
            }
        }
        best.map(|dist| {
            let c = o + d * dist;
            let nearest = self.triangles.iter().map(|t| closest_point_triangle(c, t).0).min_by(|a, b| a.distance_squared(c).total_cmp(&b.distance_squared(c))).unwrap_or(c);
            (dist, (c - nearest).normalize_or_zero())
        })
    }

    pub fn sphere_penetration(&self, center: Vec3, radius: f32) -> Vec3 {
        let mut push = Vec3::ZERO;
        for t in &self.triangles {
            let (q, _) = closest_point_triangle(center + push, t);
            let gap = center + push - q;
            let distance = gap.length();
            if distance < radius && distance > 1e-6 {
                push += gap / distance * (radius - distance);
            }
        }
        push
    }
}

/// Distance along a ray (unit `d`) to a triangle (Möller–Trumbore), either side.
pub fn ray_triangle(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> Option<f32> {
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - t[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let dist = e2.dot(q) * inv;
    (dist >= 0.0).then_some(dist)
}

/// The point of a triangle nearest `p`, and whether it is inside the face (not on an edge).
pub fn closest_point_triangle(p: Vec3, t: &[Vec3; 3]) -> (Vec3, bool) {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, false);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, false);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3)), false);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, false);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6)), false);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))), false);
    }
    let denom = 1.0 / (va + vb + vc);
    (a + ab * (vb * denom) + ac * (vc * denom), true)
}

/// A collider's shape.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Box(Obb),
    Mesh(TriangleMesh),
}

/// A static shape on some layers (a bit mask queries filter on).
#[derive(Debug, Clone, PartialEq)]
pub struct Collider {
    pub shape: Shape,
    pub layers: u32,
}

/// The first thing a cast touched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub distance: f32,
    /// Where the ray is, or the sphere's centre, at contact.
    pub point: Vec3,
    /// The surface normal at the contact, toward the caster.
    pub normal: Vec3,
    pub collider: usize,
}

/// Static colliders and the queries on them.
#[derive(Debug, Clone, Default)]
pub struct CollisionWorld {
    colliders: Vec<Collider>,
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a collider; returns its index.
    pub fn add(&mut self, shape: Shape, layers: u32) -> usize {
        self.colliders.push(Collider { shape, layers });
        self.colliders.len() - 1
    }

    pub fn add_box(&mut self, obb: Obb) -> usize {
        self.add(Shape::Box(obb), 1)
    }

    pub fn colliders(&self) -> &[Collider] {
        &self.colliders
    }

    fn nearest(&self, layers: u32, mut cast: impl FnMut(&Shape) -> Option<(f32, Vec3)>) -> Option<(f32, Vec3, usize)> {
        let mut best: Option<(f32, Vec3, usize)> = None;
        for (i, c) in self.colliders.iter().enumerate() {
            if c.layers & layers == 0 {
                continue;
            }
            if let Some((t, n)) = cast(&c.shape) {
                if best.is_none_or(|b| t < b.0) {
                    best = Some((t, n, i));
                }
            }
        }
        best
    }

    /// The first surface a ray (unit `direction`) meets within `max`.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, max: f32, layers: u32) -> Option<Hit> {
        self.nearest(layers, |s| match s {
            Shape::Box(b) => b.raycast(origin, direction, max),
            Shape::Mesh(m) => m.raycast(origin, direction, max),
        })
        .map(|(t, n, collider)| Hit { distance: t, point: origin + direction * t, normal: n, collider })
    }

    /// The first contact of a sphere swept along unit `direction` within `max`.
    pub fn sphere_cast(&self, origin: Vec3, radius: f32, direction: Vec3, max: f32, layers: u32) -> Option<Hit> {
        self.nearest(layers, |s| match s {
            Shape::Box(b) => b.sphere_cast(origin, radius, direction, max),
            Shape::Mesh(m) => m.sphere_cast(origin, radius, direction, max),
        })
        .map(|(t, n, collider)| Hit { distance: t, point: origin + direction * t, normal: n, collider })
    }

    /// Whether a sphere touches anything.
    pub fn overlap_sphere(&self, center: Vec3, radius: f32, layers: u32) -> bool {
        self.colliders.iter().filter(|c| c.layers & layers != 0).any(|c| match &c.shape {
            Shape::Box(b) => b.closest_point(center).distance_squared(center) < radius * radius,
            Shape::Mesh(m) => m.triangles.iter().any(|t| closest_point_triangle(center, t).0.distance_squared(center) < radius * radius),
        })
    }

    /// Whether a capsule (segment `a`-`b`, `radius`) touches anything: spheres along it, spaced
    /// half a radius apart.
    pub fn overlap_capsule(&self, a: Vec3, b: Vec3, radius: f32, layers: u32) -> bool {
        let steps = ((a.distance(b) / (radius * 0.5)).ceil() as usize).max(1);
        (0..=steps).any(|k| self.overlap_sphere(a.lerp(b, k as f32 / steps as f32), radius, layers))
    }

    /// Where a vertical capsule standing at `feet` (`height` tall, `radius` wide) must move to
    /// stop overlapping the world, sideways only; its lowest `step` metres are free (steps and
    /// curbs don't block). A few iterations of pushing its spheres out.
    pub fn resolve_capsule(&self, feet: Vec3, height: f32, radius: f32, step: f32, layers: u32) -> Vec3 {
        let mut position = feet;
        let bottom = step + radius;
        let top = (height - radius).max(bottom);
        let steps = (((top - bottom) / (radius * 0.5)).ceil() as usize).max(1);
        for _ in 0..4 {
            let mut push = Vec3::ZERO;
            for k in 0..=steps {
                let center = position + Vec3::Y * (bottom + (top - bottom) * k as f32 / steps as f32);
                for c in self.colliders.iter().filter(|c| c.layers & layers != 0) {
                    let p = match &c.shape {
                        Shape::Box(b) => b.sphere_penetration(center, radius),
                        Shape::Mesh(m) => m.sphere_penetration(center, radius),
                    };
                    let p = Vec3::new(p.x, 0.0, p.z);
                    if p.length_squared() > push.length_squared() {
                        push = p;
                    }
                }
            }
            if push.length_squared() < 1e-10 {
                break;
            }
            position += push;
        }
        position
    }

    /// The height of the first surface below `position`, looking from `above` metres over it down
    /// to `below` metres under it.
    pub fn ground_height(&self, position: Vec3, above: f32, below: f32, layers: u32) -> Option<f32> {
        self.raycast(position + Vec3::Y * above, Vec3::NEG_Y, above + below, layers).map(|h| h.point.y)
    }
}
