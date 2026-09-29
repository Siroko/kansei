use glam::{Quat, Vec3};

use super::*;

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 100_003) as f32 / 100_003.0
}

fn random_unit(i: u32) -> Vec3 {
    Vec3::new(hash(i) - 0.5, hash(i + 1) - 0.5, hash(i + 2) - 0.5).normalize_or(Vec3::X)
}

/// The box as 12 triangles.
fn box_mesh(b: &Obb) -> TriangleMesh {
    let corner = |i: usize| b.center + b.rotation * (b.half_extents * Vec3::new(if i & 1 == 0 { -1.0 } else { 1.0 }, if i & 2 == 0 { -1.0 } else { 1.0 }, if i & 4 == 0 { -1.0 } else { 1.0 }));
    let quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
    TriangleMesh::new(quads.iter().flat_map(|q| [[corner(q[0]), corner(q[1]), corner(q[2])], [corner(q[0]), corner(q[2]), corner(q[3])]]).collect())
}

fn tilted_box() -> Obb {
    Obb::new(Vec3::new(0.3, 0.8, -0.2), Vec3::new(1.0, 0.5, 0.7), Quat::from_euler(glam::EulerRot::YXZ, 0.7, 0.3, -0.2))
}

#[test]
fn rays_hit_boxes_where_the_faces_are() {
    let b = Obb::from_min_max(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
    let (t, n) = b.raycast(Vec3::new(-5.0, 0.5, 0.0), Vec3::X, 10.0).unwrap();
    assert!((t - 4.0).abs() < 1e-6 && n == Vec3::NEG_X, "{t} {n}");
    assert!(b.raycast(Vec3::new(-5.0, 2.5, 0.0), Vec3::X, 10.0).is_none());
    assert!(b.raycast(Vec3::new(-5.0, 0.5, 0.0), Vec3::X, 3.0).is_none());
    assert_eq!(b.raycast(Vec3::new(0.0, 0.5, 0.0), Vec3::X, 3.0).unwrap().0, 0.0);
    let (t, n) = b.raycast(Vec3::new(0.2, 5.0, 0.3), Vec3::NEG_Y, 10.0).unwrap();
    assert!((t - 3.0).abs() < 1e-6 && n == Vec3::Y);
    // a turned box agrees with its triangles
    let b = tilted_box();
    let mesh = box_mesh(&b);
    let mut hits = 0;
    for i in 0..300 {
        let origin = b.center + random_unit(i * 7) * 4.0;
        let direction = (b.center + random_unit(i * 7 + 3) * 0.8 - origin).normalize();
        let (x, y) = (b.raycast(origin, direction, 10.0), mesh.raycast(origin, direction, 10.0));
        assert_eq!(x.is_some(), y.is_some(), "ray {i}");
        if let (Some(x), Some(y)) = (x, y) {
            assert!((x.0 - y.0).abs() < 1e-4 && x.1.dot(y.1) > 0.999, "ray {i}: {x:?} {y:?}");
            hits += 1;
        }
    }
    assert!(hits > 100);
}

/// The first distance at which a sphere moving along a ray touches the box, by small steps.
fn sampled_sphere_cast(b: &Obb, o: Vec3, r: f32, d: Vec3, max: f32) -> Option<f32> {
    let step = 1e-3;
    (0..=(max / step) as usize).map(|k| k as f32 * step).find(|t| b.closest_point(o + d * t).distance(o + d * t) <= r)
}

#[test]
fn swept_spheres_touch_boxes_at_the_first_contact() {
    let b = Obb::from_min_max(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
    let (t, n) = b.sphere_cast(Vec3::new(-5.0, 1.0, 0.0), 0.5, Vec3::X, 10.0).unwrap();
    assert!((t - 3.5).abs() < 1e-5 && n.abs_diff_eq(Vec3::NEG_X, 1e-5), "{t} {n}");
    // grazing a vertical edge: the rounded corner, not the grown box's corner
    let o = Vec3::new(-5.0, 1.0, -1.4);
    let (t, n) = b.sphere_cast(o, 0.5, Vec3::X, 10.0).unwrap();
    let expected = 4.0 - (0.25f32 - 0.16).sqrt();
    assert!((t - expected).abs() < 1e-4, "{t} vs {expected}");
    assert!(n.x < 0.0 && n.z < 0.0);
    // starting in contact
    assert_eq!(b.sphere_cast(Vec3::new(-1.2, 1.0, 0.0), 0.5, Vec3::X, 10.0).unwrap().0, 0.0);
    // random sweeps against a turned box, against small steps and against its triangles
    let b = tilted_box();
    let mesh = box_mesh(&b);
    for i in 0..120 {
        let origin = b.center + random_unit(i * 11) * 3.5;
        let direction = (b.center + random_unit(i * 11 + 5) * 1.2 - origin).normalize();
        let r = 0.1 + hash(i * 3) * 0.5;
        let cast = b.sphere_cast(origin, r, direction, 8.0).map(|h| h.0);
        let sampled = sampled_sphere_cast(&b, origin, r, direction, 8.0);
        match (cast, sampled) {
            (Some(x), Some(y)) => assert!((x - y).abs() < 3e-3, "sweep {i}: {x} vs {y}"),
            (None, None) => {}
            other => panic!("sweep {i}: {other:?}"),
        }
        let m = mesh.sphere_cast(origin, r, direction, 8.0).map(|h| h.0);
        assert!(cast.zip(m).is_none_or(|(x, y)| (x - y).abs() < 1e-3) && cast.is_some() == m.is_some(), "sweep {i}: box {cast:?} mesh {m:?}");
    }
}

#[test]
fn triangles_closest_points_are_the_nearest_samples() {
    let t = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.3, 0.0), Vec3::new(0.5, 0.1, 1.5)];
    for i in 0..200 {
        let p = random_unit(i * 5) * 3.0;
        let (q, _) = closest_point_triangle(p, &t);
        let mut best = f32::MAX;
        for a in 0..=60 {
            for b in 0..=(60 - a) {
                let (u, v) = (a as f32 / 60.0, b as f32 / 60.0);
                let s = t[0] + (t[1] - t[0]) * u + (t[2] - t[0]) * v;
                best = best.min(s.distance(p));
            }
        }
        assert!(q.distance(p) <= best + 1e-5 && q.distance(p) > best - 0.03, "{p}: {} vs {best}", q.distance(p));
    }
    assert!(ray_triangle(Vec3::new(0.5, 5.0, 0.3), Vec3::NEG_Y, &t).is_some());
    assert!(ray_triangle(Vec3::new(3.0, 5.0, 3.0), Vec3::NEG_Y, &t).is_none());
}

#[test]
fn spheres_are_pushed_out_to_touching() {
    let b = tilted_box();
    for i in 0..100 {
        let center = b.center + random_unit(i * 13) * (0.2 + hash(i) * 1.5);
        let r = 0.3;
        let push = b.sphere_penetration(center, r);
        let moved = center + push;
        let gap = b.closest_point(moved).distance(moved);
        if push != Vec3::ZERO {
            assert!((gap - r).abs() < 1e-4, "{i}: {gap}");
        } else {
            assert!(b.closest_point(center).distance(center) >= r - 1e-6);
        }
    }
}

#[test]
fn a_world_answers_casts_overlaps_ground_and_capsules() {
    let mut world = CollisionWorld::new();
    // a floor slab, a 1 m box and a 0.2 m curb
    world.add_box(Obb::from_min_max(Vec3::new(-50.0, -1.0, -50.0), Vec3::new(50.0, 0.0, 50.0)));
    let crate_index = world.add_box(Obb::from_min_max(Vec3::new(2.0, 0.0, -0.5), Vec3::new(3.0, 1.0, 0.5)));
    world.add_box(Obb::from_min_max(Vec3::new(-3.0, 0.0, -1.0), Vec3::new(-2.0, 0.2, 1.0)));
    let tri = world.add(Shape::Mesh(TriangleMesh::new(vec![[Vec3::new(10.0, 0.0, -1.0), Vec3::new(10.0, 3.0, 0.0), Vec3::new(10.0, 0.0, 1.0)]])), 2);

    let hit = world.raycast(Vec3::new(0.0, 0.5, 0.0), Vec3::X, 20.0, u32::MAX).unwrap();
    assert_eq!(hit.collider, crate_index);
    assert!(hit.point.abs_diff_eq(Vec3::new(2.0, 0.5, 0.0), 1e-5) && hit.normal == Vec3::NEG_X);
    // the mesh sits on layer 2: a layer-1 ray above the crate misses it
    assert!(world.raycast(Vec3::new(0.0, 1.5, 0.0), Vec3::X, 20.0, 1).is_none());
    assert_eq!(world.raycast(Vec3::new(0.0, 1.5, 0.0), Vec3::X, 20.0, 2).unwrap().collider, tri);
    let hit = world.sphere_cast(Vec3::new(0.0, 0.5, 0.0), 0.3, Vec3::X, 20.0, u32::MAX).unwrap();
    assert!((hit.distance - 1.7).abs() < 1e-5 && hit.point.abs_diff_eq(Vec3::new(1.7, 0.5, 0.0), 1e-5));

    assert_eq!(world.ground_height(Vec3::new(2.5, 0.0, 0.0), 2.0, 2.0, u32::MAX), Some(1.0));
    assert_eq!(world.ground_height(Vec3::new(0.0, 0.0, 0.0), 2.0, 2.0, u32::MAX), Some(0.0));
    assert!(world.overlap_sphere(Vec3::new(1.9, 0.5, 0.0), 0.2, u32::MAX));
    assert!(!world.overlap_capsule(Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.0, 1.5, 0.0), 0.3, u32::MAX));

    // a capsule half into the crate's side moves out sideways; the curb is under the step
    let out = world.resolve_capsule(Vec3::new(1.9, 0.0, 0.1), 1.8, 0.3, 0.3, u32::MAX);
    assert!((out.x - 1.7).abs() < 1e-3 && (out.z - 0.1).abs() < 1e-3 && out.y == 0.0, "{out}");
    let on_curb = world.resolve_capsule(Vec3::new(-2.5, 0.0, 0.0), 1.8, 0.3, 0.3, u32::MAX);
    assert_eq!(on_curb, Vec3::new(-2.5, 0.0, 0.0));
}
