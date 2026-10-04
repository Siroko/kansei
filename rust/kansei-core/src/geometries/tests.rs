use super::*;
use glam::{Mat4, Vec3};

fn position(g: &Geometry, i: u32) -> Vec3 {
    let p = g.vertices[i as usize].position;
    Vec3::new(p[0], p[1], p[2])
}

/// Every triangle's counter-clockwise normal agrees with its vertices' normals (front faces
/// outward), and every normal is unit length.
fn assert_wound_outward(g: &Geometry) {
    for t in g.indices.chunks_exact(3) {
        let (a, b, c) = (position(g, t[0]), position(g, t[1]), position(g, t[2]));
        let face = (b - a).cross(c - a);
        // a pole's collapsed triangles have no direction
        if face.length() < 1e-6 {
            continue;
        }
        let normal: Vec3 = t.iter().map(|&i| Vec3::from(g.vertices[i as usize].normal)).sum();
        assert!(face.dot(normal) > 0.0, "{}: triangle {t:?} faces against its normals", g.label);
    }
    for v in &g.vertices {
        assert!((Vec3::from(v.normal).length() - 1.0).abs() < 1e-4, "{}: normal {:?}", g.label, v.normal);
    }
}

#[test]
fn a_heightfield_follows_its_function_and_faces_up() {
    let g = HeightfieldGeometry::new([-2.0, -1.0], [2.0, 3.0], (8, 4), |x, _| 0.5 * x);
    assert_eq!((g.vertices.len(), g.indices.len()), (9 * 5, 8 * 4 * 6));
    for v in &g.vertices {
        assert!((v.position[1] - 0.5 * v.position[0]).abs() < 1e-5);
        assert!(Vec3::from(v.normal).abs_diff_eq(Vec3::new(-0.5, 1.0, 0.0).normalize(), 1e-4));
    }
    assert_wound_outward(&g);
    assert_eq!(g.bounds(), (Vec3::new(-2.0, -1.0, -1.0), Vec3::new(2.0, 1.0, 3.0)));
}

#[test]
fn cylinders_and_cones_are_closed_and_face_out() {
    let cylinder = CylinderGeometry::new(1.0, 0.5, 2.0, 12, 3);
    assert_wound_outward(&cylinder);
    assert_eq!(cylinder.bounds().0.y, 0.0);
    assert!((cylinder.bounds().1.y - 2.0).abs() < 1e-6);
    let cone = CylinderGeometry::new(0.8, 0.0, 1.5, 9, 2);
    assert_wound_outward(&cone);
    // a cone has no top cap: only the bottom one points down
    assert!(cone.vertices.iter().all(|v| v.normal[1] > -1.0 + 1e-3 || v.position[1] == 0.0));
}

#[test]
fn an_icosphere_has_twenty_times_four_to_the_n_faces_on_its_radius() {
    for n in 0..3 {
        let g = IcosphereGeometry::new(2.5, n);
        assert_eq!(g.indices.len() / 3, 20 * 4usize.pow(n));
        assert!(g.vertices.iter().all(|v| (Vec3::new(v.position[0], v.position[1], v.position[2]).length() - 2.5).abs() < 1e-4));
        assert_wound_outward(&g);
    }
}

#[test]
fn merging_moves_each_part_and_fitting_stands_it_on_the_ground() {
    let a = BoxGeometry::new(1.0, 1.0, 1.0);
    let b = SphereGeometry::new(0.5, 8, 6);
    let merged = Geometry::merged("Both", &[(&a, Mat4::from_translation(Vec3::new(-2.0, 0.0, 0.0))), (&b, Mat4::from_scale_rotation_translation(Vec3::splat(2.0), glam::Quat::from_rotation_y(0.7), Vec3::new(3.0, 1.0, 0.0)))]);
    assert_eq!(merged.vertices.len(), a.vertices.len() + b.vertices.len());
    assert_eq!(merged.indices.len(), a.indices.len() + b.indices.len());
    assert!(merged.indices[a.indices.len()..].iter().all(|&i| i as usize >= a.vertices.len()));
    let (lo, hi) = merged.bounds();
    // the sphere is tessellated: its extremes fall a little inside its radius
    assert!(lo.abs_diff_eq(Vec3::new(-2.5, -0.5, -1.0), 0.01) && (hi.x - 4.0).abs() < 0.01, "{lo} {hi}");
    assert_wound_outward(&merged);

    let fitted = merged.fit(Vec3::new(1.0, f32::INFINITY, 1.0));
    let (lo, hi) = fitted.bounds();
    assert!((hi.x - lo.x - 1.0).abs() < 1e-4 && lo.y.abs() < 1e-5, "{lo} {hi}");
    assert!(((lo.x + hi.x) * 0.5).abs() < 1e-5 && ((lo.z + hi.z) * 0.5).abs() < 1e-5);
}

