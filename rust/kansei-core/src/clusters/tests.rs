use super::*;
use glam::Vec3;

#[test]
fn an_enclosing_sphere_contains_every_sphere() {
    let spheres = [
        Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
        Sphere { center: Vec3::new(3.0, 0.0, 0.0), radius: 0.5 },
        Sphere { center: Vec3::new(0.0, -2.0, 1.0), radius: 2.0 },
        Sphere { center: Vec3::new(0.5, 0.0, 0.0), radius: 0.1 },
    ];
    let s = Sphere::enclosing(spheres);
    for o in &spheres {
        assert!(s.contains(o), "{s:?} misses {o:?}");
    }
    // one inside another: the outer one
    let inner = Sphere { center: Vec3::ZERO, radius: 0.5 };
    let outer = Sphere { center: Vec3::new(0.1, 0.0, 0.0), radius: 2.0 };
    assert_eq!(Sphere::enclosing([inner, outer]), outer);
}
