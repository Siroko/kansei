use glam::{Mat4, Quat, Vec3};

/// A joint transform: scale, then rotation, then translation (glTF's TRS).
///
/// Composition (`mul`) is exact for uniform scales, which skeletons use; a non-uniform scale
/// under a rotated child would shear, which a TRS cannot hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE };

    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { translation, rotation, scale }
    }

    pub fn from_translation_rotation(translation: Vec3, rotation: Quat) -> Self {
        Self { translation, rotation, scale: Vec3::ONE }
    }

    /// The transform of a matrix without shear.
    pub fn from_mat4(m: &Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self { translation, rotation: rotation.normalize(), scale }
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// `self` applied after `child`: a child's local transform into its parent's space.
    pub fn mul(&self, child: &Transform) -> Transform {
        Transform {
            translation: self.transform_point(child.translation),
            rotation: (self.rotation * child.rotation).normalize(),
            scale: self.scale * child.scale,
        }
    }

    pub fn inverse(&self) -> Transform {
        let rotation = self.rotation.conjugate();
        let scale = self.scale.recip();
        Transform { translation: -(scale * (rotation * self.translation)), rotation, scale }
    }

    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.translation + self.rotation * (self.scale * p)
    }

    pub fn transform_vector(&self, v: Vec3) -> Vec3 {
        self.rotation * (self.scale * v)
    }

    /// Linear interpolation of translation and scale, normalized lerp of rotation along the
    /// shorter arc.
    pub fn lerp(&self, other: &Transform, t: f32) -> Transform {
        Transform {
            translation: self.translation.lerp(other.translation, t),
            rotation: nlerp(self.rotation, other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

/// Normalized lerp between two rotations along the shorter arc. Close to slerp for the small
/// steps between animation frames, and cheaper.
pub fn nlerp(a: Quat, b: Quat, t: f32) -> Quat {
    let b = if a.dot(b) < 0.0 { -b } else { b };
    (a * (1.0 - t) + b * t).normalize()
}

/// `q` or `-q`, whichever has a non-negative w: the same rotation, on the hemisphere where
/// `quat_log` gives the shorter rotation vector.
pub fn quat_abs(q: Quat) -> Quat {
    if q.w < 0.0 { -q } else { q }
}

/// The rotation vector (axis times half angle) of a unit quaternion: the inverse of `quat_exp`.
pub fn quat_log(q: Quat) -> Vec3 {
    let v = Vec3::new(q.x, q.y, q.z);
    let length = v.length();
    if length < 1e-8 {
        v
    } else {
        let half_angle = length.atan2(q.w.clamp(-1.0, 1.0));
        v * (half_angle / length)
    }
}

/// The unit quaternion of a rotation vector (axis times half angle).
pub fn quat_exp(v: Vec3) -> Quat {
    let half_angle = v.length();
    if half_angle < 1e-8 {
        Quat::from_xyzw(v.x, v.y, v.z, 1.0).normalize()
    } else {
        let s = half_angle.sin() / half_angle;
        Quat::from_xyzw(v.x * s, v.y * s, v.z * s, half_angle.cos())
    }
}

/// Axis times angle of a rotation, along the shorter arc.
pub fn quat_to_scaled_angle_axis(q: Quat) -> Vec3 {
    2.0 * quat_log(quat_abs(q))
}

/// The rotation of axis times angle `v`.
pub fn quat_from_scaled_angle_axis(v: Vec3) -> Quat {
    quat_exp(v * 0.5)
}

/// Angular velocity (radians per second, world axes) that turns `from` into `to` in `dt`.
pub fn angular_velocity(from: Quat, to: Quat, dt: f32) -> Vec3 {
    quat_to_scaled_angle_axis(to * from.conjugate()) / dt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).abs().max_element() < eps
    }

    fn same_rotation(a: Quat, b: Quat, eps: f32) -> bool {
        a.dot(b).abs() > 1.0 - eps
    }

    #[test]
    fn composition_matches_the_matrices() {
        let parent = Transform::new(Vec3::new(1.0, 2.0, -3.0), Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.3, 1.1), Vec3::splat(2.0));
        let child = Transform::new(Vec3::new(-0.5, 0.25, 4.0), Quat::from_euler(glam::EulerRot::YXZ, -1.2, 0.4, 0.2), Vec3::splat(0.5));
        let composed = parent.mul(&child).to_mat4();
        let expected = parent.to_mat4() * child.to_mat4();
        assert!(composed.abs_diff_eq(expected, 1e-5), "{composed} vs {expected}");
        let p = Vec3::new(0.3, -0.7, 1.9);
        assert!(close(parent.mul(&child).transform_point(p), parent.transform_point(child.transform_point(p)), 1e-5));
    }

    #[test]
    fn the_inverse_undoes_the_transform() {
        let t = Transform::new(Vec3::new(3.0, -1.0, 0.5), Quat::from_rotation_z(0.9) * Quat::from_rotation_x(-0.4), Vec3::splat(1.5));
        let identity = t.mul(&t.inverse());
        assert!(close(identity.translation, Vec3::ZERO, 1e-5));
        assert!(same_rotation(identity.rotation, Quat::IDENTITY, 1e-6));
        assert!(close(identity.scale, Vec3::ONE, 1e-6));
        assert!(t.inverse().to_mat4().abs_diff_eq(t.to_mat4().inverse(), 1e-5));
    }

    #[test]
    fn nlerp_takes_the_shorter_arc() {
        let a = Quat::from_rotation_y(0.2);
        let b = -Quat::from_rotation_y(0.6); // same rotation, other hemisphere
        let mid = nlerp(a, b, 0.5);
        assert!(same_rotation(mid, Quat::from_rotation_y(0.4), 1e-5), "{mid}");
        assert!(same_rotation(nlerp(a, b, 0.0), a, 1e-6));
        assert!(same_rotation(nlerp(a, b, 1.0), b, 1e-6));
    }

    #[test]
    fn exp_and_log_round_trip() {
        for v in [Vec3::new(0.1, -0.2, 0.3), Vec3::new(1.2, 0.0, 0.0), Vec3::ZERO, Vec3::new(0.0, 0.0, -1.5)] {
            let q = quat_exp(v);
            assert!((q.length() - 1.0).abs() < 1e-6);
            assert!(close(quat_log(q), v, 1e-5), "{v} -> {q} -> {}", quat_log(q));
        }
        let axis_angle = Vec3::new(0.0, 2.5, 0.0);
        let q = quat_from_scaled_angle_axis(axis_angle);
        assert!(same_rotation(q, Quat::from_rotation_y(2.5), 1e-6));
        assert!(close(quat_to_scaled_angle_axis(q), axis_angle, 1e-5));
        // past half a turn the shorter arc goes the other way
        let q = Quat::from_rotation_y(3.5);
        assert!(close(quat_to_scaled_angle_axis(q), Vec3::new(0.0, 3.5 - std::f32::consts::TAU, 0.0), 1e-4));
    }

    #[test]
    fn angular_velocity_turns_one_rotation_into_the_other() {
        let from = Quat::from_rotation_x(0.3);
        let to = Quat::from_rotation_z(0.2) * from;
        let w = angular_velocity(from, to, 0.5);
        assert!(close(w, Vec3::new(0.0, 0.0, 0.4), 1e-5), "{w}");
        assert!(same_rotation(quat_from_scaled_angle_axis(w * 0.5) * from, to, 1e-6));
    }
}
