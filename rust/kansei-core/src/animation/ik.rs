//! Two-joint IK and foot locking.
//!
//! The foot lock follows Daniel Holden's "Inverse Kinematics & Foot Locking"
//! (<https://theorangeduck.com/page/inverse-kinematics-foot-locking>): while the animation says a
//! foot is planted, pin it where it touched down; release it when the contact ends or the
//! animated foot strays past a radius, and let the difference fade with a spring.

use glam::{Quat, Vec3};

use super::springs::decay_spring_damper_exact;
use super::{Pose, Skeleton, Transform};

/// Bend a three-joint chain (`upper` → `middle` → `end`, e.g. thigh, knee, foot) so `end` reaches
/// `target` (model space), keeping the chain's plane when it can and `end`'s model rotation.
/// Rewrites `pose`'s local rotations of `upper` and `middle`; `model` must be `pose`'s model
/// transforms and is updated for the three joints.
pub fn two_joint_ik(skeleton: &Skeleton, pose: &mut Pose, model: &mut [Transform], upper: usize, middle: usize, end: usize, target: Vec3) {
    let (a, b, c) = (model[upper].translation, model[middle].translation, model[end].translation);
    let end_rotation = model[end].rotation;
    let eps = 1e-5;
    let lab = (b - a).length();
    let lcb = (c - b).length();
    let lat = (target - a).length().clamp(eps, (lab + lcb) * 0.9999);
    if lab < eps || lcb < eps {
        return;
    }
    let angle = |x: Vec3, y: Vec3| x.normalize_or_zero().dot(y.normalize_or_zero()).clamp(-1.0, 1.0).acos();
    // the current angles at the upper and middle joints, and the ones that reach `target`
    let ac_ab_0 = angle(c - a, b - a);
    let ba_bc_0 = angle(a - b, c - b);
    let ac_at_0 = angle(c - a, target - a);
    let ac_ab_1 = ((lcb * lcb - lab * lab - lat * lat) / (-2.0 * lab * lat)).clamp(-1.0, 1.0).acos();
    let ba_bc_1 = ((lat * lat - lab * lab - lcb * lcb) / (-2.0 * lab * lcb)).clamp(-1.0, 1.0).acos();
    // bend about the chain's normal; a straight chain bends about any perpendicular
    let mut bend_axis = (c - a).cross(b - a);
    if bend_axis.length_squared() < 1e-10 {
        bend_axis = (c - a).any_orthonormal_vector();
    }
    let bend_axis = bend_axis.normalize();
    let swing_axis = (c - a).cross(target - a);
    let (ua, ub) = (model[upper].rotation, model[middle].rotation);
    let r0 = Quat::from_axis_angle(ua.conjugate() * bend_axis, ac_ab_1 - ac_ab_0);
    let r1 = Quat::from_axis_angle(ub.conjugate() * bend_axis, ba_bc_1 - ba_bc_0);
    let r2 = if swing_axis.length_squared() > 1e-10 { Quat::from_axis_angle(ua.conjugate() * swing_axis.normalize(), ac_at_0) } else { Quat::IDENTITY };
    // bend first, then swing onto the target: in the joint's frame the swing comes last
    pose.local[upper].rotation = (pose.local[upper].rotation * (r2 * r0)).normalize();
    pose.local[middle].rotation = (pose.local[middle].rotation * r1).normalize();
    // the chain's model transforms again, with the end keeping its model rotation
    let parent = |j: usize, model: &[Transform]| skeleton.parents[j].map_or(Transform::IDENTITY, |p| model[p]);
    model[upper] = parent(upper, model).mul(&pose.local[upper]);
    model[middle] = parent(middle, model).mul(&pose.local[middle]);
    let end_parent = parent(end, model);
    pose.local[end].rotation = (end_parent.rotation.conjugate() * end_rotation).normalize();
    model[end] = end_parent.mul(&pose.local[end]);
}

/// A foot pinned to where it touched down while it is planted.
#[derive(Debug, Clone, Default)]
pub struct FootLock {
    locked: bool,
    /// Where the foot is pinned while locked.
    position: Vec3,
    /// Last frame's contact, to lock on a new one only.
    contact: bool,
    /// Added to the output, fading: it keeps the output continuous when the lock lets go.
    offset: Vec3,
    offset_velocity: Vec3,
}

impl FootLock {
    /// The foot's target this frame (world space) from where the animation puts it and whether
    /// it is planted. A new contact pins the foot there; the pin lets go when the contact ends or
    /// the animated foot strays past `unlock_radius`, and the output then fades back onto the
    /// animation with `halflife`.
    pub fn update(&mut self, animated: Vec3, contact: bool, unlock_radius: f32, halflife: f32, dt: f32) -> Vec3 {
        if self.locked && (!contact || animated.distance(self.position) > unlock_radius) {
            self.locked = false;
            // the output was the pin: carry the difference over, to fade
            self.offset += self.position - animated;
        }
        if !self.locked && contact && !self.contact {
            self.locked = true;
            self.position = animated;
        }
        self.contact = contact;
        decay_spring_damper_exact(&mut self.offset, &mut self.offset_velocity, halflife, dt);
        (if self.locked { self.position } else { animated }) + self.offset
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Release the foot and forget its history (after a teleport).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A leg: hip at 1 m, knee 0.5 m below, ankle 0.5 m below that, slightly bent forward.
    fn leg() -> (Skeleton, Pose) {
        let skeleton = Skeleton::new(
            vec!["hip".into(), "knee".into(), "ankle".into()],
            vec![None, Some(0), Some(1)],
            vec![
                Transform::from_translation_rotation(Vec3::new(0.0, 1.0, 0.0), Quat::from_rotation_x(-0.2)),
                Transform::from_translation_rotation(Vec3::new(0.0, -0.5, 0.0), Quat::from_rotation_x(0.4)),
                Transform::from_translation_rotation(Vec3::new(0.0, -0.5, 0.0), Quat::from_rotation_x(-0.2)),
            ],
        );
        let pose = Pose::rest(&skeleton);
        (skeleton, pose)
    }

    #[test]
    fn two_joint_ik_reaches_the_target_and_keeps_the_foot_rotation() {
        let (skeleton, mut pose) = leg();
        let mut model = pose.model(&skeleton);
        let foot_rotation = model[2].rotation;
        let lengths = (model[1].translation.distance(model[0].translation), model[2].translation.distance(model[1].translation));
        for target in [Vec3::new(0.1, 0.2, 0.15), Vec3::new(-0.2, 0.4, 0.3), Vec3::new(0.0, 0.1, -0.2)] {
            two_joint_ik(&skeleton, &mut pose, &mut model, 0, 1, 2, target);
            // the model transforms are the pose's
            let again = pose.model(&skeleton);
            for (x, y) in model.iter().zip(&again) {
                assert!(x.translation.abs_diff_eq(y.translation, 1e-5));
            }
            assert!(model[2].translation.abs_diff_eq(target, 1e-4), "{} vs {target}", model[2].translation);
            assert!(model[2].rotation.dot(foot_rotation).abs() > 1.0 - 1e-5);
            // bones keep their lengths
            assert!((model[1].translation.distance(model[0].translation) - lengths.0).abs() < 1e-5);
            assert!((model[2].translation.distance(model[1].translation) - lengths.1).abs() < 1e-5);
        }
        // out of reach: the leg straightens toward the target
        two_joint_ik(&skeleton, &mut pose, &mut model, 0, 1, 2, Vec3::new(0.0, -2.0, 0.0));
        assert!((model[2].translation - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-2, "{}", model[2].translation);
    }

    #[test]
    fn a_planted_foot_stays_until_released_then_blends_back() {
        let mut lock = FootLock::default();
        let dt = 1.0 / 60.0;
        // swinging, then touching down at x = 0.3
        assert_eq!(lock.update(Vec3::new(0.0, 0.1, 0.0), false, 0.2, 0.1, dt), Vec3::new(0.0, 0.1, 0.0));
        let planted = lock.update(Vec3::new(0.3, 0.0, 0.0), true, 0.2, 0.1, dt);
        assert!(lock.is_locked());
        // the animated foot slides 10 cm while planted: the output stays
        let still = lock.update(Vec3::new(0.4, 0.0, 0.0), true, 0.2, 0.1, dt);
        assert_eq!(still, planted);
        // past the radius it lets go, starting from where it was, then catches up
        let released = lock.update(Vec3::new(0.6, 0.0, 0.0), true, 0.2, 0.1, dt);
        assert!(!lock.is_locked());
        assert!(released.distance(planted) < 0.05, "{released}");
        let mut out = released;
        for _ in 0..60 {
            out = lock.update(Vec3::new(0.6, 0.1, 0.0), false, 0.2, 0.1, dt);
        }
        assert!(out.distance(Vec3::new(0.6, 0.1, 0.0)) < 1e-3, "{out}");
    }
}
