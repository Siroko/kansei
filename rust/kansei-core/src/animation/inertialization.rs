//! Inertialization: switch animations instantly, and hide the jump by adding the difference
//! between the old and the new pose as an offset that decays to nothing with a spring.
//!
//! After David Bollo, "Inertialization: High-Performance Animation Transitions in Gears of War"
//! (GDC 2018), in the spring form of Daniel Holden's "Spring-It-On" and his MIT-licensed
//! Motion-Matching code. Only the new animation is evaluated each frame.

use glam::{Quat, Vec3};

use super::springs::{decay_spring_damper_exact, decay_spring_damper_exact_quat};
use super::{angular_velocity, quat_abs, Pose, Transform};

/// Per-joint offsets (and their velocities) between what was showing and what now plays.
#[derive(Debug, Clone, Default)]
pub struct Inertializer {
    position: Vec<Vec3>,
    velocity: Vec<Vec3>,
    rotation: Vec<Quat>,
    angular: Vec<Vec3>,
}

impl Inertializer {
    pub fn new(joints: usize) -> Self {
        Self { position: vec![Vec3::ZERO; joints], velocity: vec![Vec3::ZERO; joints], rotation: vec![Quat::IDENTITY; joints], angular: vec![Vec3::ZERO; joints] }
    }

    /// Switch from the source animation (its local pose and joint velocities now) to the
    /// destination: the offsets become what is showing (source plus the offsets still decaying)
    /// minus the destination, so the output does not jump.
    #[allow(clippy::too_many_arguments)]
    pub fn transition(&mut self, source: &Pose, source_linear: &[Vec3], source_angular: &[Vec3], destination: &Pose, destination_linear: &[Vec3], destination_angular: &[Vec3]) {
        for j in 0..self.position.len() {
            let (s, d) = (&source.local[j], &destination.local[j]);
            self.position[j] = (s.translation + self.position[j]) - d.translation;
            self.velocity[j] = (source_linear[j] + self.velocity[j]) - destination_linear[j];
            self.rotation[j] = quat_abs((self.rotation[j] * s.rotation) * d.rotation.conjugate());
            self.angular[j] = (source_angular[j] + self.angular[j]) - destination_angular[j];
        }
    }

    /// Decay the offsets by `dt` (half gone every `halflife` seconds) and apply them to the pose
    /// that plays, `pose`, in place.
    pub fn update(&mut self, pose: &mut Pose, halflife: f32, dt: f32) {
        for j in 0..self.position.len() {
            decay_spring_damper_exact(&mut self.position[j], &mut self.velocity[j], halflife, dt);
            decay_spring_damper_exact_quat(&mut self.rotation[j], &mut self.angular[j], halflife, dt);
            let t = &mut pose.local[j];
            t.translation += self.position[j];
            t.rotation = (self.rotation[j] * t.rotation).normalize();
        }
    }

    /// Forget the offsets (after a teleport).
    pub fn reset(&mut self) {
        *self = Self::new(self.position.len());
    }

    /// The largest rotation offset left, in radians (for debugging and tests).
    pub fn largest_angle(&self) -> f32 {
        self.rotation.iter().map(|q| 2.0 * quat_abs(*q).w.clamp(-1.0, 1.0).acos()).fold(0.0, f32::max)
    }
}

/// Joint velocities between two poses `dt` apart: (linear, angular) per joint.
pub fn pose_velocities(from: &[Transform], to: &[Transform], dt: f32) -> (Vec<Vec3>, Vec<Vec3>) {
    from.iter().zip(to).map(|(a, b)| ((b.translation - a.translation) / dt, angular_velocity(a.rotation, b.rotation, dt))).unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(angle: f32, x: f32) -> Pose {
        Pose { local: vec![Transform::from_translation_rotation(Vec3::new(x, 0.0, 0.0), Quat::from_rotation_y(angle)); 2] }
    }

    #[test]
    fn the_output_does_not_jump_and_then_settles_on_the_destination() {
        let (source, destination) = (pose(0.5, 1.0), pose(-0.3, -0.5));
        let zero = vec![Vec3::ZERO; 2];
        let mut inertializer = Inertializer::new(2);
        inertializer.transition(&source, &zero, &zero, &destination, &zero, &zero);
        // right after the switch the output is the source
        let mut out = destination.clone();
        inertializer.update(&mut out, 0.1, 0.0);
        assert!(out.local[0].translation.abs_diff_eq(source.local[0].translation, 1e-5));
        assert!(out.local[0].rotation.dot(source.local[0].rotation).abs() > 1.0 - 1e-6);
        // it moves toward the destination without overshooting, and gets there
        let mut last = f32::MAX;
        for _ in 0..60 {
            let mut out = destination.clone();
            inertializer.update(&mut out, 0.1, 1.0 / 60.0);
            let gap = out.local[0].translation.distance(destination.local[0].translation);
            assert!(gap <= last + 1e-6, "{gap} after {last}");
            last = gap;
        }
        assert!(last < 1e-3 && inertializer.largest_angle() < 1e-3, "{last} {}", inertializer.largest_angle());
    }

    #[test]
    fn a_second_transition_keeps_the_output_continuous() {
        let zero = vec![Vec3::ZERO; 2];
        let mut inertializer = Inertializer::new(2);
        inertializer.transition(&pose(0.0, 0.0), &zero, &zero, &pose(1.0, 2.0), &zero, &zero);
        let mut shown = pose(1.0, 2.0);
        inertializer.update(&mut shown, 0.2, 0.05);
        // switch again mid-blend: the next frame starts from what was showing
        inertializer.transition(&pose(1.0, 2.0), &zero, &zero, &pose(-1.0, -3.0), &zero, &zero);
        let mut next = pose(-1.0, -3.0);
        inertializer.update(&mut next, 0.2, 0.0);
        assert!(next.local[1].translation.abs_diff_eq(shown.local[1].translation, 1e-4), "{} vs {}", next.local[1].translation, shown.local[1].translation);
        assert!(next.local[1].rotation.dot(shown.local[1].rotation).abs() > 1.0 - 1e-5);
    }

    #[test]
    fn velocities_carry_across_the_switch() {
        // the source moves at +1 m/s, the destination stands: the output keeps moving for a while
        let mut inertializer = Inertializer::new(1);
        let still = Pose { local: vec![Transform::IDENTITY] };
        inertializer.transition(&still, &[Vec3::X], &[Vec3::ZERO], &still, &[Vec3::ZERO], &[Vec3::ZERO]);
        let mut out = still.clone();
        inertializer.update(&mut out, 0.2, 1.0 / 60.0);
        assert!(out.local[0].translation.x > 0.01, "{}", out.local[0].translation);
        let (linear, angular) = pose_velocities(&[Transform::IDENTITY], &[Transform::from_translation_rotation(Vec3::X, Quat::from_rotation_z(0.1))], 0.5);
        assert!(linear[0].abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-6) && angular[0].abs_diff_eq(Vec3::new(0.0, 0.0, 0.2), 1e-5));
    }
}
