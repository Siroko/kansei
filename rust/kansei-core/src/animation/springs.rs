//! Critically damped springs, in closed form (exact for any time step).
//!
//! After Daniel Holden's "Spring-It-On: The Game Developer's Spring-Roll-Call"
//! (<https://theorangeduck.com/page/spring-roll-call>) and his MIT-licensed Motion-Matching
//! reference code (<https://github.com/orangeduck/Motion-Matching>). A spring's stiffness is given
//! as a half-life: the time to cover half the distance to its goal.

use glam::{Quat, Vec3};

use super::{quat_abs, quat_from_scaled_angle_axis, quat_to_scaled_angle_axis};

const LN2: f32 = std::f32::consts::LN_2;

/// `exp(-x)`. (Holden's rational approximation is close for a frame's step but several times
/// too large a second ahead, where trajectory prediction evaluates the springs.)
pub fn negexp(x: f32) -> f32 {
    (-x).exp()
}

/// Damping of a critically damped spring with this half-life.
pub fn halflife_to_damping(halflife: f32) -> f32 {
    (4.0 * LN2) / (halflife + 1e-5)
}

/// Move `x` toward `goal`, covering half the distance every `halflife` seconds (no velocity).
pub fn damper_exact(x: Vec3, goal: Vec3, halflife: f32, dt: f32) -> Vec3 {
    x.lerp(goal, 1.0 - negexp((LN2 * dt) / (halflife + 1e-5)))
}

/// A critically damped spring from `x` (velocity `v`) toward `goal`, advanced by `dt`.
pub fn spring_damper_exact(x: &mut Vec3, v: &mut Vec3, goal: Vec3, halflife: f32, dt: f32) {
    let y = halflife_to_damping(halflife) / 2.0;
    let j0 = *x - goal;
    let j1 = *v + j0 * y;
    let eydt = negexp(y * dt);
    *x = eydt * (j0 + j1 * dt) + goal;
    *v = eydt * (*v - j1 * y * dt);
}

/// `spring_damper_exact` toward zero: an offset (and its velocity) fading away.
pub fn decay_spring_damper_exact(x: &mut Vec3, v: &mut Vec3, halflife: f32, dt: f32) {
    let y = halflife_to_damping(halflife) / 2.0;
    let j1 = *v + *x * y;
    let eydt = negexp(y * dt);
    *x = eydt * (*x + j1 * dt);
    *v = eydt * (*v - j1 * y * dt);
}

/// `spring_damper_exact` for a rotation `q` with angular velocity `w` (radians per second).
pub fn spring_damper_exact_quat(q: &mut Quat, w: &mut Vec3, goal: Quat, halflife: f32, dt: f32) {
    let y = halflife_to_damping(halflife) / 2.0;
    let j0 = quat_to_scaled_angle_axis(quat_abs(*q * goal.conjugate()));
    let j1 = *w + j0 * y;
    let eydt = negexp(y * dt);
    *q = (quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt)) * goal).normalize();
    *w = eydt * (*w - j1 * y * dt);
}

/// `decay_spring_damper_exact` for a rotation offset: `q` fades to the identity.
pub fn decay_spring_damper_exact_quat(q: &mut Quat, w: &mut Vec3, halflife: f32, dt: f32) {
    let y = halflife_to_damping(halflife) / 2.0;
    let j0 = quat_to_scaled_angle_axis(quat_abs(*q));
    let j1 = *w + j0 * y;
    let eydt = negexp(y * dt);
    *q = quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt)).normalize();
    *w = eydt * (*w - j1 * y * dt);
}

/// A character's velocity as a critically damped spring toward `goal_velocity`, integrated into
/// its position: `x` (position), `v` (velocity), `a` (acceleration) after `dt`.
pub fn spring_character_update(x: &mut Vec3, v: &mut Vec3, a: &mut Vec3, goal_velocity: Vec3, halflife: f32, dt: f32) {
    let y = halflife_to_damping(halflife) / 2.0;
    let j0 = *v - goal_velocity;
    let j1 = *a + j0 * y;
    let eydt = negexp(y * dt);
    *x = eydt * ((-j1 / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + goal_velocity * dt + *x;
    *v = eydt * (j0 + j1 * dt) + goal_velocity;
    *a = eydt * (*a - j1 * y * dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_damper_covers_half_the_way_in_a_halflife() {
        let x = damper_exact(Vec3::ZERO, Vec3::X, 0.2, 0.2);
        assert!((x.x - 0.5).abs() < 1e-3, "{x}");
    }

    #[test]
    fn springs_settle_on_their_goal_and_are_step_independent() {
        let run = |steps: usize| {
            let (mut x, mut v) = (Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.0, 3.0, 0.0));
            let dt = 2.0 / steps as f32;
            for _ in 0..steps {
                spring_damper_exact(&mut x, &mut v, Vec3::new(4.0, 0.0, 0.0), 0.3, dt);
            }
            (x, v)
        };
        let (a, va) = run(10);
        let (b, vb) = run(1000);
        assert!(a.abs_diff_eq(b, 1e-3), "{a} vs {b}");
        assert!(va.abs_diff_eq(vb, 1e-3));
        assert!(a.abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1e-2), "{a}");
        // a decaying offset vanishes
        let (mut x, mut v) = (Vec3::new(0.3, 0.0, -0.2), Vec3::ZERO);
        decay_spring_damper_exact(&mut x, &mut v, 0.1, 1.0);
        assert!(x.length() < 1e-3 && v.length() < 1e-2);
    }

    #[test]
    fn a_rotation_spring_turns_to_its_goal() {
        let (mut q, mut w) = (Quat::IDENTITY, Vec3::ZERO);
        let goal = Quat::from_rotation_y(2.0);
        let mut last_angle = 0.0;
        for _ in 0..120 {
            spring_damper_exact_quat(&mut q, &mut w, goal, 0.2, 1.0 / 60.0);
            let angle = quat_to_scaled_angle_axis(q).y;
            // critically damped: no overshoot
            assert!(angle >= last_angle - 1e-4 && angle <= 2.0 + 1e-3, "{angle}");
            last_angle = angle;
        }
        assert!(q.dot(goal).abs() > 0.9999);
        let (mut q, mut w) = (Quat::from_rotation_x(0.5), Vec3::ZERO);
        decay_spring_damper_exact_quat(&mut q, &mut w, 0.1, 1.0);
        assert!(q.dot(Quat::IDENTITY).abs() > 0.99999);
    }

    #[test]
    fn a_character_spring_reaches_its_goal_velocity_and_integrates_position() {
        let (mut x, mut v, mut a) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        let goal = Vec3::new(0.0, 0.0, 2.0);
        let dt = 1.0 / 60.0;
        let mut integrated = Vec3::ZERO;
        for _ in 0..180 {
            let before = v;
            spring_character_update(&mut x, &mut v, &mut a, goal, 0.25, dt);
            integrated += (before + v) * 0.5 * dt;
        }
        assert!(v.abs_diff_eq(goal, 1e-3), "{v}");
        assert!(a.length() < 1e-2);
        // the closed-form position is the velocity's integral
        assert!(x.abs_diff_eq(integrated, 1e-3), "{x} vs {integrated}");
        // one big step lands where many small ones do
        let (mut x1, mut v1, mut a1) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        spring_character_update(&mut x1, &mut v1, &mut a1, goal, 0.25, 3.0);
        assert!(x1.abs_diff_eq(x, 1e-3), "{x1} vs {x}");
    }
}
