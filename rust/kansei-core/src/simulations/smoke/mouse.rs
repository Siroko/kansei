//! The mouse in the smoke: a ray from the camera through the cursor, and the push it gives the gas.
//! Fluid drags the gas near the ray toward the push (the solver carries it on as a wake); Curl
//! noise adds the push to the velocity it advects by (the gas keeps its trail); Waves keeps the
//! last pushes (`SmokePushes`), which displace the plume where the march samples it and relax.

use bytemuck::{Pod, Zeroable};

use super::MAX_SMOKE_PUSHES;
use crate::cameras::Camera;
use crate::math::{Vec2, Vec3};

/// The mouse as the smoke sees it: a world-space ray (`origin`, unit `direction`) and the push
/// it gives the gas within `radius` of it, m/s.
#[derive(Debug, Clone, Copy)]
pub struct SmokeMouse {
    pub origin: Vec3,
    pub direction: Vec3,
    pub push: Vec3,
    pub radius: f32,
}

impl SmokeMouse {
    /// No mouse: no push anywhere.
    pub const NONE: SmokeMouse = SmokeMouse { origin: Vec3::ZERO, direction: Vec3 { x: 0.0, y: 0.0, z: -1.0 }, push: Vec3::ZERO, radius: 0.0 };

    /// Whether it pushes the gas at all.
    pub fn is_active(&self) -> bool {
        self.radius > 0.0 && (self.push.x != 0.0 || self.push.y != 0.0 || self.push.z != 0.0)
    }

    /// The mouse from `MouseVectors`: `position` and `motion` are its `position` and `direction`
    /// (NDC, y down; direction = previous - current, over `dt` seconds). The ray runs from the
    /// camera through the cursor; the push is the cursor's velocity in m/s at view depth `depth`
    /// (where the smoke is), along the camera's right and up, times `strength`; `radius` metres
    /// round the ray.
    pub fn from_screen(camera: &Camera, position: Vec2, motion: Vec2, dt: f32, strength: f32, radius: f32, depth: f32) -> SmokeMouse {
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let inv = view_proj.inverse();
        let inv_view = camera.inverse_view_matrix.to_glam();
        let eye = inv_view.w_axis.truncate();
        let (right, up) = (inv_view.x_axis.truncate().normalize(), inv_view.y_axis.truncate().normalize());
        let far = inv.project_point3(glam::Vec3::new(position.x, -position.y, 1.0));
        let direction = (far - eye).normalize();
        // the cursor's velocity in NDC (y up), then in metres at `depth`
        let ndc_velocity = glam::Vec2::new(-motion.x, motion.y) / dt.max(1e-4);
        let half_height = depth * (camera.fov.to_radians() * 0.5).tan();
        let push = (right * (ndc_velocity.x * half_height * camera.aspect) + up * (ndc_velocity.y * half_height)) * strength;
        let k = |v: glam::Vec3| Vec3::new(v.x, v.y, v.z);
        SmokeMouse { origin: k(eye), direction: k(direction), push: k(push), radius }
    }
}

/// WGSL `SmokePush` (volumetric_smoke_march.wgsl): one recent push the Waves level relaxes from.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub(crate) struct SmokePushGpu {
    pub(crate) origin: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) direction: [f32; 3],
    /// seconds since it ended
    pub(crate) age: f32,
    /// the displacement it gave, metres
    pub(crate) push: [f32; 3],
    pub(crate) _pad: f32,
}

/// How much of a push reaches a point `distance` from its ray: 1 on the ray, 0 at `radius`.
pub(crate) fn falloff(distance: f32, radius: f32) -> f32 {
    let r = (distance / radius.max(1e-6)).clamp(0.0, 1.0);
    let f = 1.0 - r * r;
    f * f
}

/// The Waves level's recent pushes, a ring of `MAX_SMOKE_PUSHES`: each the displacement the mouse
/// gave over a stretch of a stir (a new one every 0.4 relax times), relaxing back over
/// `relax_time`; the oldest is dropped.
#[derive(Default, Clone)]
pub(crate) struct SmokePushes {
    pushes: Vec<SmokePushGpu>,
}

impl SmokePushes {
    /// Record `dt` seconds of the mouse pushing.
    pub(crate) fn record(&mut self, mouse: &SmokeMouse, dt: f32, relax_time: f32) {
        if !mouse.is_active() {
            return;
        }
        let shift = mouse.push * dt;
        let (o, d) = ([mouse.origin.x, mouse.origin.y, mouse.origin.z], [mouse.direction.x, mouse.direction.y, mouse.direction.z]);
        if let Some(last) = self.pushes.last_mut().filter(|p| p.age < 0.4 * relax_time) {
            // the same stretch of the stir: its displacement grows, its ray follows the mouse
            last.push = [last.push[0] + shift.x, last.push[1] + shift.y, last.push[2] + shift.z];
            (last.origin, last.direction, last.radius) = (o, d, mouse.radius);
            return;
        }
        if self.pushes.len() == MAX_SMOKE_PUSHES {
            self.pushes.remove(0);
        }
        self.pushes.push(SmokePushGpu { origin: o, radius: mouse.radius, direction: d, age: 0.0, push: [shift.x, shift.y, shift.z], _pad: 0.0 });
    }

    /// Age the pushes by `dt` seconds.
    pub(crate) fn age(&mut self, dt: f32) {
        for p in &mut self.pushes {
            p.age += dt;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.pushes.len()
    }

    pub(crate) fn gpu(&self) -> &[SmokePushGpu] {
        &self.pushes
    }

    /// Where the pushes have moved the gas at `p`, metres (the march's smokePushDisplacement).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn displacement(&self, p: Vec3, relax_time: f32) -> Vec3 {
        let p = glam::Vec3::new(p.x, p.y, p.z);
        let mut total = glam::Vec3::ZERO;
        for push in &self.pushes {
            let (o, d) = (glam::Vec3::from(push.origin), glam::Vec3::from(push.direction));
            let distance = (p - o).cross(d).length();
            total += glam::Vec3::from(push.push) * falloff(distance, push.radius) * (-push.age / relax_time.max(1e-3)).exp();
        }
        Vec3::new(total.x, total.y, total.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::Camera;
    use crate::math::Vec2;

    fn camera() -> Camera {
        let mut cam = Camera::new(50.0, 0.1, 100.0, 16.0 / 9.0);
        cam.set_position(0.0, 1.0, 3.0);
        cam.look_at(&Vec3::new(0.0, 1.0, 0.0));
        cam.update_view_matrix();
        cam
    }

    #[test]
    fn the_mouse_maps_to_a_ray_and_a_push_in_the_scene() {
        let cam = camera();
        let view_proj = cam.projection_matrix.to_glam() * cam.view_matrix.to_glam();
        let inv_view = cam.inverse_view_matrix.to_glam();
        let (right, up) = (inv_view.x_axis.truncate().normalize(), inv_view.y_axis.truncate().normalize());
        let m = SmokeMouse::from_screen(&cam, Vec2::new(0.2, -0.1), Vec2::new(-0.1, 0.0), 1.0 / 60.0, 1.0, 0.08, 2.0);
        let g = |v: Vec3| glam::Vec3::new(v.x, v.y, v.z);
        // NDC y up
        let ndc = view_proj.project_point3(g(m.origin) + g(m.direction) * 5.0);
        assert!((ndc.x - 0.2).abs() < 1e-4 && (ndc.y - 0.1).abs() < 1e-4, "{ndc}");
        let want = right * (0.1 * 2.0 * 25f32.to_radians().tan() * (16.0 / 9.0) * 60.0);
        assert!((g(m.push) - want).length() < 1e-3 * want.length(), "{:?} vs {want}", m.push);
        assert_eq!(m.radius, 0.08);
        let upward = SmokeMouse::from_screen(&cam, Vec2::ZERO, Vec2::new(0.0, 0.1), 1.0 / 60.0, 1.0, 0.08, 2.0);
        assert!(g(upward.push).dot(up) > 0.0, "{:?}", upward.push);
    }

    #[test]
    fn waves_pushes_relax() {
        let relax = 1.5;
        let mouse = SmokeMouse { origin: Vec3::new(0.0, 1.0, 1.0), direction: Vec3::new(0.0, 0.0, -1.0), push: Vec3::new(1.0, 0.0, 0.0), radius: 0.1 };
        let mut pushes = SmokePushes::default();
        pushes.record(&mouse, 1.0 / 60.0, relax);
        let p = Vec3::new(0.0, 1.0, 0.0);
        let at0 = pushes.displacement(p, relax).length();
        assert!(at0 > 0.0);
        pushes.age(3.0 * relax);
        let later = pushes.displacement(p, relax).length();
        assert!(later < 0.05 * at0, "{later} vs {at0}");
        // pushes far enough apart in time each get a slot; a ninth drops the oldest
        let mut pushes = SmokePushes::default();
        for k in 0..9 {
            let mut m = mouse;
            m.origin.x = k as f32;
            pushes.record(&m, 1.0 / 60.0, relax);
            pushes.age(relax);
        }
        assert_eq!(pushes.len(), MAX_SMOKE_PUSHES);
        assert_eq!(pushes.gpu()[0].origin[0], 1.0, "the oldest went first");
    }
}
