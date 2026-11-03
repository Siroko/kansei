//! The mouse in the smoke: a ray from the camera through the cursor, and the push it gives the gas.

use crate::math::Vec3;

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
}
