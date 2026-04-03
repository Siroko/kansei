use crate::math::Vec3;
use crate::cameras::Camera;

/// Orbit camera controls (mirrors TS CameraControls).
pub struct CameraControls {
    pub target: Vec3,
    pub radius: f32,
    azimuth: f32,
    elevation: f32,
}

impl CameraControls {
    pub fn new(target: Vec3, radius: f32) -> Self {
        Self { target, radius, azimuth: 0.0, elevation: 0.0 }
    }

    pub fn update(&mut self, camera: &mut Camera, _dt: f32) {
        let x = self.target.x + self.radius * self.azimuth.sin() * self.elevation.cos();
        let y = self.target.y + self.radius * self.elevation.sin();
        let z = self.target.z + self.radius * self.azimuth.cos() * self.elevation.cos();
        camera.set_position(x, y, z);
        camera.look_at(&self.target);
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.azimuth += dx;
        self.elevation = (self.elevation + dy).clamp(-1.5, 1.5);
    }
}
