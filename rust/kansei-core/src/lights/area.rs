use crate::math::Vec3;

pub struct AreaLight {
    pub position: Vec3,
    pub target: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub width: f32,
    pub height: f32,
    pub radius: f32,
    pub cast_shadow: bool,
}

impl AreaLight {
    pub fn new(position: Vec3, target: Vec3, color: Vec3, intensity: f32, width: f32, height: f32) -> Self {
        Self { position, target, color, intensity, width, height, radius: 50.0, cast_shadow: false }
    }

    pub fn direction(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }
}
