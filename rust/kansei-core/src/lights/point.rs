use crate::math::Vec3;

pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub cast_shadow: bool,
    pub volumetric: bool,
}

impl PointLight {
    pub fn new(position: Vec3, color: Vec3, intensity: f32, radius: f32) -> Self {
        Self { position, color, intensity, radius, cast_shadow: false, volumetric: true }
    }

    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }
}
