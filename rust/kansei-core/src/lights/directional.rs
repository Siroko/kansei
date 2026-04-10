use crate::math::Vec3;

pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cast_shadow: bool,
    pub volumetric: bool,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self { direction, color, intensity, cast_shadow: false, volumetric: true }
    }

    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }
}
