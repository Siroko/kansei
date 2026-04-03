use bytemuck::{Pod, Zeroable};
use super::Vec3;

/// 4x4 matrix — column-major layout matching WebGPU/glam.
/// Wraps glam::Mat4 for computation, stores as [f32; 16] for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Mat4 {
    pub data: [f32; 16],
}

impl Mat4 {
    pub fn identity() -> Self {
        Self::from(glam::Mat4::IDENTITY)
    }

    pub fn perspective(fov_radians: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self::from(glam::Mat4::perspective_rh(fov_radians, aspect, near, far))
    }

    pub fn look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Self {
        Self::from(glam::Mat4::look_at_rh(eye.to_glam(), target.to_glam(), up.to_glam()))
    }

    pub fn translate(x: f32, y: f32, z: f32) -> Self {
        Self::from(glam::Mat4::from_translation(glam::Vec3::new(x, y, z)))
    }

    pub fn inverse(&self) -> Self {
        Self::from(self.to_glam().inverse())
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::from(self.to_glam() * other.to_glam())
    }

    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        let m = self.to_glam();
        let v = m.transform_point3(p.to_glam());
        Vec3::from(v)
    }

    pub fn to_glam(&self) -> glam::Mat4 {
        glam::Mat4::from_cols_array(&self.data)
    }

    /// Raw f32 slice for GPU upload.
    pub fn as_slice(&self) -> &[f32; 16] {
        &self.data
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<glam::Mat4> for Mat4 {
    fn from(m: glam::Mat4) -> Self {
        Self { data: m.to_cols_array() }
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from(self.to_glam() * rhs.to_glam())
    }
}
