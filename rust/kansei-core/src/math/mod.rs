mod vector;
mod matrix;

pub use vector::{Vec2, Vec3, Vec4};
pub use matrix::Mat4;

/// A single f32 uniform value, matching the TS `Float` class.
#[derive(Debug, Clone, Copy)]
pub struct Float(pub f32);

impl Float {
    pub fn new(value: f32) -> Self {
        Self(value)
    }
}

impl From<f32> for Float {
    fn from(v: f32) -> Self {
        Self(v)
    }
}
