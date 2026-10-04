mod vector;
mod matrix;

pub use vector::{Vec2, Vec3, Vec4};
pub use matrix::Mat4;

/// A deterministic pseudo-random number in 0..1 for `i` (an integer hash, 10 007 steps): the same
/// scatter of trees, rocks or tints on every run and platform.
pub fn hash01(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

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
